"""Record-array adapters for existing MolSeeq widgets and export formats."""

from collections import defaultdict, deque
from collections.abc import MutableMapping

import numpy as np
from molcore.staging import transact


class LocalisationViews(MutableMapping):
    def __init__(self, session):
        self.session = session

    def __getitem__(self, name):
        self.session.dataset(name)
        return _RoundLocalisations(self.session, name)

    def __iter__(self):
        return iter(self.session.images)

    def __len__(self):
        return len(self.session.images)

    def __setitem__(self, name, value):
        if not value:
            return
        for channel, entry in value.items():
            self[name][channel] = entry

    def __delitem__(self, name):
        for channel in self[name]:
            del self[name][channel]


class _RoundLocalisations(MutableMapping):
    def __init__(self, session, name):
        self.session, self.name = session, name

    def __getitem__(self, channel):
        if channel not in self.session.images[self.name]:
            raise KeyError(channel)
        return _ChannelLocalisations(self.session, self.name, channel)

    def __iter__(self):
        return iter(self.session.images[self.name])

    def __len__(self):
        return len(self.session.images[self.name])

    def __setitem__(self, channel, value):
        self[channel].update(value)

    def __delitem__(self, channel):
        self[channel].clear()


class _ChannelLocalisations(MutableMapping):
    def __init__(self, session, name, channel):
        self.session, self.name, self.channel = session, name, channel
        self.key = session.dataset(name).id, channel

    @property
    def _state(self):
        return self.session._localisation_state.setdefault(self.key, {})

    def _rows(self, dataset, dtype):
        rows = []
        ids = []
        for view, offset in self.session.channel_parts(
            self.name, self.channel
        ):
            query = dataset.localizations.query(
                acquisition_id=view.acquisition.id,
                channel=self.channel,
                space="raw",
            )
            values = np.empty(len(query["id"]), dtype=dtype)
            for field in dtype.names:
                if not len(values):
                    break
                if field == "frame":
                    values[field] = (
                        view.channel_indices(query["frame"]) + offset
                    )
                else:
                    values[field] = query[field]
            rows.append(values)
            ids.append(query["id"])
        return np.concatenate(rows).view(np.recarray), np.concatenate(ids)

    def __getitem__(self, key):
        if key == "localisations":
            dtype = self._state["_dtype"]
            return self._rows(self.session.dataset(self.name), dtype)[0]
        return self._state[key]

    def __setitem__(self, key, value):
        if key == "localisations":
            self._replace(value)
        else:
            self._state[key] = value

    def _replace(self, values):
        dataset = self.session.dataset(self.name)
        dtype = self._state.get("_dtype")
        rows = np.asarray(values)
        if not rows.size and rows.dtype.names is None:
            rows = np.empty(
                0, dtype=dtype or [("frame", "u4"), ("x", "f4"), ("y", "f4")]
            )
        if rows.ndim != 1 or not {"frame", "x", "y"} <= set(
            rows.dtype.names or ()
        ):
            raise ValueError("Localisations require frame, x and y columns")
        frames = rows["frame"]
        if (
            frames.dtype.kind not in "ui"
            or np.any(frames < 0)
            or np.any(
                frames
                >= self.session.channel(self.name, self.channel).shape[0]
            )
        ):
            raise ValueError(
                "Localisation frames must belong to the logical channel"
            )

        def replace_rows(model):
            current, ids = self._rows(model, dtype or rows.dtype)
            # Reuse IDs for unchanged rows, including rows retained by filtering.
            available = defaultdict(deque)
            if current.dtype == rows.dtype:
                for row, identifier in zip(current, ids, strict=True):
                    available[row.tobytes()].append(identifier)
            added = np.ones(len(rows), dtype=bool)
            kept = set()
            for index, row in enumerate(rows):
                matches = available.get(row.tobytes())
                if matches:
                    kept.add(matches.popleft())
                    added[index] = False
            removed = np.asarray(
                [identifier for identifier in ids if identifier not in kept],
                dtype=np.uint64,
            )
            if removed.size:
                model.localizations.delete(removed)
            for view, offset in self.session.channel_parts(
                self.name, self.channel
            ):
                selected = rows[
                    added & (frames >= offset) & (frames < offset + len(view))
                ]
                if not len(selected):
                    continue
                columns = {
                    field: selected[field] for field in rows.dtype.names
                }
                columns["frame"] = view.camera_frames(
                    selected["frame"] - offset
                )
                model.localizations.add(
                    acquisition_id=view.acquisition.id,
                    channel=self.channel,
                    **columns,
                )

        transact(dataset, replace_rows)
        self._state["_dtype"] = rows.dtype
        self.session.revision += 1

    def __delitem__(self, key):
        if key == "localisations":
            self._replace([])
            del self._state["_dtype"]
        else:
            del self._state[key]

    def __iter__(self):
        for key in self._state:
            yield "localisations" if key == "_dtype" else key

    def __len__(self):
        return len(self._state)

    def copy(self):
        return dict(self)
