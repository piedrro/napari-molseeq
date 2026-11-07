"""MolSeeq's application bindings for MolCore acquisitions and datasets."""

from collections.abc import Mapping, MutableMapping
from copy import copy
from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from molcore import (
    Acquisition,
    AcquisitionSequence,
    ArraySource,
    Dataset,
    Experiment,
)
from molcore.files import image_acquisition, read_image
from molcore.staging import clone_experiment
from molcore.types.images import Source

from molseeq.model_views import LocalisationViews
from molseeq.trace_views import discard_trace_runs
from molseeq.box_views import (
    BoundingBoxView,
    LocalisationState,
    box_state,
    sync_box_definitions,
)


@dataclass(frozen=True)
class ImportSettings:
    mode: str
    layout: str = "donor-acceptor"
    first_excitation: str = "donor"
    limit: int | None = None
    pixel_size: float | None = None


@dataclass(frozen=True)
class Recording:
    name: str
    acquisition: Acquisition


class _ImportSource(Source):
    """Read a prefix of a camera movie in the legacy import pixel dtype."""

    def __init__(self, source, frame_count):
        self.source = source
        self.shape = (frame_count, *source.shape[1:])
        self.dtype = np.dtype(np.uint16)

    def _frame(self, index):
        return self.source[index].astype(self.dtype, copy=False)

    def close(self):
        self.source.close()


def read_recording(path, settings: ImportSettings) -> Recording:
    path = Path(path)
    source = read_image(path)
    mode = settings.mode.lower()
    layout = settings.layout.lower()
    first = settings.first_excitation.lower()
    if layout not in {"donor-acceptor", "acceptor-donor"} or first not in {
        "donor",
        "acceptor",
    }:
        raise ValueError("Unknown channel layout or first excitation")
    frame_count = source.shape[0]
    if settings.limit is not None:
        if settings.limit < 1:
            raise ValueError("Import limit must be positive")
        camera_limit = settings.limit * (2 if mode == "alex" else 1)
        frame_count = min(frame_count, camera_limit)
    if not frame_count:
        raise ValueError("The recording has no camera frames")
    source = _ImportSource(source, frame_count)
    if mode in {"fret", "alex"}:
        acquisition = image_acquisition(
            source,
            mode=mode,
            layout=layout.split("-")[0] + "-left",
            phase=int(first == "acceptor") if mode == "alex" else 0,
            pixel_size=settings.pixel_size,
        )
    elif mode in {
        "single channel",
        "donor",
        "acceptor",
        "dd",
        "da",
        "ad",
        "aa",
    }:
        name = "data" if mode == "single channel" else mode
        acquisition = image_acquisition(
            source, channel=name, pixel_size=settings.pixel_size
        )
    else:
        raise ValueError(f"Unknown import mode {settings.mode!r}")
    # Keep the UI's historical keys while retaining the camera geometry and phase.
    return Recording(
        path.name,
        Acquisition(
            acquisition.source,
            channel_specs={
                name.lower(): spec
                for name, spec in acquisition.channel_specs.items()
            },
            mode=acquisition.mode,
            excitation_sequence=acquisition.excitation_sequence,
            phase=acquisition.phase,
            pixel_size=acquisition.pixel_size,
            metadata={
                "source_path": str(path),
                "import_mode": mode,
                "channel_layout": settings.layout,
                "alex_first_frame": settings.first_excitation.capitalize(),
            },
        ),
    )


class ProjectSession:
    """Own one experiment and stable selector bindings for the widget."""

    def __init__(self):
        self.experiment = Experiment()
        self._names: dict[str, str] = {}
        self._bindings: dict[tuple[str, str], tuple[str, bool]] = {}
        self._working: dict[tuple[str, str], ArraySource] = {}
        self._display_pixels: dict[tuple[str, str], np.ndarray] = {}
        self._channel_state: dict[tuple[str, str], dict] = {}
        self._localisation_state: dict[tuple[str, str], dict] = {}
        self.images = _DatasetImages(self)
        self.localisations = LocalisationViews(self)
        self.revision = 0
        self.trace_revision = 0

    def dataset(self, name: str) -> Dataset:
        return self.experiment.datasets[self._names[name]]

    @property
    def bounding_boxes(self):
        return BoundingBoxView(self)

    @property
    def localisation_state(self):
        return LocalisationState(self)

    def bounding_box_ids(self):
        return tuple(
            box["id"] for box in box_state(self).get("definitions", [])
        )

    def bounding_box_token(self):
        return (
            box_state(self).get("revision", 0),
            tuple(
                (identifier, model.bounding_boxes.revision)
                for identifier, model in self.experiment.datasets.items()
            ),
        )

    def trace_token(self):
        return (
            self.trace_revision,
            tuple(
                (identifier, model.photon_sequences.revision)
                for identifier, model in self.experiment.datasets.items()
            ),
        )

    def binding(self, name, channel):
        return self._bindings[self.dataset(name).id, channel]

    def image_snapshot(self):
        """Borrow a fixed set of image sources for detached I/O."""
        snapshot = copy(self)
        snapshot.experiment = clone_experiment(self.experiment)
        snapshot._names = self._names.copy()
        snapshot._bindings = self._bindings.copy()
        snapshot._working = self._working.copy()
        return snapshot

    def channel(self, name: str, channel: str):
        dataset = self.dataset(name)
        identifier, is_sequence = self._bindings[dataset.id, channel]
        collection = dataset.sequences if is_sequence else dataset.acquisitions
        return collection[identifier].channels[channel]

    def working_source(self, name, channel):
        key = self.dataset(name).id, channel
        return self._working.get(key) or self.channel(name, channel)

    def channel_parts(self, name, channel):
        dataset = self.dataset(name)
        identifier, is_sequence = self._bindings[dataset.id, channel]
        acquisitions = (
            dataset.sequences[identifier].acquisitions
            if is_sequence
            else [dataset.acquisitions[identifier]]
        )
        offset = 0
        for acquisition in acquisitions:
            view = acquisition.channels[channel]
            yield view, offset
            offset += len(view)

    def camera_frames(self, name, mode):
        """Stream split recordings in their original camera order and geometry."""
        channels = (
            ("dd", "da", "ad", "aa")
            if mode == "alex"
            else ("donor", "acceptor")
        )
        parts = [
            list(self.channel_parts(name, channel)) for channel in channels
        ]
        sources = [self.working_source(name, channel) for channel in channels]
        groups = list(zip(*parts, strict=True))
        for group in groups:
            acquisition = group[0][0].acquisition
            if acquisition.mode != mode or any(
                view.acquisition.id != acquisition.id for view, _ in group
            ):
                raise ValueError(
                    f"{mode.upper()} export requires channels from the same camera recording"
                )
        for group in groups:
            acquisition = group[0][0].acquisition
            for frame in range(len(acquisition.source)):
                pixels = acquisition.source[frame].copy()
                for (view, offset), source in zip(group, sources, strict=True):
                    if not isinstance(source, ArraySource):
                        continue
                    index = int(np.searchsorted(view.source_frames, frame))
                    if (
                        index < len(view)
                        and view.source_frames[index] == frame
                    ):
                        y0, y1, x0, x1 = view.roi
                        pixels[y0:y1, x0:x1] = source[offset + index]
                yield pixels

    def publish_localisations(self, entries):
        """Stage all selected channel edits before swapping the live experiment."""
        staged = copy(self)
        staged.experiment = clone_experiment(self.experiment)
        staged._localisation_state = {
            key: values.copy()
            for key, values in self._localisation_state.items()
        }
        staged.images = _DatasetImages(staged)
        staged.localisations = LocalisationViews(staged)
        for name, channel, values in entries:
            staged.localisations[name][channel] = values
        self.experiment = staged.experiment
        self._localisation_state = staged._localisation_state
        self.revision = staged.revision

    def working_image_provenance(self, name, channel):
        dataset = self.dataset(name)
        state = dataset.processing.settings.get(
            "molseeq_working_images", {}
        ).get(channel, {})
        return {
            "history_namespace": "processing.settings.molseeq_working_images",
            "channel": channel,
            "history_id": state.get("active"),
            "acquisitions": [
                view.acquisition.id
                for view, _ in self.channel_parts(name, channel)
            ],
        }

    def replace_working_images(self, images, *, operation=None):
        operation = deepcopy(
            operation
            or {"method": "external_update", "parameters_available": False}
        )
        json.dumps(operation, allow_nan=False)
        replacements = {}
        for (name, channel), value in images.items():
            pixels = np.array(value, copy=True)
            if pixels.shape != self.channel(name, channel).shape:
                raise ValueError(
                    "Working images must preserve the channel frame mapping"
                )
            pixels.flags.writeable = False
            replacements[self.dataset(name).id, channel] = ArraySource(pixels)
        for dataset_id, channel in replacements:
            dataset = self.experiment.datasets[dataset_id]
            history = dataset.processing.settings.setdefault(
                "molseeq_working_images", {}
            ).setdefault(channel, {"active": None, "records": {}})
            identifier = str(uuid4())
            history["records"][identifier] = {
                "parent": history["active"],
                "operation": operation,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            history["active"] = identifier
        self._working.update(replacements)
        for key in replacements:
            self._display_pixels.pop(key, None)
        if replacements:
            self.revision += 1
            self.invalidate_traces(
                {identifier for identifier, _ in replacements}
            )

    def invalidate_traces(self, dataset_ids=None):
        if discard_trace_runs(self.experiment, dataset_ids):
            self.trace_revision += 1

    def rename(self, old: str, new: str) -> None:
        if not new.strip() or new in self._names:
            raise ValueError("Dataset names must be nonempty and unique")
        dataset = self.dataset(old)
        dataset.name = new
        self._names = {
            new if key == old else key: value
            for key, value in self._names.items()
        }
        self.revision += 1

    def delete(self, name: str) -> None:
        dataset = self.dataset(name)
        self.experiment.remove_dataset(dataset.id, cascade=True)
        del self._names[name]
        self._bindings = {
            key: binding
            for key, binding in self._bindings.items()
            if key[0] != dataset.id
        }
        for cache in (
            self._working,
            self._display_pixels,
            self._channel_state,
            self._localisation_state,
        ):
            for key in tuple(cache):
                if key[0] == dataset.id:
                    del cache[key]
        self.revision += 1

    def import_recordings(
        self,
        recordings: list[Recording],
        *,
        append_to: str | None = None,
        concatenate: bool = False,
    ) -> None:
        if not recordings:
            return
        experiment = clone_experiment(self.experiment)
        names = self._names.copy()
        bindings = self._bindings.copy()
        working = self._working.copy()
        if append_to is not None and append_to not in names:
            raise ValueError("Select an existing dataset to append to")
        groups = (
            [(append_to or recordings[0].name, recordings)]
            if concatenate
            else [
                (append_to or record.name, [record]) for record in recordings
            ]
        )
        for name, group in groups:
            if name not in names:
                dataset = experiment.add_dataset(Dataset(name=name))
                names[name] = dataset.id
            dataset = experiment.datasets[names[name]]
            incoming = [record.acquisition for record in group]
            if concatenate:
                channel_names = tuple(incoming[0].channels)
                previous = {
                    bindings.get((dataset.id, channel))
                    for channel in channel_names
                }
                if previous != {None}:
                    if len(previous) != 1 or None in previous:
                        raise ValueError(
                            "Concatenation requires identical channel bindings"
                        )
                    identifier, is_sequence = previous.pop()
                    current = (
                        dataset.sequences[identifier].acquisitions
                        if is_sequence
                        else [dataset.acquisitions[identifier]]
                    )
                else:
                    current = []
                # Validate before modifying any binding or registering the sequence.
                AcquisitionSequence([*current, *incoming])
                for acquisition in incoming:
                    dataset.add_acquisition(acquisition)
                sequence = dataset.add_sequence(
                    [acquisition.id for acquisition in [*current, *incoming]],
                    allow_cross_boundary_tracks=True,
                    reference=dataset.id,
                )
                for channel in channel_names:
                    key = dataset.id, channel
                    if key in working:
                        old = working[key].array
                        tail = sequence.channels[channel][len(old) :]
                        pixels = np.concatenate((old, tail))
                        pixels.flags.writeable = False
                        working[key] = ArraySource(pixels)
                    bindings[key] = (sequence.id, True)
                    history = dataset.processing.settings.get(
                        "molseeq_working_images", {}
                    ).get(channel)
                    if history is not None and key in working:
                        identifier = str(uuid4())
                        history["records"][identifier] = {
                            "parent": history["active"],
                            "operation": {
                                "method": "concatenate",
                                "acquisitions": [a.id for a in incoming],
                            },
                            "created_at": datetime.now(
                                timezone.utc
                            ).isoformat(),
                        }
                        history["active"] = identifier
            else:
                acquisition = dataset.add_acquisition(incoming[0])
                for channel in acquisition.channels:
                    key = dataset.id, channel
                    bindings[key] = (acquisition.id, False)
                    working.pop(key, None)
                    history = dataset.processing.settings.get(
                        "molseeq_working_images", {}
                    ).get(channel)
                    if history is not None:
                        history["active"] = None
        affected = {
            key[0]
            for key, binding in bindings.items()
            if self._bindings.get(key) != binding
        }
        if discard_trace_runs(experiment, affected):
            self.trace_revision += 1
        for cache in (self._display_pixels, self._channel_state):
            for key in tuple(cache):
                if bindings.get(key) != self._bindings.get(key):
                    del cache[key]
        if not concatenate:
            for key in tuple(self._localisation_state):
                if bindings.get(key) != self._bindings.get(key):
                    del self._localisation_state[key]
        sync_box_definitions(experiment, refresh=True)
        self.experiment, self._names, self._bindings = (
            experiment,
            names,
            bindings,
        )
        self._working = working
        self.revision += 1


class _DatasetImages(Mapping):
    """Temporary mapping for UI code that expects the legacy image keys."""

    def __init__(self, session):
        self.session = session

    def __getitem__(self, name):
        dataset_id = self.session.dataset(name).id
        return {
            channel: _ChannelImage(self.session, name, channel)
            for identifier, channel in self.session._bindings
            if identifier == dataset_id
        }

    def __iter__(self):
        return iter(self.session._names)

    def __len__(self):
        return len(self.session._names)


class _ChannelImage(MutableMapping):
    _labels = {"gap_label", "sequence_label"}

    def __init__(self, session, name, channel):
        self.session, self.name, self.channel = session, name, channel
        self.dataset = session.dataset(name)
        self.key = self.dataset.id, channel
        identifier, is_sequence = session._bindings[self.key]
        self.acquisition = (
            self.dataset.sequences[identifier].acquisitions[0]
            if is_sequence
            else self.dataset.acquisitions[identifier]
        )

    def _metadata(self):
        meta = self.acquisition.metadata
        excitation = (
            "d" if self.channel in {"donor", "acceptor", "dd", "da"} else "a"
        )
        emission = "d" if self.channel in {"donor", "dd", "ad"} else "a"
        return {
            "path": meta["source_path"],
            "channel_ref": excitation + emission,
            "excitation": excitation,
            "emission": emission,
            "channel_layout": meta["channel_layout"],
            "alex_first_frame": meta["alex_first_frame"],
            "FRET": self.acquisition.mode == "fret",
            "import_mode": meta["import_mode"],
            "pixel_size": self.acquisition.pixel_size,
            **{key: self.dataset.metadata.get(key) for key in self._labels},
            **self.session._channel_state.get(self.key, {}),
        }

    def __getitem__(self, key):
        if key != "data":
            return self._metadata()[key]
        cache = self.session._display_pixels
        if self.key not in cache:
            source = self.session.working_source(self.name, self.channel)
            pixels = (
                source.array if isinstance(source, ArraySource) else source[:]
            )
            pixels.flags.writeable = False
            cache[self.key] = pixels
        return cache[self.key]

    def __setitem__(self, key, value):
        if key == "data":
            self.session.replace_working_images(
                {(self.name, self.channel): value}
            )
        elif key in self._labels:
            self.dataset.update_metadata({key: value})
        elif key == "drift":
            self.session._channel_state.setdefault(self.key, {})[key] = value
        else:
            raise KeyError(key)

    def __delitem__(self, key):
        if key != "drift":
            raise KeyError(key)
        del self.session._channel_state[self.key][key]

    def __iter__(self):
        yield "data"
        yield from self._metadata()

    def __len__(self):
        return 1 + len(self._metadata())

    def copy(self):
        return dict(self)
