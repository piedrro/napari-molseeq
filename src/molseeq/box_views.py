"""Project MolCore bounding boxes into the existing MolSeeq editing controls."""

from collections import defaultdict, deque
from collections.abc import MutableMapping
from copy import deepcopy
from dataclasses import asdict
from uuid import uuid4
from datetime import datetime, timezone

import numpy as np
from molcore import BoundingBox
from molcore.staging import clone_experiment, experiment_state_token
from molcore import StaleResultError

_KEY = "molseeq_bounding_boxes"


def _plain(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    return value


def box_state(session):
    state = deepcopy(session.experiment.metadata.get(_KEY, {}))
    if session.experiment.datasets:
        reference = next(iter(session.experiment.datasets.values()))
        active = set(reference.bounding_boxes)
        state["definitions"] = [
            asdict(reference.bounding_boxes[value["id"]])
            for value in state.get("definitions", [])
            if value["id"] in active
        ]
    return state


def sync_box_definitions(experiment, *, refresh=False):
    state = experiment.metadata.get(_KEY, {})
    if refresh and experiment.datasets and state:
        # Existing native definitions are authoritative when another round is added.
        reference = next(iter(experiment.datasets.values()))
        active = set(reference.bounding_boxes)
        existing = [value["id"] for value in state.get("definitions", [])]
        stored = {
            box.id
            for box in reference.bounding_boxes.query(include_deleted=True)
        }
        if any(identifier in stored for identifier in existing):
            state["definitions"] = [
                asdict(reference.bounding_boxes[identifier])
                for identifier in existing
                if identifier in active
            ]
    definitions = [
        BoundingBox(**value) for value in state.get("definitions", [])
    ]
    active = {box.id for box in definitions}
    for dataset in experiment.datasets.values():
        for identifier in tuple(dataset.bounding_boxes):
            box = dataset.bounding_boxes[identifier]
            if (
                box.metadata.get("application") == "molseeq"
                and identifier not in active
            ):
                dataset.bounding_boxes.remove(identifier)
        for box in definitions:
            dataset.bounding_boxes.set(box)


def replace_box_state(session, entry):
    previous = box_state(session)
    options = {
        key: _plain(value)
        for key, value in entry.items()
        if key not in ("localisations", "_box_ids")
    }
    rows = entry.get("localisations")
    definitions = []
    dtype = None
    if rows is not None:
        rows = np.asarray(rows)
        if not rows.size and rows.dtype.names is None:
            rows = np.empty(
                0, dtype=[("frame", "u4"), ("x", "f4"), ("y", "f4")]
            )
        if rows.ndim != 1 or not {"x", "y"} <= set(rows.dtype.names or ()):
            raise ValueError("Bounding boxes require x and y columns")
        old = [
            BoundingBox(**value) for value in previous.get("definitions", [])
        ]
        old_by_id = {box.id: box for box in old}
        matches = defaultdict(deque)
        for box in old:
            matches[
                box.x, box.y, box.metadata["legacy_row"].get("frame", 0)
            ].append(box.id)
        identifiers = [None] * len(rows)
        explicit_ids = entry.get("_box_ids")
        if explicit_ids is not None:
            if len(explicit_ids) != len(rows):
                raise ValueError(
                    "Bounding box IDs must match the edited positions"
                )
            known = [
                identifier
                for identifier in explicit_ids
                if identifier is not None
            ]
            if (
                len(known) != len(set(known))
                or not set(known) <= old_by_id.keys()
            ):
                raise ValueError(
                    "Edited bounding box IDs must be distinct existing IDs"
                )
            identifiers = list(explicit_ids)
        else:
            for index, row in enumerate(rows):
                frame = row["frame"] if "frame" in rows.dtype.names else 0
                exact = matches.get(
                    (float(row["x"]), float(row["y"]), int(frame))
                )
                if exact:
                    identifiers[index] = exact.popleft()
        unused = [box.id for box in old if box.id not in identifiers]
        for index, identifier in enumerate(identifiers):
            if identifier is None:
                identifiers[index] = (
                    unused.pop(0)
                    if explicit_ids is None
                    and len(rows) == len(old)
                    and unused
                    else str(uuid4())
                )
        width = float(options.get("box_size", 3))
        origin = options.get("origin", "molseeq")
        for identifier, row in zip(identifiers, rows, strict=True):
            fields = {
                key: _plain(row[key])
                for key in rows.dtype.names
                if key not in ("x", "y")
            }
            previous_box = old_by_id.get(identifier)
            original_metadata = (
                previous_box.metadata if previous_box is not None else {}
            )
            box = BoundingBox(
                id=identifier,
                x=row["x"],
                y=row["y"],
                width=width,
                height=width,
                space="working",
                reference="molseeq:working",
                revision=int(previous.get("revision", 0)) + 1,
                created_at=previous_box.created_at
                if previous_box is not None
                else None,
                updated_at=previous_box.updated_at
                if previous_box is not None
                else None,
                metadata={
                    "application": "molseeq",
                    "origin": original_metadata.get("origin", origin),
                    "created_at": original_metadata.get(
                        "created_at", datetime.now(timezone.utc).isoformat()
                    ),
                    "source": original_metadata.get(
                        "source", options.get("source", {})
                    ),
                    "fitted": bool(options.get("fitted", False)),
                    "legacy_row": fields,
                    "last_edit": {
                        "origin": origin,
                        "source": options.get("source", {}),
                    },
                },
            )
            definitions.append(asdict(box))
        dtype = _plain(rows.dtype.descr)
    state = {"options": options, "definitions": definitions, "dtype": dtype}
    comparable = {
        key: value for key, value in previous.items() if key != "revision"
    }
    # Geometry equality ignores the selection's monotonically increasing revision.
    for definition in state["definitions"]:
        definition["revision"] = 0
    normalized_previous = deepcopy(comparable)
    for definition in normalized_previous.get("definitions", []):
        definition["revision"] = 0
    if normalized_previous == state:
        return
    state["revision"] = int(previous.get("revision", 0)) + 1
    for definition in state["definitions"]:
        definition["revision"] = state["revision"]
    expected = experiment_state_token(session.experiment)
    staged = clone_experiment(session.experiment)
    staged.metadata[_KEY] = state
    sync_box_definitions(staged)
    if experiment_state_token(session.experiment) != expected:
        raise StaleResultError("Bounding box inputs changed during editing")
    session.experiment = staged
    session.revision += 1
    session.trace_revision += 1


class BoundingBoxView(MutableMapping):
    def __init__(self, session):
        self.session = session

    def __getitem__(self, key):
        state = box_state(self.session)
        if key != "localisations":
            return deepcopy(state.get("options", {})[key])
        if state.get("dtype") is None:
            raise KeyError(key)
        descriptors = [
            tuple(field[:2])
            if len(field) == 2
            else (*field[:2], tuple(field[2]))
            for field in state["dtype"]
        ]
        boxes = state["definitions"]
        rows = np.empty(len(boxes), dtype=np.dtype(descriptors)).view(
            np.recarray
        )
        for index, box in enumerate(boxes):
            for field in rows.dtype.names:
                value = (
                    box[field]
                    if field in ("x", "y")
                    else box["metadata"]["legacy_row"].get(field)
                )
                rows[field][index] = (
                    np.nan
                    if value is None and rows.dtype[field].kind == "f"
                    else value
                )
        return rows

    def __setitem__(self, key, value):
        entry = self.copy()
        entry[key] = value
        replace_box_state(self.session, entry)

    def __delitem__(self, key):
        entry = self.copy()
        del entry[key]
        replace_box_state(self.session, entry)

    def __iter__(self):
        state = box_state(self.session)
        yield from state.get("options", {})
        if state.get("dtype") is not None:
            yield "localisations"

    def __len__(self):
        return sum(1 for _ in self)

    def copy(self):
        return dict(self)

    def __deepcopy__(self, memo):
        return deepcopy(dict(self), memo)


class LocalisationState(MutableMapping):
    def __init__(self, session):
        self.session = session

    def __getitem__(self, key):
        if key == "bounding_boxes":
            return BoundingBoxView(self.session)
        if key == "localisations":
            return self.session.localisations
        raise KeyError(key)

    def __setitem__(self, key, value):
        if key == "bounding_boxes":
            replace_box_state(self.session, value)
        elif key == "localisations":
            self.session.localisations.update(value)
        else:
            raise KeyError(key)

    def __delitem__(self, key):
        if key == "bounding_boxes":
            replace_box_state(self.session, {})
        else:
            raise KeyError(key)

    def __iter__(self):
        return iter(("bounding_boxes", "localisations"))

    def __len__(self):
        return 2
