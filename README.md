# napari-molSEEQ

[![License MIT](https://img.shields.io/pypi/l/napari-GapSeq2.svg?color=green)](https://github.com/piedrro/napari-molseeq/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-GapSeq2.svg?color=green)](https://pypi.org/project/napari-molseeq/)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-GapSeq2.svg?color=green)](https://python.org)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-GapSeq2)](https://napari-hub.org/plugins/napari-molseeq)

A **Napari** plugin for extracting single molecule sequences from single/multi-channel SMLM microscopy data.

Compatible with both ALEX and FRET data. All functions are parallelised/GPU accelerated where possible to increase performance.
Multiple datasets can be loaded and processed in parallel.

napari-molseeq uses **MolCore** to manage recordings, localisations and photon sequences. It includes channel and dataset alignment, drift correction, **Picasso** localisation and trace extraction for **ALEX** and **FRET** data. Traces can be exported in different formats for downstream analysis.

Bounding boxes have persistent IDs and metadata in MolCore. Photon sequences retain the defining box revision, extraction settings, software versions and input-image processing history. Box edits invalidate dependent results while preserving their provenance.

napari-molseeq traces can be analysed with TraceAnalyser: https://github.com/piedrro/TraceAnalyser

This is still undergoing development, so some features may not work as expected.

This was built by Dr Piers Turner from the Kapanidis Lab, University of Oxford.

----------------------------------

## Installation

The Windows desktop installer opens napari with MolSeeq already docked. See
the [Windows build and installation instructions](docs/windows-installer.md).

This project uses Python 3.14, PyQt6 and [uv](https://docs.astral.sh/uv/getting-started/installation/) for development.

From a checkout of this repository, run:

```shell
uv sync --locked
uv run molseeq
```

uv downloads Python 3.14 when needed and installs the plugin in editable mode in `.venv`.
To use MolSeeq as a plugin, run `uv run napari` and open molSEEQ from the Plugins menu.

For a runtime environment without development tools, use `uv sync --locked --no-dev`.

## Contributing

```shell
uv sync --locked
uv run pytest
```

Add runtime dependencies with `uv add PACKAGE` and development tools with
`uv add --dev PACKAGE`. Commit both `pyproject.toml` and `uv.lock` when dependencies
change. Use `uv lock --upgrade` to update locked versions deliberately.

See [MAINTENANCE.md](MAINTENANCE.md) for interactive development, UI editing, and releases.
## License

Distributed under the terms of the [MIT] license,
"napari-molseeq" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/piedrro/napari-GapSeq2/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/


## MolCore compatibility

MolSeeq pins MolCore 0.2. Processing reads immutable dataset snapshots, uses
`PartitionSelection` or `SequenceSelection` and `context=ExecutionContext(...)`,
and returns detached results for explicit publication with `Dataset.apply(...)`.
Editable staging models use `molcore.staging.clone_dataset(...)`.

MolCore 0.2 reads and writes dataset and experiment **format 7 only**. Older saved
files are rejected. Preserve original files and use an environment with the older
MolCore release to export valuable scientific data before upgrading, then import
that data explicitly; there is no automatic old-format reader.
