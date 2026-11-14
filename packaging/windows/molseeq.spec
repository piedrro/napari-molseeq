"""Freeze the Windows desktop application using its installed wheel."""

from importlib.util import find_spec
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules, copy_metadata
from PyInstaller.utils.win32.versioninfo import (
    FixedFileInfo, StringFileInfo, StringStruct, StringTable, VarFileInfo,
    VarStruct, VSVersionInfo,
)

from molseeq import __version__


packaging = Path(SPECPATH).resolve()
root = packaging.parents[1]
datas = [(str(root / "LICENSE"), ".")]
binaries = []
hiddenimports = []
excluded_modules = (
    "picasso.gui", "picasso.server", "picasso.__main__",
    "scipy._external.array_api_compat.torch",
    "scipy._external.array_api_compat.cupy",
)


def runtime_module(name):
    return not (
        set(name.split(".")) & {"tests", "_tests", "dev", "scripts", "examples", "benchmarks"}
        or name.startswith(excluded_modules)
    )


# napari and MolCore discover plugins and numerical backends at runtime.
for package in ("molseeq", "molcore", "napari", "napari_resources", "napari_builtins",
                "napari_console", "napari_svg", "npe2", "vispy", "ome_types", "picasso"):
    if find_spec(package) is None:
        continue
    data, binary, imports = collect_all(
        package, filter_submodules=runtime_module,
        exclude_datas=["**/tests/**", "**/_tests/**", "**/dev/**", "**/scripts/**"],
    )
    datas += data
    binaries += binary
    hiddenimports += imports

datas += copy_metadata("napari-molseeq", recursive=True)
datas += collect_data_files("matplotlib")
hiddenimports += collect_submodules(
    "scipy._external.array_api_compat", filter=runtime_module,
)
datas += collect_data_files("scipy._external.array_api_compat", include_py_files=True)

version = tuple(map(int, __version__.split("."))) + (0,)
version_info = VSVersionInfo(
    ffi=FixedFileInfo(filevers=version, prodvers=version, mask=0x3f,
                      flags=0, OS=0x40004, fileType=1, subtype=0, date=(0, 0)),
    kids=[StringFileInfo([StringTable("040904B0", [
        StringStruct("FileDescription", "MolSeeq single-molecule sequencing"),
        StringStruct("FileVersion", __version__),
        StringStruct("ProductName", "MolSeeq"),
        StringStruct("ProductVersion", __version__),
        StringStruct("OriginalFilename", "napari-molseeq.exe"),
    ])]), VarFileInfo([VarStruct("Translation", [1033, 1200])])],
)

a = Analysis(
    [str(packaging / "entry.py")],
    binaries=binaries, datas=datas, hiddenimports=hiddenimports,
    hookspath=[str(packaging / "hooks")],
    runtime_hooks=[str(packaging / "runtime_hook.py")],
    excludes=["PyQt5", "PySide2", "PySide6", "tkinter", "_tkinter", "pytest", "pygpufit",
              "torch", "torchvision", "cupy", *excluded_modules],
    hooksconfig={"matplotlib": {"backends": ["QtAgg"]}},
    module_collection_mode={
        package: "pyz+py" for package in ("napari", "molcore", "picasso", "trackpy")
    },
)
pyz = PYZ(a.pure)
napari_root = Path(find_spec("napari").origin).parent
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="napari-molseeq",
          icon=str(napari_root / "resources" / "icon.ico"), upx=False,
          console=False, version=version_info)
coll = COLLECT(exe, a.binaries, a.datas, upx=False, name="napari-molseeq")
