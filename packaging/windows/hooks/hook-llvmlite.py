"""Collect llvmlite's own DLLs and the private runtime beside its package."""

from PyInstaller.utils.hooks import (
    collect_delvewheel_libs_directory,
    collect_dynamic_libs,
)


datas, binaries = collect_delvewheel_libs_directory(
    "llvmlite", datas=[], binaries=collect_dynamic_libs("llvmlite")
)
