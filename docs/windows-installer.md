# MolSeeq Windows desktop app

The Windows x64 installer includes Python and napari and opens a viewer with
the versioned MolSeeq dock already visible. The desktop build runs scientific
processing on the CPU. It requires Windows 10 (1809 or later) or Windows 11
and a compatible OpenGL driver. [Qt 6.12 Windows support](https://doc.qt.io/qt-6.12/windows.html)

## Build

Install uv and Inno Setup 6, then run from the repository root in 64-bit
PowerShell.

```powershell
.\build-app.ps1
```

Pass `-IsccPath 'C:\path\to\ISCC.exe'` when Inno Setup is installed in a
custom location. Use `-BundleOnly` to build just the runnable folder without
Inno Setup. `-SkipTests` skips the source test suite and still runs the frozen
application smoke test.

The wrapper refreshes the dedicated Windows lockfile and builds in
`packaging/windows/.venv` with Python 3.14. For a build using the committed
lockfile, run `.\packaging\windows\build.ps1`. The manually triggered
**Windows installer** workflow also uses the committed lockfile.

The executable lives at
`dist\windows\napari-molseeq\napari-molseeq.exe`. Keep its folder together
when copying it to another machine. The installer, SHA-256 checksum and
`smoke-test.json` are in `dist\windows`. The executable and installer use the
napari application icon included in the installed napari package.

## Install and verify

Run `napari-molseeq-Setup-<version>-x64.exe` as a standard user. The installer
adds a Start menu shortcut and offers a desktop shortcut. The default install
location is `%LOCALAPPDATA%\Programs\napari-molseeq`. The application keeps
its logs and caches under `%LOCALAPPDATA%\MolSeeq`.

On a clean Windows x64 system without Python, launch MolSeeq from Start and
check the versioned dock. Import a small ALEX TIFF, run localisation and
trace extraction, then export a TIFF and trace data. Run the installer again
to exercise the upgrade path, then uninstall and check that exported files
and logs remain available.

To run the same smoke test that gates installer compilation, use the installed
app in PowerShell.

```powershell
$app = Join-Path $env:LOCALAPPDATA 'Programs\napari-molseeq\napari-molseeq.exe'
$report = Join-Path $env:TEMP 'molseeq-smoke.json'
$process = Start-Process $app -ArgumentList @('--smoke-test', ('"' + $report + '"')) -Wait -PassThru -WindowStyle Hidden
$process.ExitCode
Get-Content -LiteralPath $report
```
