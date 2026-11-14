[CmdletBinding()]
param(
    [string]$IsccPath = '',
    [switch]$BundleOnly,
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Program failed with exit code $LASTEXITCODE."
    }
}

if ([Environment]::OSVersion.Platform -ne 'Win32NT' -or
    $env:PROCESSOR_ARCHITECTURE -ne 'AMD64') {
    throw 'Build on Windows x64 using 64-bit PowerShell.'
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
if (-not $BundleOnly) {
    if (-not $IsccPath) {
        $compiler = Get-Command ISCC.exe -ErrorAction SilentlyContinue
        if ($compiler) {
            $IsccPath = $compiler.Source
        } else {
            $candidates = @(
                "${env:ProgramFiles(x86)}/Inno Setup 6/ISCC.exe",
                "$env:ProgramFiles/Inno Setup 6/ISCC.exe"
            )
            $IsccPath = $candidates | Where-Object { Test-Path -LiteralPath $_ } |
                Select-Object -First 1
        }
    }
    if (-not $IsccPath -or -not (Test-Path -LiteralPath $IsccPath -PathType Leaf)) {
        throw 'Install Inno Setup 6 or pass -IsccPath with the path to ISCC.exe.'
    }
    $IsccPath = (Resolve-Path -LiteralPath $IsccPath).Path
}

$uv = (Get-Command uv -ErrorAction Stop).Source
$environment = Join-Path $PSScriptRoot '.venv'
$python = Join-Path $environment 'Scripts/python.exe'
$output = Join-Path $repoRoot 'dist/windows'
$bundle = Join-Path $output 'napari-molseeq'
$work = Join-Path $repoRoot 'build/windows'
$environmentOverrides = @{
    QT_API = 'pyqt6'
    NUMBA_DISABLE_CUDA = '1'
    UV_PROJECT_ENVIRONMENT = $environment
    PYINSTALLER_CONFIG_DIR = (Join-Path $work 'pyinstaller-cache')
}
$previousEnvironment = @{}
$previousPath = $env:PATH

Push-Location $repoRoot
try {
    foreach ($key in $environmentOverrides.Keys) {
        $previousEnvironment[$key] = [Environment]::GetEnvironmentVariable($key)
        [Environment]::SetEnvironmentVariable($key, $environmentOverrides[$key])
    }
    New-Item -ItemType Directory -Force -Path $output, $work | Out-Null
    Invoke-Checked $uv @('sync', '--project', $PSScriptRoot, '--locked', '--no-editable',
        '--reinstall-package', 'napari-molseeq', '--python', '3.14')

    $verify = @"
import platform, sys
from importlib.metadata import version
from molseeq import __version__
from qtpy import API_NAME
assert sys.version_info[:2] == (3, 14)
assert platform.machine() == 'AMD64'
assert API_NAME == 'PyQt6'
assert version('napari-molseeq') == __version__
print(__version__)
"@
    $version = & $python -c $verify
    if ($LASTEXITCODE -ne 0 -or $version -notmatch '^\d+\.\d+\.\d+$') {
        throw "Failed to verify the Windows runtime and package version. Got: $version"
    }
    if (-not $SkipTests) {
        Invoke-Checked $python @('-m', 'pytest', 'tests', '-q', '--tb=short')
    }

    # Prevent unrelated tools on PATH from providing incompatible native DLLs.
    $basePython = & $python -c 'import sys; print(sys.base_prefix)'
    if ($LASTEXITCODE -ne 0) { throw 'Could not locate the Python runtime.' }
    $systemPath = @("$env:SystemRoot/System32", "$env:SystemRoot",
        "$env:SystemRoot/System32/Wbem")
    $env:PATH = (@((Split-Path $python), $basePython) + $systemPath) -join [IO.Path]::PathSeparator
    Invoke-Checked $python @('-m', 'PyInstaller', '--noconfirm', '--clean',
        '--distpath', $output, '--workpath', $work, (Join-Path $PSScriptRoot 'molseeq.spec'))

    # Check that the frozen app also runs without either build Python on PATH.
    $env:PATH = $systemPath -join [IO.Path]::PathSeparator
    $report = Join-Path $output 'smoke-test.json'
    if (Test-Path -LiteralPath $report) { Remove-Item -LiteralPath $report }
    $process = Start-Process -FilePath (Join-Path $bundle 'napari-molseeq.exe') `
        -ArgumentList @('--smoke-test', ('"' + $report + '"')) -WorkingDirectory $work `
        -PassThru -WindowStyle Hidden
    if (-not $process.WaitForExit(300000)) {
        Stop-Process -Id $process.Id -Force
        throw 'Packaged smoke test timed out. Inspect %LOCALAPPDATA%/MolSeeq/logs.'
    }
    if ($process.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $report)) {
        throw "Packaged smoke test failed with exit code $($process.ExitCode). Inspect %LOCALAPPDATA%/MolSeeq/logs."
    }
    $result = Get-Content -LiteralPath $report -Raw | ConvertFrom-Json
    if ($result.ok -ne $true -or $result.version -ne $version -or
        $result.dock -ne "MolSeeq v$version" -or $result.worker -le 0) {
        throw 'Packaged smoke report did not verify the application, versioned dock and worker.'
    }

    $manifest = & $uv pip freeze --python $python
    if ($LASTEXITCODE -ne 0) { throw 'Failed to export the dependency manifest.' }
    $manifest | Set-Content -LiteralPath (Join-Path $output 'dependencies.txt') -Encoding UTF8
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'uv.lock') -Destination $output
    if ($BundleOnly) {
        Write-Host "Verified application bundle: $bundle"
        return
    }

    Invoke-Checked $IsccPath @("/DAppVersion=$version", "/DBundleDir=$bundle",
        "/DOutputDir=$output", (Join-Path $PSScriptRoot 'installer.iss'))
    $installer = Join-Path $output "napari-molseeq-Setup-$version-x64.exe"
    $hash = (Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  $([IO.Path]::GetFileName($installer))" |
        Set-Content -LiteralPath "$installer.sha256" -Encoding ascii
    Write-Host "Verified application bundle: $bundle"
    Write-Host "Installer: $installer"
}
finally {
    foreach ($key in $previousEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($key, $previousEnvironment[$key])
    }
    $env:PATH = $previousPath
    Pop-Location
}
