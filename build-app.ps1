[CmdletBinding()]
param(
    [string]$IsccPath = '',
    [switch]$BundleOnly,
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
$project = Join-Path $PSScriptRoot 'packaging/windows'
$uv = (Get-Command uv -ErrorAction Stop).Source
& $uv lock --project $project --python 3.14
if ($LASTEXITCODE -ne 0) {
    throw "Windows build lock refresh failed with exit code $LASTEXITCODE."
}

& (Join-Path $project 'build.ps1') -IsccPath $IsccPath `
    -BundleOnly:$BundleOnly -SkipTests:$SkipTests
