# SPDX-FileCopyrightText: 2026 Felipe Reyes
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Downloads the fast-alpr YOLOv9-t 384 license-plate-detection ONNX model
# into data/models so the plugin can find it at load time.
#
# Usage:   powershell -ExecutionPolicy Bypass -File scripts/fetch_model.ps1

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot = Split-Path -Parent $ScriptDir

$ModelUrl = if ($env:MODEL_URL) { $env:MODEL_URL } else { "https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-t-384-license-plates-end2end.onnx" }
$TargetDir = Join-Path $RepoRoot "data/models"
$TargetFile = Join-Path $TargetDir "yolo-v9-t-384-license-plate-end2end.onnx"

New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null

if (Test-Path $TargetFile) {
    Write-Host "Model already present at: $TargetFile"
    Write-Host "(delete it if you want to re-download)"
    exit 0
}

Write-Host "Fetching plate detection model from:"
Write-Host "  $ModelUrl"
Write-Host "Saving to: $TargetFile"

Invoke-WebRequest -Uri $ModelUrl -OutFile $TargetFile

$size = (Get-Item $TargetFile).Length
Write-Host ("Downloaded {0:N1} MB to {1}" -f ($size / 1MB), $TargetFile)
