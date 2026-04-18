#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Felipe Reyes
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Downloads the fast-alpr YOLOv9-t 384 license-plate-detection ONNX model
# into data/models so the plugin can find it at load time.
#
# Usage:   scripts/fetch_model.sh
#
# If fast-alpr relocates the release asset in the future, update MODEL_URL
# below (and the SHA256 if you're being careful).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# The pretrained YOLOv9-t plate detector is published by ankandrew's
# open-image-models project (which fast-alpr depends on for detection). The
# release tag is literally "assets" and the asset name uses the plural
# "license-plates" — we rename to "license-plate" locally to match the path
# baked into src/plate-blur-consts.h.
MODEL_URL="${MODEL_URL:-https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-t-384-license-plates-end2end.onnx}"
TARGET_DIR="$REPO_ROOT/data/models"
TARGET_FILE="$TARGET_DIR/yolo-v9-t-384-license-plate-end2end.onnx"

mkdir -p "$TARGET_DIR"

if [[ -f "$TARGET_FILE" ]]; then
  echo "Model already present at: $TARGET_FILE"
  echo "(delete it if you want to re-download)"
  exit 0
fi

echo "Fetching plate detection model from:"
echo "  $MODEL_URL"
echo "Saving to: $TARGET_FILE"

if command -v curl >/dev/null 2>&1; then
  curl -L --fail --progress-bar -o "$TARGET_FILE" "$MODEL_URL"
elif command -v wget >/dev/null 2>&1; then
  wget --show-progress -O "$TARGET_FILE" "$MODEL_URL"
else
  echo "ERROR: neither curl nor wget is available on PATH" >&2
  exit 1
fi

echo "Downloaded $(du -h "$TARGET_FILE" | cut -f1) to $TARGET_FILE"
