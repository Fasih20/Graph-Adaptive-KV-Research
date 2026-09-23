#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 OUTPUT_DIRECTORY" >&2
  exit 2
fi
output_dir="${1%/}"
if [[ ! -d "$output_dir" ]]; then
  echo "Not a directory: $output_dir" >&2
  exit 2
fi
archive="${output_dir}_bundle.zip"
python validate_repaired.py "$output_dir"
python analyze_repaired.py "$output_dir"
python -m zipfile -c "$archive" "$output_dir"
sha256sum "$archive" > "${archive}.sha256"
echo "$archive"

