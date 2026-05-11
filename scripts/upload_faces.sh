#!/usr/bin/env bash
# Upload training photos for a family member to home-watcher.
#
# Usage:
#   ./scripts/upload_faces.sh <subject_name> <photo_dir>
#
# Example:
#   ./scripts/upload_faces.sh Malin ~/photos/malin/
#
# Env vars:
#   HOME_WATCHER_URL   default: http://192.168.1.250:30040

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <subject_name> <photo_dir>" >&2
    exit 1
fi

subject="$1"
photo_dir="$2"
url="${HOME_WATCHER_URL:-http://192.168.1.250:30040}"

if [ ! -d "$photo_dir" ]; then
    echo "Photo dir not found: $photo_dir" >&2
    exit 1
fi

shopt -s nullglob nocaseglob
photos=("$photo_dir"/*.jpg "$photo_dir"/*.jpeg "$photo_dir"/*.png)
shopt -u nullglob nocaseglob

if [ ${#photos[@]} -eq 0 ]; then
    echo "No photos found in $photo_dir (expected .jpg/.jpeg/.png)" >&2
    exit 1
fi

echo "Uploading ${#photos[@]} photo(s) for $subject to $url"

ok=0
fail=0
for photo in "${photos[@]}"; do
    filename=$(basename "$photo")
    if curl -fsS -F "photo=@$photo" "$url/api/subjects/$subject/photos" >/dev/null 2>&1; then
        echo "  ok    $filename"
        ok=$((ok+1))
    else
        echo "  FAIL  $filename (no face detected or upload error)"
        fail=$((fail+1))
    fi
done

echo
echo "Reloading face DB cache..."
curl -fsS -X POST "$url/api/reload" >/dev/null
echo "Subjects in DB:"
curl -fsS "$url/api/subjects"
echo

echo "Done: $ok uploaded, $fail failed"
