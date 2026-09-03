#!/usr/bin/env bash
# Download a NICE-SLAM processed Replica zip and extract it (optionally just one scene).
# Demo.zip (~157MB, single scene) is the fastest dev data.
#   scripts/prepare_replica.sh --url https://cvg-data.inf.ethz.ch/nice-slam/data/Demo.zip --out data/replica_demo
set -euo pipefail
export https_proxy="${PROXY:-http://127.0.0.1:7897}"
export http_proxy="${https_proxy}"

URL="https://cvg-data.inf.ethz.ch/nice-slam/data/Demo.zip"
OUT="data/replica_demo"
SCENE=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --url) URL="$2"; shift 2;;
        --out) OUT="$2"; shift 2;;
        --scene) SCENE="$2"; shift 2;;
        *) echo "unknown arg $1"; exit 1;;
    esac
done

mkdir -p "$OUT"
ZIP="/tmp/$(basename "$URL")"
echo ">> downloading $URL"
curl -fL --retry 5 --retry-all-errors -C - -x "$https_proxy" -o "$ZIP" "$URL"

echo ">> unzipping into $OUT"
SCENE_ARG="$SCENE" ZIP="$ZIP" OUT="$OUT" python3 - <<'PY'
import os, zipfile, pathlib
z, out, scene = os.environ["ZIP"], pathlib.Path(os.environ["OUT"]), os.environ["SCENE"]
with zipfile.ZipFile(z) as zf:
    names = [m for m in zf.namelist() if (scene in m) if scene] if scene else zf.namelist()
    if scene:
        names = [m for m in zf.namelist() if scene in m]
    else:
        names = zf.namelist()
    for m in names:
        zf.extract(m, out)
    print(f"extracted {len(names)} files")
PY
rm -f "$ZIP"
echo ">> done. Contents (top levels):"
find "$OUT" -maxdepth 3 -type d | head -20
