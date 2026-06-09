#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/URA_challenge
mkdir -p third_party/llama.cpp logs/setup

LOG=logs/setup/llamacpp_setup.log
exec > >(tee "$LOG") 2>&1

echo "date=$(date -Iseconds)"
echo "workspace=$(pwd)"

if command -v llama-server >/dev/null 2>&1; then
  echo "llama_server=$(command -v llama-server)"
  llama-server --version || true
  exit 0
fi

release_json=third_party/llama.cpp/latest_release.json
curl -L --fail --silent --show-error https://api.github.com/repos/ggml-org/llama.cpp/releases/latest -o "$release_json"

python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("third_party/llama.cpp/latest_release.json").read_text())
assets = data.get("assets", [])
names = {a["name"]: a.get("browser_download_url") for a in assets}
for preferred in [
    "llama-b8958-bin-ubuntu-vulkan-x64.tar.gz",
    "llama-b8958-bin-ubuntu-x64.tar.gz",
]:
    if preferred in names:
        print(names[preferred])
        print(preferred)
        break
else:
    for asset in assets:
        name = asset.get("name", "")
        if "bin-ubuntu-vulkan-x64.tar.gz" in name or "bin-ubuntu-x64.tar.gz" in name:
            print(asset["browser_download_url"])
            print(name)
            break
    else:
        raise SystemExit("No suitable Linux llama.cpp release asset found")
PY

asset_url=$(python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("third_party/llama.cpp/latest_release.json").read_text())
assets = data.get("assets", [])
for suffix in ["bin-ubuntu-vulkan-x64.tar.gz", "bin-ubuntu-x64.tar.gz"]:
    matches = [a for a in assets if a.get("name", "").endswith(suffix)]
    if matches:
        print(matches[0]["browser_download_url"])
        break
else:
    raise SystemExit(1)
PY
)
asset_name=$(basename "$asset_url")
archive="third_party/llama.cpp/$asset_name"

if [ ! -f "$archive" ]; then
  echo "downloading=$asset_url"
  curl -L --fail --show-error "$asset_url" -o "$archive"
fi

rm -rf third_party/llama.cpp/bin
mkdir -p third_party/llama.cpp/bin
tar -xzf "$archive" -C third_party/llama.cpp/bin --strip-components=1

server_path=$(find third_party/llama.cpp/bin -type f -name llama-server | head -n 1)
if [ -z "$server_path" ]; then
  echo "error=llama-server not found in $asset_name"
  exit 1
fi

chmod +x "$server_path"
ln -sf "$(pwd)/$server_path" third_party/llama.cpp/llama-server

echo "asset=$asset_name"
echo "llama_server=$(pwd)/third_party/llama.cpp/llama-server"
"$(pwd)/third_party/llama.cpp/llama-server" --version
