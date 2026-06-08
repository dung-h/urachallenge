"""Install `llama-cpp-python` with CUDA enabled.

Historical utility: used when the project experimented with llama.cpp as a local backend.
If vLLM is the only backend used, this script can be treated as dev-only.

Notes:
- Designed for WSL/Linux paths.
- Forces reinstall and disables pip cache.
"""

import os
import subprocess
import sys

# Filter PATH to remove any /mnt/ paths
old_path = os.environ.get("PATH", "")
filtered_paths = [p for p in old_path.split(":") if "/mnt/" not in p and p.strip()]
# Add /usr/local/cuda/bin to path
filtered_paths.insert(0, "/usr/local/cuda/bin")
new_path = ":".join(filtered_paths)

# Setup environment variables for compilation
env = os.environ.copy()
env["PATH"] = new_path
env["CUDA_PATH"] = "/usr/local/cuda"
env["CUDACXX"] = "/usr/local/cuda/bin/nvcc"
env["CMAKE_ARGS"] = "-DGGML_CUDA=on"

print(f"Old PATH length: {len(old_path)}")
print(f"New PATH: {new_path}")
print(f"CMAKE_ARGS: {env.get('CMAKE_ARGS')}")

# Run pip install
cmd = [sys.executable, "-m", "pip", "install", "llama-cpp-python", "--verbose", "--force-reinstall", "--no-cache-dir"]
print(f"Running command: {' '.join(cmd)}")
subprocess.run(cmd, env=env, check=True)
