# WSL Networking & LLM Server / Inference IP Alignment

How to keep the LLM server (port 8001) and the API/inference client
(`URA_LLM_BASE_URL`) pointing at the same address. Misalignment here caused real
outages in earlier sessions (the API 500'd or fell back to a mock because `.env`
pointed at a stale IP).

## 0. THE ONE FACT THAT MATTERS (empirically verified on this machine)

A server bound inside WSL on `0.0.0.0:8001` IS reachable at the **WSL LAN IP**
(`192.168.1.5:8001`) but is **NOT reachable at `127.0.0.1:8001` or
`localhost:8001`**. Verified directly:

```
curl http://127.0.0.1:8001/v1/models   -> (empty / connection fails)
curl http://localhost:8001/v1/models    -> (empty / connection fails)
curl http://192.168.1.5:8001/v1/models  -> {"object":"list","data":[...]}   OK
```

This is a WSL2 **mirrored-mode** loopback quirk: even though `ss` shows the
socket on `0.0.0.0:8001`, loopback (`127.0.0.1`) does not route to it the way a
native Linux host would. **So `URA_LLM_BASE_URL` MUST use the WSL LAN IP, not
`127.0.0.1`.** (The repo `README.md` and `start_vllm_server.sh` default of
`127.0.0.1` is WRONG for this machine — do not trust it.)

## 1. This machine's networking facts (verified)

- WSL is in **mirrored networking mode** (`%USERPROFILE%\.wslconfig` →
  `[wsl2] networkingMode=mirrored`, `localhostForwarding=true`). WSL version 2.7.3.
- WSL shares the Windows host LAN IP. Right now that is `192.168.1.5` (default
  route via `192.168.1.1`, interface `eth9`); the same IP appears in both
  `wsl hostname -I` and Windows `ipconfig`.
- The LAN IP is DHCP-assigned and **can change on reboot** — so do NOT hardcode a
  literal IP and forget it. Derive it dynamically (see §3).

## 2. The golden rule

- **Server bind**: always `--host 0.0.0.0` (LLM server AND API server).
- **Client connect**: use the **WSL LAN IP**, i.e. `http://<wsl-ip>:8001/v1`.
  NOT `127.0.0.1`, NOT `localhost`.
- Re-derive `<wsl-ip>` after every reboot before starting the API server.

## 3. Get the current WSL LAN IP (use this, don't hardcode blindly)

```bash
# first IPv4 address WSL holds == the LAN IP that works for 8001
WSL_IP=$(hostname -I | awk '{print $1}')
echo "$WSL_IP"            # e.g. 192.168.1.5
```

Start the API server using that IP:

```bash
WSL_IP=$(hostname -I | awk '{print $1}')
source .venv/bin/activate && \
  URA_LLM_MODEL='<model-id-served-on-8001>' \
  URA_LLM_BASE_URL="http://$WSL_IP:8001/v1" \
  URA_ENABLE_QUALITATIVE_PARSER=1 \
  python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Keep `.env` in sync when the IP changes:

```bash
WSL_IP=$(hostname -I | awk '{print $1}')
sed -i "s|^URA_LLM_BASE_URL=.*|URA_LLM_BASE_URL=http://$WSL_IP:8001/v1|" .env
grep URA_LLM_BASE_URL .env
```

> The API server (port 8000) is reachable from the Windows browser/curl at
> `http://localhost:8000` (Windows→WSL loopback works via `localhostForwarding`).
> It is the WSL-internal client→8001 hop that needs the LAN IP, not `127.0.0.1`.

## 4. Verify alignment in 4 commands (run inside WSL)

```bash
hostname -I                                   # (a) current WSL IP(s)
ss -tlnp | grep -E ':800[01]'                 # (b) servers listening?
curl -s -m 5 http://$(hostname -I|awk '{print $1}'):8001/v1/models | head -c 200  # (c) LLM answers on LAN IP?
grep URA_LLM_BASE_URL .env                     # (d) does .env match (a)?
```

Alignment is correct when (c) returns a JSON model list AND the host in
`URA_LLM_BASE_URL` equals the first IP from (a).

## 5. Common failure modes and fixes

| Symptom | Cause | Fix |
|---|---|---|
| `/predict` 500 / `ConnectionError [Errno 111]` | LLM server down, OR `URA_LLM_BASE_URL` uses `127.0.0.1` (which does not reach the server here), OR a stale LAN IP | Start LLM server `--host 0.0.0.0 --port 8001`; set `URA_LLM_BASE_URL` to `http://<hostname -I first IP>:8001/v1`; restart API server |
| `curl 127.0.0.1:8001` empty but server is up | mirrored-mode loopback quirk (see §0) | Use the WSL LAN IP, never `127.0.0.1`/`localhost` |
| `/health` shows `provider: mock`, `/predict` 404 | a stale non-project server squats on :8000 | `ss -tlnp \| grep :8000`, `kill -9 <pid>`, restart `uvicorn app.main:app` |
| All answers "unknown" after reboot | both servers not restarted (no auto-start) | start LLM (8001) then API (8000) |
| Was working, broke after reboot | DHCP changed the LAN IP; `.env` now stale | re-derive `WSL_IP` and update `.env` (§3) |
| Port already in use on launch | old server process alive | `pkill -9 -f vllm.entrypoints` / `pkill -9 -f llama_cpp.server`; confirm with `ss -tlnp` + `nvidia-smi` |

## 6. One LLM at a time (8GB GPU)

RTX 4060 = 8GB VRAM; only ONE LLM server may hold the GPU. Before swapping
backends (Qwen vLLM ↔ Gemma llama-cpp):

```bash
pkill -9 -f vllm.entrypoints      # or: pkill -9 -f llama_cpp.server
sleep 3
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader   # confirm freed
```

Then start the new server on `0.0.0.0:8001`. `URA_LLM_BASE_URL` stays the same
(LAN IP:8001); only `URA_LLM_MODEL` changes to match the served model id.

## 7. Notes

- vLLM also needs `VLLM_USE_V1=0` on this box (flashinfer ABI bug crashes the V1
  engine). See CLAUDE.md / AGENTS.md.
- If WSL is ever switched out of mirrored mode (NAT default), the WSL IP becomes a
  private `172.x`/`10.x` address; `hostname -I | awk '{print $1}'` still yields the
  correct client target, so the §3 dynamic-IP approach keeps working.
