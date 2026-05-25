URA EXACT Challenge — Quickstart

LLM Backends (vLLM / OpenAI-compatible first)
- Mặc định runtime đang nhắm tới một OpenAI-compatible local server, thường là vLLM:

```powershell
$env:URA_LLM_BACKEND='openai-compatible'
$env:URA_LLM_BASE_URL='http://127.0.0.1:8001/v1'
$env:URA_LLM_MODEL='Qwen/Qwen2.5-7B-Instruct'
```

- Nếu bạn muốn smoke test với Ollama local, vẫn có thể override:

```powershell
$env:URA_LLM_BACKEND='ollama'
$env:URA_LLM_BASE_URL='http://localhost:11434'
$env:URA_LLM_MODEL='qwen2.5:7b'
```

Mục tiêu
- Ứng dụng FastAPI để trả lời câu hỏi giáo dục (physics, logic).
- Python-based deterministic solvers là authority chính; LLM chỉ là orchestrator / worker / explanation writer khi được bật qua backend cục bộ.
- Final JSON luôn do backend assemble và validate.

Yêu cầu
- Python 3.10+
- Tạo virtual environment trong workspace.

Nhanh — PowerShell

1) Tạo và kích hoạt virtualenv

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Cài dependencies

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

3) Chạy server

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

4) Kiểm tra cơ bản
- Health: `http://127.0.0.1:8000/health`
- Demo UI: `http://127.0.0.1:8000/demo`

Gọi `/predict` (ví dụ Python)

```python
import httpx, json
body = {
  "question": "A 12 V battery drives a 3 ohm resistor. What current flows?",
  "task_type": "physics",
  "allow_llm_fallback": True
}
print(json.dumps(httpx.post('http://127.0.0.1:8000/predict', json=body, timeout=30).json(), indent=2))
```

OpenCode operator path
- Repo có một luồng operator thống nhất cho OpenCode: `exact-runner` trong [.opencode/opencode.json](/mnt/d/ura_challenge/.opencode/opencode.json) gọi `scripts/exact_agent_request.py`, script này gọi `/predict` trên cùng backend authority và chỉ báo cáo response đã được backend validate.
- Mặc định helper này nhắm tới `http://127.0.0.1:8000`. Có thể override bằng `URA_API_BASE_URL` nếu cần.

Analysis tooling
- Cụm tooling phân tích/disagreement được nhóm dưới `scripts/analysis/` để phân biệt rõ với entrypoint chính.

Pipeline configuration
- Sửa `configs/pipeline.yaml` hoặc `app/pipeline_config.py` để bật/tắt các tính năng hiện có như `enable_mcq_symbolic`, `enable_hybrid_solver`, `enable_z3_sidecar`.
- LLM backend và model được điều khiển qua biến môi trường `URA_LLM_BACKEND`, `URA_LLM_BASE_URL`, `OPENAI_BASE_URL`, `URA_LLM_MODEL`.
- `openai-compatible` là đường mặc định; `ollama` chỉ là fallback/manual smoke test.

Ghi chú
- Các module experimental (RAG, search-assisted, hybrid/Z3, LLM extractors) được lưu rời hoặc tắt theo cấu hình; core solver giữ hành vi xác định.
- Thư mục `archive/experimental_removed` lưu bản sao tạm của các module đã được di chuyển.

Hỗ trợ tiếp theo
- Muốn tôi chạy `pytest` hoặc dọn tiếp các import fallback khác không? Chỉ cần trả lời tôi sẽ làm bước tiếp theo.
