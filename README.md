URA EXACT Challenge — Quickstart

LLM Backends (Ollama / Hugging Face)
- Ollama (local): nếu bạn chạy Ollama locally, set `URA_LLM_BACKEND=ollama` và `URA_LLM_BASE_URL` tới endpoint (ví dụ `http://localhost:11434`).
  Ví dụ PowerShell:

```powershell
$env:URA_LLM_BACKEND='ollama'
$env:URA_LLM_BASE_URL='http://localhost:11434'
$env:URA_LLM_MODEL='<model-name>'
```

- Hugging Face (local or Inference API): set `URA_LLM_BACKEND='huggingface'` và `URA_HF_MODEL` (repo hoặc model id). Để dùng HF Inference API thay vì tải model local, set `HF_INFERENCE_API_KEY` theo nhu cầu triển khai.
  Ví dụ PowerShell (local/inference):

```powershell
$env:URA_LLM_BACKEND='huggingface'
$env:URA_HF_MODEL='gpt2'
```

Mục tiêu
- Ứng dụng FastAPI để trả lời câu hỏi giáo dục (physics, logic). Python-based deterministic solvers là nguồn kết quả chính; các module LLM/fallback là tuỳ chọn và chỉ bật khi cấu hình cho phép.

Yêu cầu
- Python 3.10+
- Tạo virtual environment trong workspace (khuyến nghị)

Nhanh — PowerShell (từ thư mục dự án)

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
- Demo UI: `http://127.0.0.1:8000/`

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

Tuỳ chọn LLM backend (không bắt buộc)
- Hugging Face local/inference: đặt `URA_LLM_BACKEND='huggingface'` và `URA_HF_MODEL` trước khi khởi chạy server.
- OpenAI-compatible endpoint: đặt `URA_LLM_BASE_URL` / `OPENAI_BASE_URL` và `URA_LLM_MODEL`.

Biến môi trường ví dụ (PowerShell):

```powershell
$env:URA_LLM_BACKEND='huggingface'
$env:URA_HF_MODEL='gpt2'
```

Pipeline configuration
- Sửa `configs/pipeline.yaml` hoặc `app/pipeline_config.py` để bật/tắt các tính năng như `enable_llm_fallback`, `enable_llm_explanation`, `enable_hybrid_solver`.

Ghi chú
- Các module experimental (RAG, search-assisted, hybrid/Z3, LLM extractors) được lưu rời hoặc tắt theo cấu hình; core solver giữ hành vi xác định.
- Thư mục `archive/experimental_removed` lưu bản sao tạm của các module đã được di chuyển.

Hỗ trợ tiếp theo
- Muốn tôi chạy toàn bộ `pytest` hay dọn tiếp các import fallback khác không? Chỉ cần trả lời tôi sẽ làm bước tiếp theo.
