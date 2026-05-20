# URA EXACT Challenge - Quick Start Guide

Local, explainable QA system for educational questions (logic & physics).

## What It Does

- **Physics problems**: Solves V=IR, P=VI, resistance/capacitance formulas with Python, returns answer + explanation
- **Logic problems**: Answers questions based on given premises, validates with backend
- **JSON output**: Always validates response through Pydantic before returning

## System Requirements

- **OS**: Windows with WSL2, or native Linux/Mac
- **Python**: 3.10+ (check: `python --version`)
- **Disk**: ~500MB free
- **RAM**: 2GB minimum
- **GPU**: Optional (not required)

## Quick Start (5 min)

### 1. Activate Environment

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux/Mac
source .venv/bin/activate

# Verify
python --version
```

If `.venv` doesn't exist, create it:

```bash
python -m venv .venv
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Optional (only if you want Z3-based logic features):

```bash
pip install z3-solver
```

Verify:

```bash
pip list | grep -E "fastapi|pydantic|uvicorn"
```

Windows PowerShell equivalent:

```powershell
pip list | Select-String "fastapi|pydantic|uvicorn"
```

### 3. Start API Server

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Notes:
- Use `http://127.0.0.1:8000` from the same machine.
- `--host 0.0.0.0` avoids IPv4/IPv6 loopback binding surprises on some Windows setups.
- If you have Docker apps running (DVWA / Juice Shop), port `8000` is commonly taken. If `/health` does not return `{"status":"ok"}`, switch to another port (example: `8002`).

Quick sanity check (should return `{"status":"ok"}`):

```bash
python -c "import httpx; print(httpx.get('http://127.0.0.1:8000/health', timeout=5).text)"
```

Output should show:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### 4. Test Endpoints (Choose Your Method)

#### Option A: Windows PowerShell

Physics question:

```powershell
$body = @{
    question = "In a circuit with V=12V and R=4 ohms, what is the current?"
    task_type = "physics"
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body $body

$response | ConvertTo-Json -Depth 10
```

Logic question:

```powershell
$body = @{
    question = "Is John riding a bicycle?"
    premises = @("John owns a bicycle", "John is riding his bicycle")
    task_type = "logic"
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body $body

$response | ConvertTo-Json -Depth 10
```

If `curl` in your PowerShell maps to `Invoke-WebRequest`, use `curl.exe` explicitly:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"In a circuit with V=12V and R=4 ohms, what is the current?\",\"task_type\":\"physics\"}"
```

#### Option B: Python (Recommended)

Physics question:

```python
import json
import httpx

response = httpx.post(
    "http://127.0.0.1:8000/predict",
    json={
        "question": "In a circuit with V=12V and R=4 ohms, what is the current?",
        "task_type": "physics",
    },
    timeout=10,
)

print(json.dumps(response.json(), indent=2))
```

Logic question:

```python
import json
import httpx

response = httpx.post(
    "http://127.0.0.1:8000/predict",
    json={
        "question": "Is John riding a bicycle?",
        "premises": [
            "John owns a bicycle",
            "John is riding his bicycle",
        ],
        "task_type": "logic",
    },
    timeout=10,
)

print(json.dumps(response.json(), indent=2))
```

#### Option C: Linux/Mac (bash + curl)

Physics question:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "question": "In a circuit with V=12V and R=4 ohms, what is the current?",
    "task_type": "physics"
  }' | jq .
```

Logic question:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Is John riding a bicycle?",
    "premises": ["John owns a bicycle", "John is riding his bicycle"],
    "task_type": "logic"
  }' | jq .
```

#### Option D: JavaScript (Browser/Node.js)

Physics question:

```javascript
const response = await fetch("http://127.0.0.1:8000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    question: "In a circuit with V=12V and R=4 ohms, what is the current?",
    task_type: "physics"
  })
});

const data = await response.json();
console.log(data);
```

Logic question:

```javascript
const response = await fetch("http://127.0.0.1:8000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    question: "Is John riding a bicycle?",
    premises: [
      "John owns a bicycle",
      "John is riding his bicycle"
    ],
    task_type: "logic"
  })
});

const data = await response.json();
console.log(data);
```

#### Option E: Web UI (Browser Only)

Open: `http://127.0.0.1:8000/` (redirects to `/demo`)

Fill form, click "Run /predict", see results instantly.

### Expected Response Format

All endpoints return JSON:

```json
{
  "answer": "yes | no | unknown | <number> <unit>",
  "explanation": "Human-readable explanation",
  "premises": ["P0", "P1"],
  "cot": ["step 1", "step 2"],
  "fol": "formula_id | logic_statement",
  "confidence": 0.0-1.0,
  "task_type": "physics | logic",
  "raw_json_validity": true | false,
  "repaired_json_validity": true | false
}
```

### 5. Using Web UI

## Supported Physics Formulas

| Formula | Variables | Example |
|---------|-----------|---------|
| Ohm's Law | I = V / R | Current from voltage & resistance |
| Power (Voltage) | P = V * I | Power from voltage & current |
| Power (Resistance) | P = I² * R | Power from current & resistance |
| Power (Voltage/R) | P = V² / R | Power from voltage & resistance |
| Series Resistance | R_total = R1 + R2 + ... | Combined series resistors |
| Parallel Resistance | 1/R_total = 1/R1 + 1/R2 + ... | Combined parallel resistors |
| Capacitor Charge | Q = C * V | Charge from capacitance & voltage |
| Capacitor Energy | E = 0.5 * C * V² | Energy stored in capacitor |
| Capacitor Series | 1/C_total = 1/C1 + 1/C2 + ... | Series capacitors |
| Capacitor Parallel | C_total = C1 + C2 + ... | Parallel capacitors |
| Coulomb Force | F = k * q1 * q2 / r² | Force between charges |
| Electric Field (Force) | E = F / q | Field from force & charge |
| Electric Field (Coulomb) | E = k * q / r² | Field from charge & distance |

**Units supported**: V (volts), A (amps), ohm (Ω), W (watts), F (farads), C (coulombs), J (joules), N (newtons)

**Prefixes**: m (milli), u (micro), k (kilo), M (mega)

## Supported Logic Reasoning

- Single-hop entailment (Premise → Answer)
- Multiple premises (select relevant ones)
- Yes/No/Unknown answers
- Premise ID tracking
- Proof trace (reasoning steps)

## Run Tests

```bash
pytest tests/ -v
```

Run specific test:

```bash
pytest tests/test_physics_solver.py -v
```

## File Structure

```
.
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── app/
│   ├── main.py                   # FastAPI app entry point
│   ├── router.py                 # /predict endpoint
│   ├── schemas.py                # Request/Response models
│   ├── physics/
│   │   ├── solver.py            # Physics formula solver
│   │   ├── formulas.py          # Formula registry
│   │   └── unit_converter.py    # Unit conversion
│   └── logic/
│       ├── solver.py            # Logic reasoning
│       └── premise_selector.py  # Premise matching
├── tests/                        # Test suite (69 tests)
├── outputs/
│   └── traces/                  # Request/response logs
└── reports/                      # Analysis & benchmarks
```

## Common Issues

### "ModuleNotFoundError: No module named 'app'"

**Fix**: Run from project root directory
```powershell
Set-Location -LiteralPath "D:\URA_challenge"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### "Address already in use" (port 8000)

**Fix**: Use different port
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

Or kill existing process:
```powershell
# Find process listening on port 8000 and kill it
$p = (Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique)
if ($p) { Stop-Process -Id $p -Force }
```

### Browser app cannot call API (CORS error)

If your frontend runs on a different origin (example: `http://localhost:5173`) the browser may block the request.

Options:
- Use the built-in UI at `http://127.0.0.1:8000/demo` (same origin)
- Add CORS middleware in `app/main.py` for your frontend origin

### Physics question returns "unknown"

**Causes**:
- Question doesn't match physics hints (voltage, current, resistance, etc.)
- Formula not in registry
- Not enough variables provided

**Fix**: Use explicit `"task_type": "physics"` in request

### Dependencies installation fails

**Fix**:
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --no-cache-dir
```

## API Reference

### POST /predict

Request:
```json
{
  "question": "string (required)",
  "premises": ["string"] (optional),
  "task_type": "physics|logic|auto" (default: "auto")
}
```

Response:
```json
{
  "answer": "string",
  "explanation": "string",
  "premises": ["string"],
  "cot": ["string"],
  "fol": "string (FOL formula)",
  "confidence": 0.0-1.0,
  "task_type": "physics|logic",
  "raw_json_validity": boolean,
  "repaired_json_validity": boolean
}
```

Response headers:
- `X-Request-ID`: server-generated request id
- `X-Trace-URL`: relative URL to fetch the trace JSON (example: `/trace/<request_id>`)

### GET /trace/{request_id}

Returns the server-side trace record written under `outputs/traces/production_like/`.

### GET /health

Returns:
```json
{"status": "ok"}
```

### GET /demo

Opens interactive web UI for testing

## Production Defaults

- Physics arithmetic: always recomputed by Python (not LLM)
- Premise validation: checked against supplied premises (no hallucination)
- JSON output: always validated by Pydantic schema
- Traces: saved to `outputs/traces/production_like/` for audit

## Architecture

```
User Question
    ↓
[Route Task] → Auto-detect physics vs logic
    ↓
PHYSICS PATH              | LOGIC PATH
├ Parse question          | ├ Normalize premises
├ Extract variables       | ├ Select relevant premises
├ Match formula           | └ Apply reasoning rules
├ Unit conversion         |
├ Python computation      |
└ Generate trace          |
    ↓
[Validate JSON]
    ↓
Return Response
```

## Advanced: Local LLM Fallback (Optional)

The system can optionally use local LLMs for:
- Premise selection
- Explanation rewriting
- Unknown answer handling

This is disabled by default. To enable, see `docs/live_fallback_findings.md`.

## Benchmarks

- Physics accuracy: **100%** (deterministic solver)
- Logic accuracy: **98.8%** (rule-based + LLM fallback)
- JSON validity: **100%**
- Hallucinated premise rate: **0%**
- Crash rate: **0%**

See `reports/` for detailed benchmark reports.

## Support & Feedback

- Check error message in response
- Review `outputs/traces/` for request/response logs
- Search `tests/` for similar test cases
- Read `docs/` for technical details

## License

[Specify your license]

## Project Status

✅ Production ready  
✅ All tests passing  
✅ API responsive  
✅ Physics solver 100% accurate  
✅ Logic solver 98.8% accurate
