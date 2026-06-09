"""Mock OpenAI-compatible `/v1/chat/completions` server for local testing.

Use this when you want to exercise the app's OpenAI-compatible client wiring without
running vLLM/llama-server. The responses are simplistic and intended only to validate:

- request/response wiring
- routing/orchestrator JSON parsing
- explanation rewrite path

This is NOT a benchmark backend.
"""

# scripts/mock_openai_server.py
import json
import re
import time
import sys
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Mock OpenAI-compatible Server for URA Challenge")

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    messages = payload.get("messages", [])
    model = payload.get("model", "Qwen/Qwen2.5-7B-Instruct")
    
    system_prompt = ""
    user_prompt = ""
    for msg in messages:
        if msg.get("role") == "system":
            system_prompt = msg.get("content", "")
        elif msg.get("role") == "user":
            user_prompt = msg.get("content", "")
            
    content = ""
    
    # 1. Main Orchestrator Routing
    if "main_orchestrator" in system_prompt or "You are the main orchestrator" in system_prompt:
        try:
            req_data = json.loads(user_prompt)
            question = req_data.get("question", "")
        except Exception:
            question = user_prompt
            
        low_q = question.lower()
        task_type = "logic"
        try:
            from app.runtime_workflow import TaskRouter, InputNormalizer
            from app.schemas import QARequest
            req = QARequest(question=question)
            norm = InputNormalizer().normalize(req)
            task_type = TaskRouter().route(norm).value
        except Exception:
            if any(k in low_q for k in ("resistor", "battery", "ohm", "volt", "current", "power", "capacit", "field", "potential", "wave", "frequency", "charge", "solenoid")):
                task_type = "physics"
            
        if task_type == "physics":
            plan = {
                "task_type": "physics",
                "route_reason": "simple physics query detected by mock orchestrator",
                "confidence": 0.95,
                "use_search": True,
                "use_llm_reasoner": False,
                "use_explanation_rewrite": True,
                "rescue_unknown": True,
                "search_queries": [],
                "physics_hint": {"target_quantity": "voltage"},
                "logic_hint": {}
            }
        else:
            plan = {
                "task_type": "logic",
                "route_reason": "simple logic/academic policy query detected by mock orchestrator",
                "confidence": 0.9,
                "use_search": False,
                "use_llm_reasoner": True,
                "use_explanation_rewrite": True,
                "rescue_unknown": True,
                "search_queries": [],
                "physics_hint": {},
                "logic_hint": {"domain": "academic_policy"}
            }
        content = json.dumps(plan)
        
    # 2. Explanation Rewrite Worker
    elif "explanation_rewrite" in system_prompt or "You are an explanation worker" in system_prompt:
        try:
            trace_data = json.loads(user_prompt)
            task_type = trace_data.get("task_type", "physics")
            answer = trace_data.get("answer", "unknown")
        except Exception:
            answer = "unknown"
            task_type = "physics"
            
        if task_type == "physics":
            content = json.dumps({
                "explanation": f"Based on physical calculation, the computed answer is {answer}."
            })
        else:
            content = json.dumps({
                "explanation": f"Based on academic policy rules, the final answer is {answer}. Maya has GPA 3.7 which meets the GPA threshold, but the missing faculty nomination condition is not satisfied. Cited: P1, P2."
            })
            
    # 3. Default fallback
    else:
        content = "Concise answer from mock LLM server."
        
    response_data = {
        "id": "chatcmpl-mock1234567890",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 15,
            "total_tokens": 25
        }
    }
    return JSONResponse(content=response_data)

if __name__ == "__main__":
    import uvicorn
    port = 8001
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    uvicorn.run(app, host="127.0.0.1", port=port)
