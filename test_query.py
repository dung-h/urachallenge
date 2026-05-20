import os, json, traceback
os.environ['URA_LLM_BACKEND'] = 'huggingface'
os.environ['URA_HF_MODEL'] = 'gpt2'
try:
    from app.router import predict_with_metadata
    from app.schemas import QARequest
    req = QARequest(question='A 12 V battery drives a 3 ohm resistor. What current flows?', task_type='physics', allow_llm_fallback=True)
    resp, meta = predict_with_metadata(req, write_trace=False)
    out = {
        'response': resp.model_dump() if hasattr(resp, 'model_dump') else resp.__dict__,
        'metadata': meta,
    }
    print(json.dumps(out, default=str, indent=2))
except Exception as e:
    traceback.print_exc()
    print('ERROR:', str(e))
