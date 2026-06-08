# Implementation Roadmap - LLM Rescue & Missing Features

**Date:** 2026-06-05  
**Priority:** HIGH

## Overview

Hiện tại project thiếu 3 cơ chế quan trọng:

1. **LLM Rescue Fallback** — Khi Python/Z3 solver fail, cần để LLM thử làm số học/suy luận trực tiếp
2. **Search Formula Coverage** — Cơ chế search formula đã có nhưng chưa được tích hợp đầy đủ
3. **Fraction Parsing** — Parser chưa xử lý được input dạng phân số (1/3, 2/5, 1 1/2)

---

## Task 1: LLM Rescue for Physics Solver

### Priority: CRITICAL

### Current State

- `app/physics/solver.py` có `run_physics_agent()` được gọi ở cuối `_solve_impl()`
- Chỉ chạy khi `llm_client` available và `rescue_unknown=True`
- Agent-based rescue đã có nhưng có thể không đủ aggressive

### What Needs to Be Done

1. **Verify rescue path coverage** trong `_solve_impl()`:
   ```python
   # Near end of _solve_impl(), after line ~2850
   if not solution.success and llm_client and rescue_unknown:
       agent_outcome = run_physics_agent(...)
       if agent_outcome.success:
           return agent_outcome
       # ADD: If agent fails, try direct LLM call
       direct = _direct_llm_physics_rescue(question, parsed, llm_client)
       if direct.success:
           return direct
   ```

2. **Implement `_direct_llm_physics_rescue()`** trong `app/physics/solver.py`:
   ```python
   def _direct_llm_physics_rescue(
       question: str,
       parsed: ParsedPhysicsProblem,
       llm_client: Any
   ) -> PhysicsSolution:
       """Direct LLM call as last-resort physics rescue."""
       from app.physics.templates import LLM_RESCUE_PHYSICS_PROMPT
       
       quantities_str = "\n".join(
           f"- {q.original_text}: {q.si_value} {q.si_unit}"
           for q in parsed.quantities[:10]
       )
       
       prompt = LLM_RESCUE_PHYSICS_PROMPT.format(
           question=question,
           quantities=quantities_str or "None extracted"
       )
       
       try:
           response = llm_client.generate(prompt, max_tokens=512, temperature=0.0)
           # Parse response for "ANSWER: <number> <unit>"
           answer_match = re.search(r"ANSWER:\s*([\d.]+)\s*([a-zA-ZΩμ]+)", response)
           if not answer_match:
               return PhysicsSolution(
                   success=False,
                   answer="unknown",
                   explanation="LLM rescue could not extract answer",
                   formula_id=None,
                   parsed=parsed,
                   llm_rescue_used=True,
                   model_calls=1
               )
           
           value = float(answer_match.group(1))
           unit = answer_match.group(2)
           
           # Dimensional validation
           target_unit = _TARGET_UNIT_HINTS.get(parsed.target_quantity or "")
           dimensional_valid = _dimensional_agreement(unit, target_unit)
           confidence = 0.70 if dimensional_valid else 0.50
           
           answer_str = format_best_unit(value, unit)
           
           return PhysicsSolution(
               success=True,
               answer=answer_str,
               explanation=f"LLM rescue solution: {response[:200]}",
               formula_id="llm_rescue_direct",
               variables={},
               cot=[f"LLM reasoning: {response}"],
               confidence=confidence,
               parsed=parsed,
               llm_rescue_used=True,
               llm_rescue_raw_response=response,
               llm_rescue_verified=bool(dimensional_valid),
               fallback_used=True,
               model_calls=1,
               answer_source=AnswerSource.VALIDATED_LLM_PROPOSAL,
               dimensional_valid=dimensional_valid,
           )
       except Exception as e:
           return PhysicsSolution(
               success=False,
               answer="unknown",
               explanation=f"LLM rescue error: {e}",
               formula_id=None,
               parsed=parsed,
               llm_rescue_used=True,
               model_calls=1
           )
   ```

3. **Test**:
   ```bash
   wsl bash -c "cd /mnt/d/URA_challenge && source .venv/bin/activate && python -m py_compile app/physics/solver.py"
   ```

### Files to Modify

- `app/physics/solver.py` — Add `_direct_llm_physics_rescue()`, call it in `_solve_impl()`

### Success Criteria

- [ ] `_direct_llm_physics_rescue()` function exists
- [ ] Called in `_solve_impl()` after agent rescue fails
- [ ] Compilation passes
- [ ] Test with a question that deterministic solver cannot handle

---

## Task 2: LLM Rescue for Logic Solver

### Priority: HIGH

### Current State

- `app/logic/solver.py` có `run_logic_agent()` được gọi trong `solve()`
- Logic rescue đã có nhưng chỉ chạy trong một số trường hợp nhất định

### What Needs to Be Done

1. **Find the rescue invocation** trong `app/logic/solver.py` `solve()` function (line ~600+):
   - Tìm nơi `run_logic_agent()` được gọi
   - Verify nó chạy khi: FOL/Z3 fail → BFS returns None → policy returns unknown → MCQ cannot commit

2. **Add direct LLM reasoning fallback** nếu agent rescue cũng fail:
   ```python
   def _direct_llm_logic_rescue(
       question: str,
       normalized: list[Premise],
       llm_client: Any
   ) -> LogicSolution:
       """Direct LLM reasoning as final logic rescue."""
       premises_str = "\n".join(
           f"{p.id}: {p.text}" for p in normalized
       )
       
       prompt = f"""Given these premises, answer the question.

Premises:
{premises_str}

Question: {question}

Answer with yes/no/unknown and explain which premises you used.

ANSWER: <yes/no/unknown>
PREMISES: <list premise IDs>
REASONING: <explanation>
"""
       
       try:
           response = llm_client.generate(prompt, max_tokens=256, temperature=0.0)
           answer_match = re.search(r"ANSWER:\s*(yes|no|unknown)", response, re.I)
           if not answer_match:
               return LogicSolution(answer="unknown", ...)
           
           answer = answer_match.group(1).lower()
           
           # Extract premise IDs
           premise_match = re.search(r"PREMISES:\s*(.*)", response)
           cited_ids = []
           if premise_match:
               cited_ids = re.findall(r"P\d+", premise_match.group(1))
           
           # Check for hallucination
           valid_ids = {p.id for p in normalized}
           hallucinated = [pid for pid in cited_ids if pid not in valid_ids]
           if hallucinated:
               # LLM cited non-existent premises
               return LogicSolution(
                   answer="unknown",
                   explanation="LLM rescue cited premises not in input",
                   llm_fallback_used=True,
                   model_calls=1
               )
           
           return LogicSolution(
               answer=answer,
               explanation=f"LLM reasoning: {response}",
               premises=cited_ids,
               confidence=0.60 if answer != "unknown" else 0.3,
               llm_fallback_used=True,
               model_calls=1,
               answer_source=AnswerSource.VALIDATED_LLM_PROPOSAL
           )
       except Exception as e:
           return LogicSolution(answer="unknown", ...)
   ```

3. **Integrate into `solve()`**: Add call before final return unknown

4. **Test**:
   ```bash
   wsl bash -c "cd /mnt/d/URA_challenge && source .venv/bin/activate && python -m py_compile app/logic/solver.py"
   ```

### Files to Modify

- `app/logic/solver.py` — Add `_direct_llm_logic_rescue()`, call in `solve()`

### Success Criteria

- [ ] `_direct_llm_logic_rescue()` function exists
- [ ] Called in `solve()` as final fallback
- [ ] Hallucination detection works
- [ ] Compilation passes

---

## Task 3: Fraction Parsing

### Priority: MEDIUM

### Current State

- `app/physics/unit_converter.py` has `extract_quantities()`
- Uses `NUMBER_PATTERN` regex to find numeric values
- Cannot parse fractions like `1/3`, `2/5`, `1 1/2`

### What Needs to Be Done

1. **Add FRACTION_PATTERN** trong `app/physics/unit_converter.py`:
   ```python
   # After NUMBER_PATTERN definition
   FRACTION_PATTERN = r'(?:(\d+)\s+)?(\d+)/(\d+)'
   ```

2. **Update `extract_quantities()`** to handle fractions:
   ```python
   def extract_quantities(text: str) -> list[Quantity]:
       """Extract quantities including fractions."""
       text = normalize_number_words(text)
       quantities: list[Quantity] = []
       
       # First, find fractions (before general NUMBER_PATTERN)
       for match in re.finditer(FRACTION_PATTERN, text):
           whole_str, num_str, denom_str = match.groups()
           whole = int(whole_str) if whole_str else 0
           numerator = int(num_str)
           denominator = int(denom_str)
           
           if denominator == 0:
               continue  # Skip division by zero
           
           # Check if this is a formula context (skip if so)
           start, end = match.span()
           context_before = text[max(0, start-10):start]
           context_after = text[end:min(len(text), end+10)]
           
           # Skip if looks like a formula (has ^, _, uppercase vars)
           if re.search(r'[\^_]|[A-Z]\d', context_before + context_after):
               continue
           
           # Convert to float
           value = whole + (numerator / denominator)
           
           # Look for unit after fraction
           unit_match = re.match(r'\s*([a-zA-ZΩμ]+)', context_after)
           if unit_match:
               unit_str = unit_match.group(1)
               # Create Quantity
               # ... (same logic as existing code)
       
       # Continue with existing NUMBER_PATTERN logic...
       # (existing code)
   ```

3. **Test cases** to add:
   ```python
   # Test 1: Simple fraction
   qs = extract_quantities("A capacitor of 1/3 F")
   assert any(abs(q.si_value - 1/3) < 1e-9 for q in qs)
   
   # Test 2: Fraction with prefix
   qs = extract_quantities("Resistance 2/5 kΩ")
   assert any(abs(q.si_value - 400.0) < 1e-6 for q in qs)
   
   # Test 3: Mixed number
   qs = extract_quantities("Distance 1 1/2 m")
   assert any(abs(q.si_value - 1.5) < 1e-9 for q in qs)
   
   # Test 4: Don't break formulas
   qs = extract_quantities("Power P = V^2/R where R = 10 Ω")
   # Should NOT extract a fraction from V^2/R
   ```

4. **Compile check**:
   ```bash
   wsl bash -c "cd /mnt/d/URA_challenge && source .venv/bin/activate && python -m py_compile app/physics/unit_converter.py app/physics/parser.py"
   ```

### Files to Modify

- `app/physics/unit_converter.py` — Add `FRACTION_PATTERN`, update `extract_quantities()`

### Success Criteria

- [ ] FRACTION_PATTERN defined
- [ ] `extract_quantities()` handles simple and mixed fractions
- [ ] Formula contexts (V^2/R, 1/R1) are NOT parsed as values
- [ ] Compilation passes
- [ ] Manual test cases work

---

## Execution Order

**Recommended sequence:**

1. **Task 1 (Physics LLM Rescue)** — CRITICAL, enables LLM to handle unseen physics problems
2. **Task 2 (Logic LLM Rescue)** — HIGH, enables LLM to handle complex logic when BFS fails
3. **Task 3 (Fraction Parsing)** — MEDIUM, expands coverage for fractional inputs

**Parallel execution:** Tasks 1, 2, 3 are independent and can be done in parallel if multiple people/agents are available.

---

## Testing Plan

After all 3 tasks complete:

1. **Manual smoke test**: Run `scripts/real_smoke_tests.py` to verify nothing broke
2. **Router tests**: `pytest tests/test_router.py -q` (should still pass existing tests)
3. **Physics tests**: `pytest tests/test_physics_solver.py -q` (if exists)
4. **Logic tests**: `pytest tests/test_logic_solver.py -q`

---

## Notes

- All work should be done in WSL at `/mnt/d/URA_challenge`
- Use vLLM server at `http://192.168.30.3:8001/v1` with model `Qwen/Qwen2.5-3B-Instruct-AWQ`
- Always activate `.venv` before running commands
- Run `python -m py_compile <file>` after each change to catch syntax errors early
- Follow AGENTS.md rules: no hardcoded question-specific overrides, only component-level fixes
