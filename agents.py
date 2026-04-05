import os
import json
import sympy as sp
from groq import Groq
from dotenv import load_dotenv
from rag_pipeline import retrieve
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.3-70b-versatile"


def call_llm(system_prompt: str, user_message: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content.strip()


def solve_implicit_differentiation(problem_text: str, equation_str: str, x_val: float) -> str:
    try:
        x, y = sp.symbols('x y')

        eq_prompt = f"""Convert this math equation to a sympy expression equal to zero.
Problem: {problem_text}
Equation hint: {equation_str}

Rules:
- Use x and y as variables
- Move everything to left side (= 0)
- Use sp.ln() for natural log
- Use sp.sin, sp.cos, sp.exp etc
- Return ONLY the sympy expression, nothing else
- Example: for ln(x+y) = 4xy return: sp.ln(x+y) - 4*x*y

Return the expression only, no code, no explanation."""

        eq_str = call_llm("Return sympy expression only.", eq_prompt).strip()
        eq_str = eq_str.replace("```", "").strip()

        namespace = {
            "sp": sp, "x": x, "y": y,
            "ln": sp.ln, "log": sp.ln,
            "sin": sp.sin, "cos": sp.cos,
            "tan": sp.tan, "exp": sp.exp,
            "sqrt": sp.sqrt
        }
        eq = eval(eq_str, namespace)

        y_val = sp.solve(eq.subs(x, x_val), y)
        if not y_val:
            return f"Could not find y at x={x_val}"
        y_val = y_val[0]

        dydx   = sp.idiff(eq, y, x)
        d2ydx2 = sp.idiff(eq, y, x, 2)

        dydx_at = dydx.subs([(x, x_val), (y, y_val)])
        d2y_at  = d2ydx2.subs([(x, x_val), (y, y_val)])

        return (
            f"y at x={x_val}: {y_val}\n"
            f"dy/dx at x={x_val}: {sp.nsimplify(sp.re(dydx_at))}\n"
            f"d2y/dx2 at x={x_val}: {sp.nsimplify(sp.re(d2y_at))}"
        )

    except Exception as e:
        return f"Implicit diff error: {str(e)}"


# ── Agent 1: Parser ───────────────────────────────────────────────────────────
def parser_agent(raw_input: str) -> dict:
    system = """You are a math problem parser for JEE-level questions.
Clean and structure raw input that may come from OCR or speech.

Return ONLY a valid JSON object:
{
  "problem_text": "cleaned and complete problem statement",
  "topic": "one of: algebra | probability | calculus | linear_algebra | general",
  "variables": ["list", "of", "variables"],
  "constraints": ["list of constraints if any"],
  "needs_clarification": false,
  "clarification_reason": ""
}

Rules:
- Fix OCR errors (e.g. '0' vs 'O', '1' vs 'l')
- Fix speech-to-text math phrases (e.g. 'square root of x' → 'sqrt(x)')
- Set needs_clarification to true only if the problem is genuinely ambiguous
- Return ONLY the JSON, no explanation, no markdown"""

    result = call_llm(system, raw_input).strip()
    if result.startswith("```"):
        result = result.split("```")[1]
        if result.startswith("json"):
            result = result[4:]
    result = result.strip()

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return {
            "problem_text":        raw_input,
            "topic":               "general",
            "variables":           [],
            "constraints":         [],
            "needs_clarification": True,
            "clarification_reason": "Could not parse problem structure."
        }


# ── Agent 2: Intent Router ────────────────────────────────────────────────────
def intent_router_agent(parsed_problem: dict) -> dict:
    system = """You are a math problem classifier for JEE-level questions.
Given a structured math problem, decide the solution strategy.

Return ONLY a valid JSON object:
{
  "topic": "algebra | probability | calculus | linear_algebra | general",
  "subtopic": "specific subtopic e.g. quadratic_equations, derivatives, matrices",
  "difficulty": "easy | medium | hard",
  "strategy": "brief description of approach to solve this problem",
  "tools_needed": ["calculator", "formula_lookup"]
}

Return ONLY the JSON, no explanation, no markdown."""

    result = call_llm(system, json.dumps(parsed_problem)).strip()
    if result.startswith("```"):
        result = result.split("```")[1]
        if result.startswith("json"):
            result = result[4:]
    result = result.strip()

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return {
            "topic":        parsed_problem.get("topic", "general"),
            "subtopic":     "unknown",
            "difficulty":   "medium",
            "strategy":     "Solve step by step using relevant formulas.",
            "tools_needed": []
        }


# ── Agent 3: Solver ───────────────────────────────────────────────────────────
def solver_agent(parsed_problem: dict, routing_info: dict, memory_context: str = "") -> dict:
    from calculator import execute_math_code
    from wolfram_solver import query_wolfram, format_for_wolfram

    query        = parsed_problem["problem_text"]
    topic        = routing_info.get("topic", "general")
    subtopic     = routing_info.get("subtopic", "")
    context_docs = retrieve(query, top_k=3)
    context_text = "\n\n".join([f"[{d['title']}]\n{d['content']}" for d in context_docs])

    wolfram_context = ""
    calc_context    = ""
    generated_code  = ""
    calc_output     = ""
    query_lower     = query.lower()

    is_implicit = (
        any(k in query_lower for k in [
            "d2y", "d^2y", "dy/dx", "y''", "find d2y",
            "find dy/dx", "implicit differentiation"
        ]) or (
            any(k in query_lower for k in ["ln(x+y)", "log(x+y)", "loge(x+y)"]) and
            any(k in query_lower for k in ["dy", "d2y", "derivative of y", "at x="])
        )
    )

    # Handle implicit differentiation directly with SymPy
    if is_implicit and not any(k in query_lower for k in ["parabola", "closest", "distance"]):
        import re
        x_match = re.search(r'at x\s*=\s*(-?\d+\.?\d*)', query_lower)
        x_val   = float(x_match.group(1)) if x_match else 0.0

        implicit_result = solve_implicit_differentiation(query, query, x_val)
        context_docs    = retrieve(query, top_k=3)
        context_text    = "\n\n".join([f"[{d['title']}]\n{d['content']}" for d in context_docs])

        system   = f"""You are an expert JEE math solver.
Use the exact computation result below to write the solution.

COMPUTATION RESULT (use these exact values):
{implicit_result}

REFERENCE MATERIAL:
{context_text}"""
        solution = call_llm(system, f"Problem: {query}\n\nExplain the solution.")

        return {
            "solution":       solution,
            "sources_used":   [d["title"] for d in context_docs],
            "context_docs":   context_docs,
            "generated_code": "sp.idiff() used directly",
            "calc_output":    implicit_result
        }

    use_wolfram = not is_implicit and (
        any(k in topic.lower() for k in ["calculus", "algebra"]) or
        any(k in subtopic.lower() for k in [
            "derivative", "integral", "integration", "limit",
            "maximum", "minimum", "roots", "equation",
            "differentiation", "optimization", "decreasing",
            "increasing", "critical"
        ])
    )

    use_calculator = any(k in topic.lower() for k in ["probability", "combinatorics"]) or \
                     any(k in subtopic.lower() for k in [
                         "combination", "permutation", "probability", "distribution",
                         "variance", "expectation", "hypergeometric", "binomial", "committee"
                     ])

    # Tool 1: Wolfram Alpha
    if use_wolfram:
        wolfram_query  = format_for_wolfram(query)
        wolfram_result = query_wolfram(wolfram_query)

        if wolfram_result["success"] and wolfram_result["answer"]:
            context_str = "\n".join([
                f"[{r['title']}]: {r['content'][:200]}"
                for r in wolfram_result["all_results"][:5]
            ])
            wolfram_context = f"""WOLFRAM ALPHA RESULT:
Answer: {wolfram_result['answer']}

{context_str}"""
        else:
            use_wolfram    = False
            use_calculator = True

    # Tool 2: Python + SymPy calculator
    if use_calculator and not use_wolfram:
        code_prompt = f"""You are an expert JEE math solver.
Write Python code to solve this problem and print the final answer.

Available tools (no imports needed):
- SymPy: symbols, diff, solve, integrate, limit, simplify, ln, Abs, Rational, sp.re
- Numeric: comb(n,r), perm(n,r), factorial(n), Fraction(a,b)

# Probability problems
- For "graph strictly above x-axis": condition is discriminant D < 0
- Count favorable integers in range, express as Fraction(favorable, total)
- Never print probability as decimal

# Distribution rules
- Without replacement → Hypergeometric: Var(X) = n*(K/N)*(1-K/N)*(N-n)/(N-1)
- With replacement → Binomial: Var(X) = n*p*(1-p)

# Calculus rules
- Absolute value functions: split into cases manually
- Clean imaginary parts: sp.re(sp.simplify(result))

# Implicit differentiation
- Always use sp.idiff()
- Use plain sp.symbols('x y'), never sp.Function('y')
- Substitute actual numbers directly, not placeholder variables

- Always verify answer makes sense before printing
- Raw code only, no markdown

REFERENCE MATERIAL:
{context_text}

Problem: {query}
Topic: {topic} / {subtopic}
Strategy: {routing_info.get('strategy', '')}"""

        generated_code = call_llm("Write clean Python math code. Raw code only, no markdown.", code_prompt).strip()

        if "```python" in generated_code:
            generated_code = generated_code.split("```python")[1].split("```")[0].strip()
        elif "```" in generated_code:
            generated_code = generated_code.split("```")[1].split("```")[0].strip()

        calc_result = execute_math_code(generated_code)

        if calc_result["success"] and calc_result["output"]:
            calc_output  = calc_result["output"]
            calc_context = f"PYTHON CALCULATION RESULT:\n{calc_output}"
        else:
            retry_prompt = f"""Previous code had error: {calc_result['error']}
Fix and rewrite for: {query}
Raw code only, no markdown, no imports."""

            generated_code = call_llm("Fix Python math code. Raw code only.", retry_prompt).strip()
            if "```python" in generated_code:
                generated_code = generated_code.split("```python")[1].split("```")[0].strip()
            elif "```" in generated_code:
                generated_code = generated_code.split("```")[1].split("```")[0].strip()

            calc_result = execute_math_code(generated_code)
            if calc_result["success"] and calc_result["output"]:
                calc_output  = calc_result["output"]
                calc_context = f"PYTHON CALCULATION RESULT:\n{calc_output}"
            else:
                calc_context = "Calculator unavailable. Solve carefully step by step."

    computation_result = wolfram_context if wolfram_context else calc_context
    memory_section     = f"VERIFIED PAST SOLUTIONS:\n{memory_context}\n" if memory_context else ""

    system = f"""You are an expert JEE math solver.

REFERENCE MATERIAL:
{context_text}

{memory_section}
{computation_result}

Instructions:
- Use the exact numbers from the computation result above
- Show method and reasoning clearly
- Do not redo arithmetic — trust the computation result
- Be concise but complete"""

    solution = call_llm(system, f"Problem: {query}\n\nWrite the complete solution.")

    return {
        "solution":       solution,
        "sources_used":   [d["title"] for d in context_docs],
        "context_docs":   context_docs,
        "generated_code": generated_code,
        "calc_output":    calc_output if calc_output else (
            wolfram_result.get("answer", "") if "wolfram_result" in dir() else ""
        )
    }


# ── Agent 4: Verifier ─────────────────────────────────────────────────────────
def llm_verify_probability(problem_text: str, solution_data: dict) -> dict:
    system = """You are a strict math verifier for probability and counting problems.
Re-derive the answer independently then compare with the given solution.
Return ONLY a valid JSON object, nothing else:
{"is_correct": true, "confidence": 0.85, "issues_found": [], "corrected_answer": "", "needs_human_review": false, "review_reason": "", "derived_answer": ""}"""

    user_msg = f"""Problem: {problem_text}

Solution to verify:
{solution_data['solution'][:500]}

Derive the answer yourself first, then check if it matches. Return ONLY the JSON."""

    result = call_llm(system, user_msg).strip()

    if "```" in result:
        parts = result.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                result = part
                break

    start = result.find("{")
    end   = result.rfind("}") + 1
    if start != -1 and end > start:
        result = result[start:end]

    try:
        parsed = json.loads(result)
        if not parsed.get("is_correct"):
            parsed["needs_human_review"] = True
        return parsed
    except json.JSONDecodeError:
        return {
            "is_correct":         True,
            "confidence":         0.75,
            "issues_found":       [],
            "corrected_answer":   "",
            "needs_human_review": True,
            "review_reason":      "Auto-verify inconclusive — please confirm manually"
        }


def llm_fallback_verifier(problem_text: str, solution_data: dict, parse_error: str) -> dict:
    system = """You are a strict math solution verifier.
Return ONLY valid JSON:
{"is_correct": false, "confidence": 0.6, "issues_found": [], "corrected_answer": "", "needs_human_review": true, "review_reason": ""}"""

    user_msg = f"""Problem: {problem_text}
Solution: {solution_data['solution']}
Note: Automatic verification failed: {parse_error}
Check manually and return JSON."""

    result = call_llm(system, user_msg).strip()
    if result.startswith("```"):
        result = result.split("```")[1]
        if result.startswith("json"):
            result = result[4:]
    result = result.strip()

    try:
        parsed = json.loads(result)
        parsed["needs_human_review"] = True
        parsed["confidence"]         = min(parsed.get("confidence", 0.6), 0.65)
        return parsed
    except json.JSONDecodeError:
        return {
            "is_correct":         False,
            "confidence":         0.5,
            "issues_found":       ["Verification failed completely"],
            "corrected_answer":   "",
            "needs_human_review": True,
            "review_reason":      "Both symbolic and LLM verification failed"
        }


def verify_solution_symbolically(problem_text: str, solution_data: dict) -> dict:
    probability_keywords = [
        "probability", "how many values", "number of values",
        "number of integers", "how many integers", "strictly above",
        "strictly below", "lies above", "lies below", "counting",
        "favorable", "sample space"
    ]

    if any(k in problem_text.lower() for k in probability_keywords):
        return llm_verify_probability(problem_text, solution_data)

    try:
        extraction_prompt = f"""Extract from this problem and solution:
1. The original equation (left side and right side separately)
2. All candidate values of x that were found

Problem: {problem_text}
Solution: {solution_data['solution']}

Return ONLY valid JSON:
{{
  "variable": "x",
  "lhs": "abs(3*x**2 + 12*x + 6)",
  "rhs": "5*x + 16",
  "candidates": [1, -3.333, -2, -3.667]
}}

Use abs() for absolute values, sqrt() for roots. Return ONLY the JSON."""

        extracted = call_llm("You are a math expression extractor. Return only JSON.", extraction_prompt).strip()
        if extracted.startswith("```"):
            extracted = extracted.split("```")[1]
            if extracted.startswith("json"):
                extracted = extracted[4:]
        extracted = extracted.strip()

        parsed   = json.loads(extracted)
        x        = sp.Symbol('x')
        lhs_expr = sp.sympify(parsed['lhs'])
        rhs_expr = sp.sympify(parsed['rhs'])

        valid_solutions   = []
        invalid_solutions = []

        for candidate in parsed['candidates']:
            try:
                lhs_val = float(lhs_expr.subs(x, candidate))
                rhs_val = float(rhs_expr.subs(x, candidate))

                if abs(lhs_val - rhs_val) < 0.001:
                    valid_solutions.append(candidate)
                else:
                    invalid_solutions.append({
                        "value":      candidate,
                        "lhs_result": round(lhs_val, 4),
                        "rhs_result": round(rhs_val, 4),
                        "difference": round(abs(lhs_val - rhs_val), 4)
                    })
            except Exception as e:
                invalid_solutions.append({"value": candidate, "error": str(e)})

        issues = []
        for inv in invalid_solutions:
            if "error" in inv:
                issues.append(f"x = {inv['value']} caused error: {inv['error']}")
            else:
                issues.append(f"x = {inv['value']} is INVALID: LHS={inv['lhs_result']} ≠ RHS={inv['rhs_result']}")

        is_correct = len(invalid_solutions) == 0 and len(valid_solutions) > 0

        return {
            "is_correct":         is_correct,
            "confidence":         0.99 if is_correct else 0.95,
            "valid_solutions":    valid_solutions,
            "invalid_solutions":  invalid_solutions,
            "issues_found":       issues,
            "corrected_answer":   f"Valid solutions: {valid_solutions}" if valid_solutions else "No valid solutions found",
            "needs_human_review": len(valid_solutions) == 0,
            "review_reason":      "No valid solutions passed substitution check" if len(valid_solutions) == 0 else ""
        }

    except Exception as e:
        return llm_fallback_verifier(problem_text, solution_data, str(e))


def verifier_agent(parsed_problem: dict, solution_data: dict) -> dict:
    return verify_solution_symbolically(parsed_problem['problem_text'], solution_data)


# ── Agent 5: Explainer ────────────────────────────────────────────────────────
def explainer_agent(parsed_problem: dict, solution_data: dict, verification: dict) -> str:
    final_solution = (
        verification["corrected_answer"]
        if verification.get("corrected_answer")
        else solution_data["solution"]
    )

    system = """You are a friendly JEE math tutor explaining solutions to Class 11-12 students.

Format your explanation as:
## Understanding the Problem
## Key Concepts Used
## Step-by-Step Solution
## Final Answer
## Tips to Remember

Keep language simple and precise."""

    return call_llm(system, f"Problem: {parsed_problem['problem_text']}\n\nSolution:\n{final_solution}\n\nExplain this clearly.")


# ── Master Pipeline ───────────────────────────────────────────────────────────
def run_pipeline(raw_input: str) -> dict:
    from memory import retrieve_similar

    print("\n[1/5] Parser Agent running...")
    parsed  = parser_agent(raw_input)

    print("[2/5] Intent Router Agent running...")
    routing = intent_router_agent(parsed)

    similar_memories   = retrieve_similar(raw_input, top_k=2)
    successful_similar = similar_memories.get("successful", [])
    correction_similar = similar_memories.get("corrections", [])

    verified_similar = [
        s for s in successful_similar
        if s.get("similarity_score", 999) < 2.0
    ]

    memory_context = ""

    if verified_similar:
        print(f"[Memory] Found {len(verified_similar)} similar correct problem(s) — using as reference")
        memory_context += "\n\n".join([
            f"VERIFIED PAST SOLUTION:\nProblem: {s['problem_text']}\nSolution: {s['solution']}\nTopic: {s['topic']}"
            for s in verified_similar
        ])

    if correction_similar:
        print(f"[Memory] Found {len(correction_similar)} similar incorrect problem(s) — warning solver")
        memory_context += "\n\n" + "\n\n".join([
            f"WARNING — SYSTEM WAS WRONG ON SIMILAR PROBLEM:\nProblem: {s['problem_text']}\n"
            f"Wrong answer: {s['solution'][:150]}\nCorrect answer: {s['user_correction']}\nDo not repeat this mistake."
            for s in correction_similar
        ])

    print("[3/5] Solver Agent running...")

    # Serve stored correct answer directly if available
    direct_answer = None
    for s in correction_similar:
        if s.get("user_correction") and s.get("similarity_score", 999) < 0.8:
            direct_answer = s["user_correction"]
            print("[Memory] Found stored correct answer — serving directly")
            break

    if direct_answer:
        solution = {
            "solution":       direct_answer,
            "sources_used":   ["Memory — User Verified Correction"],
            "context_docs":   [],
            "generated_code": "",
            "calc_output":    direct_answer
        }
        verification = {
            "is_correct":         True,
            "confidence":         0.99,
            "issues_found":       [],
            "corrected_answer":   direct_answer,
            "needs_human_review": False,
            "review_reason":      ""
        }
        print("[4/5] Skipping verifier — using human-verified answer")
        explanation = explainer_agent(parsed, solution, verification)
        return {
            "parsed_problem":   parsed,
            "routing_info":     routing,
            "solution_data":    solution,
            "verification":     verification,
            "explanation":      explanation,
            "memory_used":      verified_similar,
            "corrections_used": correction_similar
        }

    solution = solver_agent(parsed, routing, memory_context=memory_context)

    print("[4/5] Verifier Agent running...")
    verification = verifier_agent(parsed, solution)

    # Correction loop — retry with different strategy if answer fails verification
    if not verification.get("is_correct") and not verification.get("needs_human_review"):
        print("[4b] Verifier rejected answer — retrying with different strategy...")

        retry_context = f"""PREVIOUS ATTEMPT FAILED:
Wrong answer: {solution['solution'][:200]}
Issues: {', '.join(verification.get('issues_found', []))}

Try a completely different approach. If algebraic failed, try numerical.
Verify each candidate satisfies the original equation."""

        routing_retry           = routing.copy()
        routing_retry["strategy"] = "Alternative approach — " + routing.get("strategy", "")

        solution_retry     = solver_agent(parsed, routing_retry, memory_context=retry_context)
        verification_retry = verifier_agent(parsed, solution_retry)

        if verification_retry.get("is_correct") or \
           verification_retry.get("confidence", 0) > verification.get("confidence", 0):
            print("[4b] Retry succeeded — using corrected solution")
            solution                           = solution_retry
            verification                       = verification_retry
            verification["correction_attempted"] = True
        else:
            print("[4b] Retry also failed — escalating to human review")
            verification["needs_human_review"] = True
            verification["review_reason"]      = "Both attempts failed verification — human check required"

    print("[5/5] Explainer Agent running...")
    explanation = explainer_agent(parsed, solution, verification)

    return {
        "parsed_problem":   parsed,
        "routing_info":     routing,
        "solution_data":    solution,
        "verification":     verification,
        "explanation":      explanation,
        "memory_used":      verified_similar,
        "corrections_used": correction_similar
    }


if __name__ == "__main__":
    test_problem = "Find the roots of x^2 - 5x + 6 = 0"
    result       = run_pipeline(test_problem)
    print("\nFINAL EXPLANATION:")
    print(result["explanation"])