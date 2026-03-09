import os
import json
from groq import Groq
from dotenv import load_dotenv
from rag_pipeline import retrieve

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

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
        """
        Solves implicit differentiation exactly using sympy idiff.
        """
        import sympy as sp
        try:
            x, y = sp.symbols('x y')

            # Ask LLM to convert problem to sympy equation string only
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

            # Safe eval of equation
            namespace = {
                "sp": sp, "x": x, "y": y,
                "ln": sp.ln, "log": sp.ln,
                "sin": sp.sin, "cos": sp.cos,
                "tan": sp.tan, "exp": sp.exp,
                "sqrt": sp.sqrt
            }
            eq = eval(eq_str, namespace)

            # Find y at x=x_val
            y_val = sp.solve(eq.subs(x, x_val), y)
            if not y_val:
                return f"Could not find y at x={x_val}"
            y_val = y_val[0]

            # Compute derivatives using idiff
            dydx    = sp.idiff(eq, y, x)
            d2ydx2  = sp.idiff(eq, y, x, 2)

            dydx_at  = dydx.subs([(x, x_val), (y, y_val)])
            d2y_at   = d2ydx2.subs([(x, x_val), (y, y_val)])

            dydx_simplified  = sp.nsimplify(sp.re(dydx_at))
            d2y_simplified   = sp.nsimplify(sp.re(d2y_at))

            return (
                f"y at x={x_val}: {y_val}\n"
                f"dy/dx at x={x_val}: {dydx_simplified}\n"
                f"d2y/dx2 at x={x_val}: {d2y_simplified}"
            )

        except Exception as e:
            return f"Implicit diff error: {str(e)}"


# ── AGENT 1: Parser Agent ────────────────────────────────────────────────────
def parser_agent(raw_input: str) -> dict:
    """
    Converts raw OCR / ASR / typed input into a structured problem dict.
    """
    system = """You are a math problem parser for JEE-level questions.
Your job is to clean and structure raw input (which may come from OCR or speech).

Return ONLY a valid JSON object with these exact keys:
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
- Set needs_clarification to true if the problem is ambiguous or incomplete
- Return ONLY the JSON, no explanation, no markdown
"""
    result = call_llm(system, raw_input)

    # Clean markdown fences if LLM adds them
    result = result.strip()
    if result.startswith("```"):
        result = result.split("```")[1]
        if result.startswith("json"):
            result = result[4:]
    result = result.strip()

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return {
            "problem_text": raw_input,
            "topic": "general",
            "variables": [],
            "constraints": [],
            "needs_clarification": True,
            "clarification_reason": "Could not parse problem structure."
        }


# ── AGENT 2: Intent Router Agent ────────────────────────────────────────────
def intent_router_agent(parsed_problem: dict) -> dict:
    """
    Classifies the problem and decides the solution strategy.
    """
    system = """You are a math problem classifier for JEE-level questions.
Given a structured math problem, decide the solution strategy.

Return ONLY a valid JSON object with these exact keys:
{
  "topic": "algebra | probability | calculus | linear_algebra | general",
  "subtopic": "specific subtopic e.g. quadratic_equations, derivatives, matrices",
  "difficulty": "easy | medium | hard",
  "strategy": "brief description of approach to solve this problem",
  "tools_needed": ["calculator", "formula_lookup"]
}

Return ONLY the JSON, no explanation, no markdown.
"""
    problem_str = json.dumps(parsed_problem)
    result = call_llm(system, problem_str)

    result = result.strip()
    if result.startswith("```"):
        result = result.split("```")[1]
        if result.startswith("json"):
            result = result[4:]
    result = result.strip()

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return {
            "topic": parsed_problem.get("topic", "general"),
            "subtopic": "unknown",
            "difficulty": "medium",
            "strategy": "Solve step by step using relevant formulas.",
            "tools_needed": []
        }


# ── AGENT 3: Solver Agent ────────────────────────────────────────────────────
def solver_agent(parsed_problem: dict, routing_info: dict, memory_context: str = "") -> dict:
    """
    Hybrid solver: Wolfram Alpha for calculus/algebra,
    Python calculator for probability/combinations,
    Sympy as fallback.
    """
    from calculator import execute_math_code
    from wolfram_solver import query_wolfram, format_for_wolfram

    query        = parsed_problem["problem_text"]
    topic        = routing_info.get("topic", "general")
    subtopic     = routing_info.get("subtopic", "")
    context_docs = retrieve(query, top_k=3)
    context_text = "\n\n".join([
        f"[{d['title']}]\n{d['content']}" for d in context_docs
    ])

    wolfram_context = ""
    calc_context    = ""
    generated_code  = ""
    calc_output     = ""

    # ── Route to best computation tool ───────────────────────────────────────

    query_lower = query.lower()

    # Force sympy for implicit differentiation
    is_implicit = (
        any(k in query_lower for k in [
            "d2y", "d^2y", "dy/dx", "y''",
            "find d2y", "find dy/dx",
            "implicit differentiation"
        ]) or (
            any(k in query_lower for k in ["ln(x+y)", "log(x+y)", "loge(x+y)"]) and
            any(k in query_lower for k in ["dy", "d2y", "derivative of y", "at x="])
        )
    )

    # ── Handle implicit differentiation directly ──────────────────────────
    if is_implicit and "parabola" not in query_lower and "closest" not in query_lower and "distance" not in query_lower:
        import re
        # Extract x value from problem
        x_match = re.search(r'at x\s*=\s*(-?\d+\.?\d*)', query_lower)
        x_val   = float(x_match.group(1)) if x_match else 0.0

        implicit_result = solve_implicit_differentiation(
            query, query, x_val
        )

        context_docs = retrieve(query, top_k=3)
        context_text = "\n\n".join([
            f"[{d['title']}]\n{d['content']}" for d in context_docs
        ])

        system = f"""You are an expert JEE math solver.
Use the exact computation result below to write the solution.

COMPUTATION RESULT (100% correct):
{implicit_result}

REFERENCE MATERIAL:
{context_text}

Use the exact values from computation result. Show method clearly.
"""
        solution = call_llm(system, f"Problem: {query}\n\nExplain the solution.")

        return {
            "solution":       solution,
            "sources_used":   [d["title"] for d in context_docs],
            "context_docs":   context_docs,
            "generated_code": f"sp.idiff() used directly",
            "calc_output":    implicit_result
        }

    use_wolfram = not is_implicit and (
        any(k in topic.lower() for k in ["calculus", "algebra"]) or
        any(k in subtopic.lower() for k in [
            "derivative", "integral", "limit", "maximum", "minimum",
            "roots", "equation", "differentiation", "optimization",
            "decreasing", "increasing", "critical"
        ])
    )

    use_calculator = is_implicit or any(k in topic.lower() for k in [
        "probability", "combinatorics"
    ]) or any(k in subtopic.lower() for k in [
        "combination", "permutation", "probability",
        "distribution", "variance", "expectation",
        "hypergeometric", "binomial", "committee",
        "implicit", "second derivative"
    ])
    use_calculator = any(k in topic.lower() for k in [
        "probability", "combinatorics"
    ]) or any(k in subtopic.lower() for k in [
        "combination", "permutation", "probability",
        "distribution", "variance", "expectation",
        "hypergeometric", "binomial", "committee"
    ])

    # ── Tool 1: Wolfram Alpha ─────────────────────────────────────────────────
    if use_wolfram:
        wolfram_query  = format_for_wolfram(query)
        wolfram_result = query_wolfram(wolfram_query)

        if wolfram_result["success"] and wolfram_result["answer"]:
            all_results = wolfram_result["all_results"]
            context_str = "\n".join([
                f"[{r['title']}]: {r['content'][:200]}"
                for r in all_results[:5]
            ])
            wolfram_context = f"""
WOLFRAM ALPHA COMPUTATION RESULT (100% mathematically correct):
Answer: {wolfram_result['answer']}

All computation results:
{context_str}
"""
        else:
            # Fallback to sympy if Wolfram fails
            use_wolfram    = False
            use_calculator = True

    # ── Tool 2: Python Calculator + Sympy ────────────────────────────────────
    if use_calculator or not use_wolfram:
        code_prompt = f"""You are an expert JEE math solver.
Write Python code to solve this problem EXACTLY and print the final answer.

AVAILABLE TOOLS (no imports needed):

SYMPY (for calculus and algebra):
- symbols('x')           — declare variable
- diff(expr, x)          — derivative
- diff(expr, x, 2)       — second derivative
- solve(expr, x)         — solve equation
- integrate(expr, x)     — integral
- limit(expr, x, val)    — limit
- simplify(expr)         — simplify
- ln(x), Abs(x)          — log and absolute value
- Rational(a,b)          — exact fraction
- sp.re(expr)            — real part only
- sympy available as both 'sp' and 'sympy'

NUMERIC (for probability):
- comb(n,r), perm(n,r), factorial(n)
- Fraction(a,b)          — exact fraction

RULES:
- Use Fraction() for probability answers
- Use sympy for calculus/algebra
- Always print final answer clearly
- After sympy computation: clean with sp.re(sp.simplify(answer))
- No imports needed
- Raw code only, no markdown

CRITICAL DISTRIBUTION RULES:
- Sampling WITHOUT replacement → Hypergeometric
  Var(X) = n*(K/N)*(1-K/N)*(N-n)/(N-1)
- Sampling WITH replacement → Binomial
  Var(X) = n*p*(1-p)

CRITICAL CALCULUS RULES:
- For absolute value functions: split into cases manually
- For interval (2*alpha, alpha): solve both endpoints
- After computing: clean imaginary parts with sp.re(sp.simplify(result))

CRITICAL IMPLICIT DIFFERENTIATION RULES:
- ALWAYS use sp.idiff() for implicit differentiation
- NEVER include x_value or x_val as variable names in code
- NEVER copy template placeholders literally
- Substitute actual numbers directly, example:
  x, y = sp.symbols('x y')
  eq = sp.ln(x + y) - 4*x*y
  y_val = sp.solve(eq.subs(x, 0), y)[0]
  dydx = sp.idiff(eq, y, x)
  d2ydx2 = sp.idiff(eq, y, x, 2)
  ans = d2ydx2.subs([(x, 0), (y, y_val)])
  print(sp.nsimplify(sp.re(ans)))
- NEVER use sp.Function('y')
- ALWAYS use plain sp.symbols('x y')

REFERENCE MATERIAL:
{context_text}

Problem: {query}
Topic: {topic} / {subtopic}
Strategy: {routing_info.get('strategy', '')}

Write code to solve exactly. Print final answer.
"""
        generated_code = call_llm(
            "Write clean Python math code. Raw code only, no markdown.",
            code_prompt
        )

        # Clean markdown fences
        generated_code = generated_code.strip()
        if "```python" in generated_code:
            generated_code = generated_code.split("```python")[1].split("```")[0].strip()
        elif "```" in generated_code:
            generated_code = generated_code.split("```")[1].split("```")[0].strip()

        # Execute code
        calc_result = execute_math_code(generated_code)

        if calc_result["success"] and calc_result["output"]:
            calc_output  = calc_result["output"]
            calc_context = f"""
EXACT PYTHON CALCULATION RESULT:
{calc_output}
"""
        else:
            # Retry once with error feedback
            retry_prompt = f"""Previous code had error: {calc_result['error']}
Fix and rewrite Python code for: {query}
Raw code only, no markdown, no imports."""

            generated_code = call_llm(
                "Fix Python math code. Raw code only.", retry_prompt
            )
            generated_code = generated_code.strip()
            if "```python" in generated_code:
                generated_code = generated_code.split("```python")[1].split("```")[0].strip()
            elif "```" in generated_code:
                generated_code = generated_code.split("```")[1].split("```")[0].strip()

            calc_result = execute_math_code(generated_code)
            if calc_result["success"] and calc_result["output"]:
                calc_output  = calc_result["output"]
                calc_context = f"""
EXACT PYTHON CALCULATION RESULT:
{calc_output}
"""
            else:
                calc_context = "Calculator unavailable. Solve carefully step by step."

    # ── Final explanation by LLM ──────────────────────────────────────────────
    computation_result = wolfram_context if wolfram_context else calc_context

    memory_section = f"""
VERIFIED PAST SOLUTIONS FROM MEMORY (use these as reference for method):
{memory_context}
""" if memory_context else ""

    system = f"""You are an expert JEE math solver.
Use the reference material and computation result below to write the solution.

REFERENCE MATERIAL:
{context_text}

{memory_section}

{computation_result}

INSTRUCTIONS:
- The computation result above is 100% correct — use those exact numbers
- If verified past solutions are provided, follow the same method
- Show the method and reasoning clearly
- State the final answer using the exact computed value
- Do not redo arithmetic — trust the computation result
- Be concise but complete
"""

    solution = call_llm(
        system,
        f"Problem: {query}\n\nWrite the complete solution using the exact results above."
    )

    return {
        "solution":       solution,
        "sources_used":   [d["title"] for d in context_docs],
        "context_docs":   context_docs,
        "generated_code": generated_code,
        "calc_output":    calc_output if calc_output else (wolfram_result.get("answer", "") if 'wolfram_result' in dir() else "")
    }

# ── AGENT 4: Verifier Agent ──────────────────────────────────────────────────
def verifier_agent(parsed_problem: dict, solution_data: dict) -> dict:
    """
    Checks the solution for correctness, domain issues, edge cases.
    """
    system = """You are a strict math solution verifier for JEE problems.
Check the given solution for:
1. Mathematical correctness
2. Domain/constraint violations (e.g. sqrt of negative, log of zero)
3. Missing edge cases
4. Arithmetic errors

Return ONLY a valid JSON object:
{
  "is_correct": true,
  "confidence": 0.95,
  "issues_found": [],
  "corrected_answer": "",
  "needs_human_review": false,
  "review_reason": ""
}

Rules:
- confidence is a float between 0.0 and 1.0
- issues_found is a list of strings describing any problems
- corrected_answer is empty string if no correction needed
- needs_human_review is true if confidence < 0.8 or serious issues found
- Return ONLY the JSON, no explanation, no markdown
"""
    user_msg = f"""Problem: {parsed_problem['problem_text']}

Solution to verify:
{solution_data['solution']}
"""
    result = call_llm(system, user_msg)

    result = result.strip()
    if result.startswith("```"):
        result = result.split("```")[1]
        if result.startswith("json"):
            result = result[4:]
    result = result.strip()

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return {
            "is_correct":         True,
            "confidence":         0.7,
            "issues_found":       [],
            "corrected_answer":   "",
            "needs_human_review": True,
            "review_reason":      "Could not parse verifier response."
        }


# ── AGENT 5: Explainer Agent ─────────────────────────────────────────────────
def explainer_agent(parsed_problem: dict, solution_data: dict, verification: dict) -> str:
    """
    Produces a clear, student-friendly step-by-step explanation.
    """
    final_solution = (
        verification["corrected_answer"]
        if verification.get("corrected_answer")
        else solution_data["solution"]
    )

    system = """You are a friendly and encouraging JEE math tutor.
Your job is to explain the solution in a way that a Class 11-12 student can easily follow.

Format your explanation as:
## Understanding the Problem
(what the problem is asking)

## Key Concepts Used
(list the formulas/concepts needed)

## Step-by-Step Solution
(numbered steps, clear and simple)

## Final Answer
(clearly stated)

## Tips to Remember
(1-2 useful tips for similar problems)

Keep language simple, encouraging, and precise.
"""
    user_msg = f"""Problem: {parsed_problem['problem_text']}

Solution:
{final_solution}

Create a student-friendly explanation."""

    return call_llm(system, user_msg)

def solve_implicit_differentiation(problem_text: str, equation_str: str, x_val: float) -> str:
    """
    Solves implicit differentiation exactly using sympy idiff.
    """
    import sympy as sp
    try:
        x, y = sp.symbols('x y')

        # Ask LLM to convert problem to sympy equation string only
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

        # Safe eval of equation
        namespace = {
            "sp": sp, "x": x, "y": y,
            "ln": sp.ln, "log": sp.ln,
            "sin": sp.sin, "cos": sp.cos,
            "tan": sp.tan, "exp": sp.exp,
            "sqrt": sp.sqrt
        }
        eq = eval(eq_str, namespace)

        # Find y at x=x_val
        y_val = sp.solve(eq.subs(x, x_val), y)
        if not y_val:
            return f"Could not find y at x={x_val}"
        y_val = y_val[0]

        # Compute derivatives using idiff
        dydx    = sp.idiff(eq, y, x)
        d2ydx2  = sp.idiff(eq, y, x, 2)

        dydx_at  = dydx.subs([(x, x_val), (y, y_val)])
        d2y_at   = d2ydx2.subs([(x, x_val), (y, y_val)])

        dydx_simplified  = sp.nsimplify(sp.re(dydx_at))
        d2y_simplified   = sp.nsimplify(sp.re(d2y_at))

        return (
            f"y at x={x_val}: {y_val}\n"
            f"dy/dx at x={x_val}: {dydx_simplified}\n"
            f"d2y/dx2 at x={x_val}: {d2y_simplified}"
        )

    except Exception as e:
        return f"Implicit diff error: {str(e)}"


# ── Master Pipeline ───────────────────────────────────────────────────────────
def run_pipeline(raw_input: str) -> dict:
    """
    Runs all 5 agents in sequence and returns the full result.
    Incorporates memory of similar past problems.
    """
    from memory import retrieve_similar

    print("\n[1/5] Parser Agent running...")
    parsed  = parser_agent(raw_input)

    print("[2/5] Intent Router Agent running...")
    routing = intent_router_agent(parsed)

    # ── Check memory for similar verified problems ────────────────────────
    similar = retrieve_similar(raw_input, top_k=2)
    verified_similar = [
        s for s in similar
        if s.get("feedback") == "correct"
        and s.get("similarity_score", 999) < 0.5
    ]

    memory_context = ""
    if verified_similar:
        print(f"[Memory] Found {len(verified_similar)} similar verified problem(s) — using as reference")
        memory_context = "\n\n".join([
            f"VERIFIED PAST SOLUTION (marked correct by user):\n"
            f"Problem: {s['problem_text']}\n"
            f"Solution: {s['solution']}\n"
            f"Topic: {s['topic']}"
            for s in verified_similar
        ])

    print("[3/5] Solver Agent running...")
    solution = solver_agent(parsed, routing, memory_context=memory_context)

    print("[4/5] Verifier Agent running...")
    verification = verifier_agent(parsed, solution)

    print("[5/5] Explainer Agent running...")
    explanation = explainer_agent(parsed, solution, verification)

    return {
        "parsed_problem":  parsed,
        "routing_info":    routing,
        "solution_data":   solution,
        "verification":    verification,
        "explanation":     explanation,
        "memory_used":     verified_similar
    }


# ── Quick Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_problem = "The probability of forming a 12 persons committee from 4 engineers, 2 doctors and 10 professors containing at least 3 engineers and at least 1 doctor"
    result = run_pipeline(test_problem)

    print("\n" + "="*60)
    print("GENERATED CODE:")
    print(result["solution_data"].get("generated_code", "N/A"))

    print("\nCALCULATOR OUTPUT:")
    print(result["solution_data"].get("calc_output", "N/A"))

    print("\nFINAL EXPLANATION:")
    print(result["explanation"])