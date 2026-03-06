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
def solver_agent(parsed_problem: dict, routing_info: dict) -> dict:
    """
    Solves the problem using RAG context + Python calculator for exact arithmetic.
    """
    from calculator import execute_math_code

    query        = parsed_problem["problem_text"]
    context_docs = retrieve(query, top_k=3)
    context_text = "\n\n".join([
        f"[{d['title']}]\n{d['content']}" for d in context_docs
    ])

    # ── Step 1: Ask LLM to write Python code to solve it ─────────────────────
    code_prompt = f"""You are an expert JEE math solver.
Write Python code using sympy to solve this problem EXACTLY and symbolically.

AVAILABLE TOOLS (no imports needed, all ready to use):

SYMPY SYMBOLIC MATH (use these for calculus, algebra, equations):
- symbols('x')          — declare symbolic variable
- diff(expr, x)         — derivative of expression
- diff(expr, x, 2)      — second derivative
- solve(expr, x)        — solve equation = 0
- solve([eq1,eq2],[x,y])— solve system of equations
- integrate(expr, x)    — indefinite integral
- integrate(expr,(x,a,b))— definite integral
- limit(expr, x, val)   — compute limit
- simplify(expr)        — simplify expression
- factor(expr)          — factor expression
- ln(x)                 — natural log (sympy)
- Abs(x)                — absolute value (sympy)
- Rational(a,b)         — exact fraction a/b
- sqrt_sp(x)            — symbolic square root
- oo                    — infinity
- Matrix([[a,b],[c,d]]) — create matrix
- det([[a,b],[c,d]])    — determinant

NUMERIC MATH (use for combinations, probability):
- comb(n,r), perm(n,r), factorial(n)
- Fraction(a,b)         — exact fraction

CRITICAL RULES:
- Use sympy (sp, symbols, diff, solve) for ALL calculus and algebra problems
- Use comb/Fraction for ALL probability and combinations problems
- Always print final answer with print()
- For decimals from sympy: use float(answer) to convert
- For exact fractions from sympy: use sp.Rational or keep symbolic
- Never use numerical search loops — use solve() instead
- No import statements needed
- Return ONLY raw Python code, no markdown

REFERENCE MATERIAL:
{context_text}

CRITICAL DISTRIBUTION SELECTION RULES:
- Sampling WITHOUT replacement from finite population → Hypergeometric
  Var(X) = n*(K/N)*(1-K/N)*(N-n)/(N-1)
- Sampling WITH replacement → Binomial
  Var(X) = n*p*(1-p)

CRITICAL CALCULUS RULES:
- For strictly decreasing intervals with absolute value like |t+1|/t^2:
  Split into cases manually, do NOT rely on solve(f'<0) for abs functions
  Case 1: t < -1: substitute |t+1| = -(t+1), differentiate, find where f'<0
  Case 2: -1 < t < 0: substitute |t+1| = (t+1), differentiate, find where f'<0
  Combine to get largest interval, match with given interval notation to find alpha

- For interval (2*alpha, alpha): solve 2*alpha = left_end AND alpha = right_end
  Example: if decreasing on (-2,-1), then alpha=-1 since 2*(-1)=-2 ✓

- For local max/min: use diff(), solve(diff==0), substitute back
- ln(1) = 0 always — if critical point gives ln(1), answer is clean integer
- Never hardcode alpha — always derive it from the interval analysis
- After computing final answer with sympy, always clean it with:
  final = sp.re(sp.simplify(answer))
  print(int(final) if final == int(final) else final)

Problem: {query}
Topic: {routing_info['topic']} / {routing_info['subtopic']}
Strategy: {routing_info['strategy']}

Write sympy code to solve this exactly. Print the final answer clearly.
"""

    code = call_llm(
        "You write clean Python math code. Return raw code only, absolutely no markdown fences, no explanation.",
        code_prompt
    )

    # Clean any markdown fences if LLM adds them
    code = code.strip()
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].split("```")[0].strip()

    # ── Step 2: Execute the code ──────────────────────────────────────────────
    calc_result = execute_math_code(code)

    if calc_result["success"] and calc_result["output"]:
        calc_context = f"""
EXACT PYTHON CALCULATION RESULT (these numbers are 100% correct):
{calc_result['output']}

Python code that produced this:
{code}
"""
    else:
        # If code failed, try once more with error feedback
        retry_prompt = f"""Your previous code had an error: {calc_result['error']}
Fix and rewrite the Python code to solve:
{query}
Return raw code only, no markdown, no imports."""

        code = call_llm("Fix this Python math code. Return raw code only.", retry_prompt)
        code = code.strip()
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()

        calc_result = execute_math_code(code)
        if calc_result["success"] and calc_result["output"]:
            calc_context = f"""
EXACT PYTHON CALCULATION RESULT (these numbers are 100% correct):
{calc_result['output']}

Python code that produced this:
{code}
"""
        else:
            calc_context = "Calculator could not solve this. Solve carefully step by step showing all arithmetic."

    # ── Step 3: Ask LLM to explain using exact results ────────────────────────
    system = f"""You are an expert JEE math solver.
Use the reference material and exact calculation result below.

REFERENCE MATERIAL:
{context_text}

{calc_context}

Instructions:
- The EXACT PYTHON CALCULATION RESULT above is correct, use those numbers directly
- Show the method and reasoning clearly
- State the final answer using the exact computed value
- Do not redo arithmetic, trust the calculator output
"""

    solution = call_llm(
        system,
        f"Problem: {query}\n\nWrite the full solution using the exact results above."
    )

    return {
        "solution":       solution,
        "sources_used":   [d["title"] for d in context_docs],
        "context_docs":   context_docs,
        "generated_code": code,
        "calc_output":    calc_result.get("output", "")
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


# ── Master Pipeline ───────────────────────────────────────────────────────────
def run_pipeline(raw_input: str) -> dict:
    """
    Runs all 5 agents in sequence and returns the full result.
    """
    print("\n[1/5] Parser Agent running...")
    parsed       = parser_agent(raw_input)

    print("[2/5] Intent Router Agent running...")
    routing      = intent_router_agent(parsed)

    print("[3/5] Solver Agent running...")
    solution     = solver_agent(parsed, routing)

    print("[4/5] Verifier Agent running...")
    verification = verifier_agent(parsed, solution)

    print("[5/5] Explainer Agent running...")
    explanation  = explainer_agent(parsed, solution, verification)

    return {
        "parsed_problem": parsed,
        "routing_info":   routing,
        "solution_data":  solution,
        "verification":   verification,
        "explanation":    explanation
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