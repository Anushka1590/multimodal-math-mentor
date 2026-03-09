import os
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

load_dotenv()

WOLFRAM_APP_ID = os.getenv("WOLFRAM_APP_ID")
WOLFRAM_URL    = "http://api.wolframalpha.com/v2/query"


def query_wolfram_raw(problem: str, timeout: int = 20) -> list:
    """
    Returns all pods from Wolfram Alpha as a list of dicts.
    """
    try:
        params = {
            "input":  problem,
            "appid":  WOLFRAM_APP_ID,
            "format": "plaintext",
            "output": "XML"
        }
        response = requests.get(WOLFRAM_URL, params=params, timeout=timeout)

        if response.status_code != 200:
            return []

        root         = ET.fromstring(response.text)
        success_attr = root.attrib.get("success", "false")

        if success_attr == "false":
            return []

        pods = []
        for pod in root.findall("pod"):
            title    = pod.attrib.get("title", "")
            pod_id   = pod.attrib.get("id", "")
            texts    = []

            for subpod in pod.findall("subpod"):
                pt = subpod.find("plaintext")
                if pt is not None and pt.text:
                    texts.append(pt.text.strip())

            if texts:
                pods.append({
                    "id":      pod_id,
                    "title":   title,
                    "content": "\n".join(texts)
                })

        return pods

    except requests.Timeout:
        return []
    except Exception:
        return []


def get_best_answer(pods: list, query: str = "") -> str:
    """
    Extracts the most relevant answer from pods.
    """
    if not pods:
        return ""

    query_lower = query.lower()

    # For derivative queries — grab Input pod which has the result
    if any(k in query_lower for k in ["derivative", "differentiate", "d/dx"]):
        for pod in pods:
            if pod["id"] == "Input" and "derivative" in pod["title"].lower():
                return pod["content"]

    # For integral queries
    if any(k in query_lower for k in ["integral", "integrate", "antiderivative"]):
        for pod in pods:
            if pod["id"] == "IndefiniteIntegral":
                return pod["content"]
        for pod in pods:
            if pod["id"] == "DefiniteIntegral":
                return pod["content"]

    # Priority 1: Exact pod ID matches
    priority_ids = [
        "Result", "Solution", "Limit",
        "Root", "Maximum", "Minimum",
        "Value", "Probability"
    ]
    for pid in priority_ids:
        for pod in pods:
            if pod["id"] == pid:
                return pod["content"]

    # Priority 2: Title matching
    priority_titles = [
        "result", "solution", "roots",
        "maximum", "minimum", "value"
    ]
    for title_key in priority_titles:
        for pod in pods:
            if title_key in pod["title"].lower():
                return pod["content"]

    # Fallback: second pod
    if len(pods) > 1:
        return pods[1]["content"]

    return pods[0]["content"] if pods else ""


def query_wolfram(problem: str) -> dict:
    """
    Main function — queries Wolfram Alpha and returns clean result.
    """
    # Format problem for better Wolfram parsing
    formatted = format_for_wolfram(problem)
    pods      = query_wolfram_raw(formatted)

    if not pods:
        return {
            "success":     False,
            "answer":      "",
            "steps":       "",
            "all_results": [],
            "error":       "No results or timeout"
        }

    answer = get_best_answer(pods, formatted)
    steps  = next(
        (p["content"] for p in pods if "step" in p["title"].lower()),
        ""
    )

    return {
        "success":     True,
        "answer":      answer,
        "steps":       steps,
        "all_results": pods,
        "error":       ""
    }


def format_for_wolfram(problem: str) -> str:
    """
    Converts common math notation to Wolfram-friendly format.
    """
    replacements = {
        "log_e(":    "ln(",
        "log_e ":    "ln ",
        "loge(":     "ln(",
        "d2y/dx2":   "d^2y/dx^2",
        "alpha":     "alpha",
        "beta":      "beta",
        "theta":     "theta",
        "<=":        "<=",
        ">=":        ">=",
    }
    result = problem
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result

    def debug_pods(problem: str):
        """Shows ALL pods returned by Wolfram Alpha for a query."""
        params = {
            "input":  problem,
            "appid":  WOLFRAM_APP_ID,
            "format": "plaintext",
            "output": "XML"
        }
        response = requests.get(WOLFRAM_URL, params=params, timeout=20)
        root     = ET.fromstring(response.text)

        print(f"\nAll pods for: '{problem}'")
        print("=" * 60)
        for pod in root.findall("pod"):
            title  = pod.attrib.get("title", "")
            pod_id = pod.attrib.get("id", "")
            texts  = []
            for subpod in pod.findall("subpod"):
                pt = subpod.find("plaintext")
                if pt is not None and pt.text:
                    texts.append(pt.text.strip())
            if texts:
                print(f"\n[ID: {pod_id}] Title: {title}")
                print("\n".join(texts[:3]))
        print("=" * 60)


if __name__ == "__main__":
    tests = [
        ("Derivative",      "derivative of x^3 * ln(x)"),
        ("Quadratic roots", "roots of x^2 - 5x + 6 = 0"),
        ("Max value",       "maximum of -x^2 + 4x + 1"),
        ("Integral",        "integral of x^2 * ln(x)"),
        ("Limit",           "limit of sin(x)/x as x->0"),
    ]

    for name, query in tests:
        print(f"\nTest: {name}")
        result = query_wolfram(query)
        print(f"Success: {result['success']}")
        print(f"Answer:  {result['answer'][:200] if result['answer'] else 'NONE'}")
        print("-" * 50)