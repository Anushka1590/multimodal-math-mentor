import os
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

load_dotenv()

WOLFRAM_APP_ID = os.getenv("WOLFRAM_APP_ID")
WOLFRAM_URL    = "http://api.wolframalpha.com/v2/query"


def query_wolfram_raw(problem: str, timeout: int = 8) -> list:
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

        root = ET.fromstring(response.text)
        if root.attrib.get("success", "false") == "false":
            return []

        pods = []
        for pod in root.findall("pod"):
            texts = []
            for subpod in pod.findall("subpod"):
                pt = subpod.find("plaintext")
                if pt is not None and pt.text:
                    texts.append(pt.text.strip())
            if texts:
                pods.append({
                    "id":      pod.attrib.get("id", ""),
                    "title":   pod.attrib.get("title", ""),
                    "content": "\n".join(texts)
                })

        return pods

    except requests.Timeout:
        return []
    except Exception:
        return []


def get_best_answer(pods: list, query: str = "") -> str:
    if not pods:
        return ""

    query_lower = query.lower()

    if any(k in query_lower for k in ["derivative", "differentiate", "d/dx"]):
        for pod in pods:
            if pod["id"] == "Input" and "derivative" in pod["title"].lower():
                return pod["content"]

    if any(k in query_lower for k in ["integral", "integrate", "antiderivative", "∫"]):
        for pod in pods:
            if pod["id"] == "IndefiniteIntegral":
                return pod["content"]
        for pod in pods:
            if pod["id"] == "DefiniteIntegral":
                return pod["content"]
        for pod in pods:
            if "indefinite integral" in pod["title"].lower():
                return pod["content"]
        for pod in pods:
            if "integral" in pod["title"].lower():
                return pod["content"]

    for pid in ["Result", "Solution", "Limit", "Root", "Maximum", "Minimum", "Value", "Probability"]:
        for pod in pods:
            if pod["id"] == pid:
                return pod["content"]

    for title_key in ["result", "solution", "roots", "maximum", "minimum", "value"]:
        for pod in pods:
            if title_key in pod["title"].lower():
                return pod["content"]

    return pods[1]["content"] if len(pods) > 1 else pods[0]["content"]


def format_for_wolfram(problem: str) -> str:
    replacements = {
        "log_e(": "ln(",
        "log_e ": "ln ",
        "loge(":  "ln(",
        "d2y/dx2": "d^2y/dx^2",
    }
    result = problem
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result


def query_wolfram(problem: str) -> dict:
    formatted     = format_for_wolfram(problem)
    problem_lower = problem.lower()

    if any(k in problem_lower for k in ["∫", "integral", "integrate"]):
        formatted = formatted.replace("∫", "integrate ")
        formatted = formatted.replace("{", "").replace("}", "")
        formatted = formatted.replace("[", "").replace("]", "")

    pods = query_wolfram_raw(formatted)

    if not pods:
        return {
            "success":     False,
            "answer":      "",
            "steps":       "",
            "all_results": [],
            "error":       "No results or timeout"
        }

    return {
        "success":     True,
        "answer":      get_best_answer(pods, formatted),
        "steps":       next((p["content"] for p in pods if "step" in p["title"].lower()), ""),
        "all_results": pods,
        "error":       ""
    }


if __name__ == "__main__":
    tests = [
        ("Derivative",      "derivative of x^3 * ln(x)"),
        ("Quadratic roots", "roots of x^2 - 5x + 6 = 0"),
        ("Max value",       "maximum of -x^2 + 4x + 1"),
        ("Integral",        "integral of x^2 * ln(x)"),
        ("Limit",           "limit of sin(x)/x as x->0"),
    ]

    for name, query in tests:
        result = query_wolfram(query)
        print(f"\n{name}: {'✓' if result['success'] else '✗'}")
        print(f"Answer: {result['answer'][:200] if result['answer'] else 'NONE'}")