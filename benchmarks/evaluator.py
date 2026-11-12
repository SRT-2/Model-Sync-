"""Blinded, reproducible response quality evaluator for LLM benchmark tasks."""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional, Tuple


def extract_code_blocks(text: str) -> List[str]:
    """Extract code snippets from markdown fences."""
    pattern = r"```(?:python|py)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return [m.strip() for m in matches]
    return []


def check_python_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """Safe static AST parse check without executing code."""
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)


def evaluate_response_quality(
    prompt_meta: Dict[str, Any],
    response_text: str,
) -> Dict[str, Any]:
    """Blinded evaluation of response quality against deterministic criteria and rubric.
    
    Returns:
      {
        "score": float (0-10),
        "passed": bool (score >= 7.0),
        "syntax_valid": Optional[bool],
        "elements_matched": List[str],
        "elements_missed": List[str],
        "notes": str
      }
    """
    if not response_text or not response_text.strip():
        return {
            "score": 0.0,
            "passed": False,
            "syntax_valid": False,
            "elements_matched": [],
            "elements_missed": prompt_meta.get("rubric", {}).get("required_elements", []),
            "notes": "Empty response received.",
        }

    rubric = prompt_meta.get("rubric", {})
    required = rubric.get("required_elements", [])
    max_score = float(rubric.get("max_score", 10.0))
    category = prompt_meta.get("category", "")
    eval_type = prompt_meta.get("evaluation_type", "rubric")

    text_lower = response_text.lower()
    matched = []
    missed = []

    for elem in required:
        elem_lower = elem.lower()
        if elem_lower in text_lower:
            matched.append(elem)
        else:
            missed.append(elem)

    # 1. Base score from required element coverage (up to 7.0 points)
    coverage_ratio = len(matched) / len(required) if required else 1.0
    element_points = coverage_ratio * 7.0

    # 2. Syntax / structural checks (up to 2.0 points)
    syntax_valid: Optional[bool] = None
    structural_points = 0.0
    notes_list = []

    if category in ("CODING", "UNIT_TESTING"):
        blocks = extract_code_blocks(response_text)
        if blocks:
            # Check syntax of largest block
            largest_block = max(blocks, key=len)
            valid, err = check_python_syntax(largest_block)
            syntax_valid = valid
            if valid:
                structural_points += 2.0
                notes_list.append("Code block AST syntax validated successfully.")
            else:
                structural_points += 0.5
                notes_list.append(f"Code block contains syntax error: {err}")
        else:
            # Check if inline code parses
            if "def " in response_text or "class " in response_text:
                structural_points += 1.0
                notes_list.append("Inline code provided without markdown fences.")
            else:
                notes_list.append("No code block found in response.")
    else:
        # For REASONING, check reasoning structure (step-by-step, clarity)
        has_steps = bool(re.search(r"(?:step\s*\d|case\s*\d|first|second|1\.|2\.)", text_lower))
        if has_steps:
            structural_points += 2.0
            notes_list.append("Well-structured step-by-step reasoning presented.")
        else:
            structural_points += 1.0

    # 3. Completeness & length heuristic (up to 1.0 point)
    length_points = 1.0 if len(response_text.strip().split()) >= 30 else 0.5

    total_score = round(min(max_score, element_points + structural_points + length_points), 2)
    passed = total_score >= 7.0

    return {
        "score": total_score,
        "passed": passed,
        "syntax_valid": syntax_valid,
        "elements_matched": matched,
        "elements_missed": missed,
        "notes": "; ".join(notes_list) if notes_list else "Evaluation completed.",
    }
