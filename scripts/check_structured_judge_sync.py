"""Compare the vendored structured-judge constants in prime-rl against the
source-of-truth in activation_oracles_dev. Exits non-zero if they drift.

Drift candidates: JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE, JUDGE_TOOL. If any
of these differ, manually copy the new text from source into the vendored file.
"""

from __future__ import annotations

import re
import sys

SOURCE = "/workspace-vast/adamk/activation_oracles_dev/investigations/model_understanding_eval/eval_model_understanding.py"
VENDORED = "/workspace-vast/adamk/prime-rl/environments/model_understanding_rl_structured/model_understanding_rl_structured/judge.py"


def extract_triple_quoted(text: str, marker_pat: str) -> str | None:
    m = re.search(marker_pat, text, re.M)
    if not m:
        return None
    start = m.end()
    end = text.find('"""', start)
    if end == -1:
        return None
    return text[start:end]


def extract_dict_literal(text: str, name: str) -> str | None:
    # Match `NAME: dict[str, Any] = {` ... up to first line-starting `}`
    pat = rf"^{re.escape(name)}\s*:.*?^\}}"
    m = re.search(pat, text, re.M | re.S)
    return m.group(0) if m else None


def main() -> int:
    source_text = open(SOURCE).read()
    vendored_text = open(VENDORED).read()

    drift: list[str] = []

    # Style-agnostic — handles both `"""\\<newline>body` and `"""body` openings.
    triple_quoted_checks = {
        "JUDGE_SYSTEM_PROMPT": r'^JUDGE_SYSTEM_PROMPT\s*=\s*"""',
        "JUDGE_USER_TEMPLATE": r'^JUDGE_USER_TEMPLATE\s*=\s*"""',
    }
    for name, pat in triple_quoted_checks.items():
        src = extract_triple_quoted(source_text, pat)
        ven = extract_triple_quoted(vendored_text, pat)
        if src is None or ven is None:
            print(f"  {name}: missing (source={src is not None}, vendored={ven is not None})")
            drift.append(name)
            continue
        if src != ven:
            print(f"  {name}: DRIFT (source {len(src)} chars, vendored {len(ven)} chars)")
            drift.append(name)
        else:
            print(f"  {name}: OK ({len(src)} chars)")

    src_tool = extract_dict_literal(source_text, "JUDGE_TOOL")
    ven_tool = extract_dict_literal(vendored_text, "JUDGE_TOOL")
    if src_tool is None or ven_tool is None:
        print(f"  JUDGE_TOOL: missing (source={src_tool is not None}, vendored={ven_tool is not None})")
        drift.append("JUDGE_TOOL")
    elif src_tool == ven_tool:
        print(f"  JUDGE_TOOL: OK ({len(src_tool)} chars)")
    else:
        print(f"  JUDGE_TOOL: DRIFT (source {len(src_tool)} chars, vendored {len(ven_tool)} chars)")
        drift.append("JUDGE_TOOL")

    if drift:
        print(f"\nDRIFT detected in: {drift}")
        print(f"Re-vendor by editing {VENDORED} to match the source.")
        return 2
    print("\nAll synced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
