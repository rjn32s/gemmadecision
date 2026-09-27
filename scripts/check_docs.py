"""Compile Markdown Python examples without importing or running them."""
from pathlib import Path
import ast
import re


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    count = 0
    for path in sorted((root / "docs").rglob("*.md")):
        text = path.read_text()
        for match in re.finditer(r"^```python[^\n]*\n(.*?)^```\s*$", text, re.M | re.S):
            count += 1
            ast.parse(match.group(1), filename=str(path))
    print(f"Checked {count} Python examples without running inference.")


if __name__ == "__main__":
    main()
