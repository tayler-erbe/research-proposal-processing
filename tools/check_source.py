"""Offline syntax and notebook checks; never imports production modules."""
import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
count = 0
for path in root.rglob("*.py"):
    if any(part in (".venv", "venv", "source-review", "__pycache__", ".git") for part in path.parts):
        continue
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    count += 1
notebook = json.loads((root / "notebooks" / "main.ipynb").read_text(encoding="utf-8"))
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        assert not cell.get("outputs"), "Notebook outputs must be stripped"
        assert cell.get("execution_count") is None
        ast.parse("".join(cell["source"]))
print(f"Parsed {count} Python files and production notebook cells without executing them.")
