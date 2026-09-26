#!/usr/bin/env python3
"""Заготовки упражнений из эталонных решений.

    python3 scripts/make_exercises.py modules/M01-math        # создать или обновить exercises/ex*.py
    python3 scripts/make_exercises.py --check modules/*        # проверить, что заготовки не разошлись

Заготовка — это решение, в котором тело каждой функции верхнего уровня (кроме докстринга)
заменено на `raise NotImplementedError("TODO")`. Всё остальное — импорты, константы, классы,
комментарии — остаётся как в решении. Так заготовка и решение отличаются только телами функций.
Функции, помеченные комментарием `# готово` в строке с `def`, переносятся без изменений
(вспомогательный код, который студенту писать не нужно). Строка `# подсказка: …` в теле функции
решения превращается в текст исключения: `raise NotImplementedError("TODO: …")`.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

HINT = "# подсказка:"


def make_stub(source: str) -> str:
    lines = source.splitlines()
    tree = ast.parse(source)
    replacements: list[tuple[int, int, str]] = []  # (начало тела после докстринга, конец, подсказка), строки с 1
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if "# готово" in lines[node.lineno - 1]:
            continue
        first = node.body[0]
        has_doc = isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant)
        has_doc = has_doc and isinstance(first.value.value, str)
        start = (first.end_lineno + 1) if has_doc else first.lineno
        body_lines = (ln.strip() for ln in lines[start - 1 : node.end_lineno])
        hints = [ln[len(HINT) :].strip() for ln in body_lines if ln.startswith(HINT)]
        replacements.append((start, node.end_lineno, hints[0] if hints else ""))
    for start, end, hint in reversed(replacements):
        message = f"TODO: {hint}" if hint else "TODO"
        lines[start - 1 : end] = [f"    raise NotImplementedError({json.dumps(message, ensure_ascii=False)})"]
    return "\n".join(lines) + "\n"


def exercise_path(solution: Path) -> Path:
    return solution.parent.parent / "exercises" / solution.name


def main(argv: list[str]) -> int:
    check = "--check" in argv
    modules = [Path(a) for a in argv if a != "--check"]
    if not modules:
        print(__doc__)
        return 2
    bad = 0
    for module in modules:
        for solution in sorted((module / "solutions").glob("ex*.py")):
            target = exercise_path(solution)
            expected = make_stub(solution.read_text())
            if check:
                if not target.exists() or target.read_text() != expected:
                    print(f"✗ {target} не совпадает с заготовкой из {solution.name}: python3 {__file__} {module}")
                    bad += 1
            elif not target.exists() or target.read_text() != expected:
                target.write_text(expected)
                print(f"✓ {target}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
