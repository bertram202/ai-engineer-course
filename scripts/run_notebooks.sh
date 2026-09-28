#!/usr/bin/env bash
# Выполнить все ноутбуки модуля: `scripts/run_notebooks.sh modules/M00-setup [--inplace]`.
#
# Без --inplace результаты выполнения отбрасываются (проверка, что ноутбуки работают).
# С --inplace выходы сохраняются в ноутбук (запись урока вместе с кассетами).
# Каждый ноутбук выполняется из своей папки — там же лежат его кассеты.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
module="$(cd "$1" && pwd)"
mode="${2:-}"
status=0
for nb in "$module"/lessons/*/lesson.ipynb; do
  dir="$(dirname "$nb")"
  # Тяжёлые уроки (гигабайты весов, долгое обучение) помечены в метаданных ноутбука
  # `mlcourse: {ci: skip}`: в CI (CI=true) они пропускаются, локально `make check` выполняет всё.
  if [[ "${CI:-}" == "true" ]] && python3 -c 'import json, sys
meta = json.load(open(sys.argv[1]))["metadata"].get("mlcourse", {})
sys.exit(0 if meta.get("ci") == "skip" else 1)' "$nb"; then
    echo "⏭ ${dir#"$root"/}: тяжёлый урок, в CI пропущен"
    continue
  fi
  echo "▶ ${dir#"$root"/}"
  if [[ "$mode" == "--inplace" ]]; then
    args=(--inplace)
  else
    args=(--stdout)
  fi
  if ! (cd "$dir" && "$root/scripts/jupyter.sh" "$module" nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=1800 "${args[@]}" "$nb" > /dev/null); then
    echo "✗ ошибка в ${dir#"$root"/}"
    status=1
  fi
done
exit $status
