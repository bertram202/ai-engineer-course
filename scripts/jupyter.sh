#!/usr/bin/env bash
# Запуск jupyter в окружении модуля: `scripts/jupyter.sh modules/M00-setup lab`.
#
# JUPYTER_PATH ставит ядро python3 из окружения модуля впереди пользовательских
# kernelspec (~/Library/Jupyter/kernels), которые могут указывать на чужой Python.
set -euo pipefail
module="$1"
shift
cd "$module"
uv sync --quiet
prefix="$(uv run --no-sync python -c 'import sys; print(sys.prefix)')"
export JUPYTER_PATH="$prefix/share/jupyter${JUPYTER_PATH:+:$JUPYTER_PATH}"
exec uv run --no-sync jupyter "$@"
