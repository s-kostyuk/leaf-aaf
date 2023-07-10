#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=".:$PYTHONPATH"

SRGK_VENV_PYTHON="./venv/bin/python3"

if [[ -f "$SRGK_VENV_PYTHON" ]]; then
  SRGK_PYEXEC="$SRGK_VENV_PYTHON"
else
  SRGK_PYEXEC="python3"
fi


"$SRGK_PYEXEC" "$@"
