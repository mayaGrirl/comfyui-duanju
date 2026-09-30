#!/bin/bash
cd "$(dirname "$0")"
if [ $# -eq 0 ]; then
  .venv/bin/python 短剧.py --help
  echo
  echo "用法写在 使用.txt"
  read -n 1 -s -r -p "按任意键关闭"
  echo
else
  exec .venv/bin/python 短剧.py "$@"
fi
