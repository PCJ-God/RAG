#!/usr/bin/env python
"""
运行问答脚本
"""
import sys
from pathlib import Path

# 添加项目根目录到sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.app.cli import main

if __name__ == "__main__":
    main()
