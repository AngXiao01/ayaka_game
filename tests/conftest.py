import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 引擎级测试默认用 tests/fixtures/yuebai（线索多、分支全），不依赖随游戏发布的故事内容。
# 要在导入时（收集阶段）就生效，崩溃矩阵的基线是模块加载时跑的；要测发布的 content/，测试里显式写 content_dir。
os.environ["CONTENT_DIR"] = str(ROOT / "tests" / "fixtures" / "yuebai")
