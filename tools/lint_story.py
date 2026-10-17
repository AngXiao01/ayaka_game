"""故事内容的静态校验（见 core/story_lint.py）：线索拿得到吗、信号达成得了吗、作证的人那个时段真的在场吗。

  python3 tools/lint_story.py            有 error 时退出码为 1，只有 warn 时为 0
  python3 tools/lint_story.py --strict   warn 也算失败
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load(content_dir: Path | None = None, quest_id: str = "ayaka_main"):
    from core.god_env import QuestDef
    from core.world import World
    from models.personality import Personality
    c = content_dir or ROOT / "content"
    return QuestDef.load(c / "quests" / f"{quest_id}.json"), World.load(c / "world.json"), Personality.load_all(c / "npcs")


def main(argv=None) -> int:
    from core.story_lint import lint
    ap = argparse.ArgumentParser(description="故事内容的静态校验")
    ap.add_argument("--strict", action="store_true", help="warn 也算失败")
    ap.add_argument("--content", help="内容目录，默认 content/")
    args = ap.parse_args(argv)
    findings = lint(*load(Path(args.content) if args.content else None))
    for f in findings:
        print(f)
    errors = sum(f.level == "error" for f in findings)
    warns = len(findings) - errors
    print(f"\n{errors} 个错误，{warns} 个需要作者决定的疑点" if findings else "没有发现问题")
    return 1 if errors or (args.strict and warns) else 0


if __name__ == "__main__":
    sys.exit(main())
