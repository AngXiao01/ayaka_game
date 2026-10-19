from pathlib import Path

from core.god_env import GodEnv, QuestDef
from core.world import World

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests/fixtures/yuebai"          # 引擎级测试用的故事，不随游戏发布
QUEST = QuestDef.load(ROOT / "tests/fixtures/yuebai/quests/ayaka_main.json")
WORLD = World.load(ROOT / "tests/fixtures/yuebai/world.json")


def make_god(state=None, trust=None):
    g = GodEnv(QUEST, WORLD.phases, WORLD.start, WORLD.turns_per_phase, state, clock=WORLD.clock())
    trust = trust or {}
    g.trust_lookup = lambda npc: trust.get(npc, 20)
    return g
