from pathlib import Path

from core.god_env import GodEnv, QuestDef
from core.world import World

ROOT = Path(__file__).resolve().parent.parent
QUEST = QuestDef.load(ROOT / "content/quests/ayaka_main.json")
WORLD = World.load(ROOT / "content/world.json")


def make_god(state=None, trust=None):
    g = GodEnv(QUEST, WORLD.phases, WORLD.start, WORLD.turns_per_phase, state)
    trust = trust or {}
    g.trust_lookup = lambda npc: trust.get(npc, 20)
    return g
