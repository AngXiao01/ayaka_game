"""运行配置：环境变量 / .env → Settings。

优先级：进程环境变量 > 项目根目录 .env > 默认值。

后端（LLM_BACKEND）：
  openrouter（默认）  OpenRouter，变量前缀 OPENROUTER_；key 读 OPENROUTER_API_KEY，
                     否则读 OPENROUTER_API_KEY_FILE（默认 ~/.openrouter_key）；
                     OPENROUTER_MODEL 默认 cheapest = 启动时自动挑能用的最便宜模型（见 llm/model_picker.py）
  gemma              自建 vLLM，变量前缀 GEMMA_；key 读 GEMMA_API_KEY，否则读 GEMMA_API_KEY_FILE
                     （默认 ~/.agentrt/api_key，与 agentrt/scripts/up51.sh 共用）
  mock               离线
每个后端只读自己前缀的变量：切到 openrouter 时 .env 里留着的 GEMMA_* 不生效，
自建服务的 key 也不会被发给第三方。
"""
from __future__ import annotations

import json
import re
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if value[:1] in ('"', "'") and value[0] in value[1:]:
            value = value[1:value.index(value[0], 1)]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()   # 去掉行尾注释
        os.environ.setdefault(key.strip(), value)


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None or v.strip() == "" else v.strip()


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


@dataclass(frozen=True)
class Provider:
    prefix: str                 # 环境变量前缀
    base_url: str
    model: str
    key_file: str               # 没设 <PREFIX>_API_KEY 时读的文件
    headers: tuple = ()         # 额外请求头
    reasoning_effort: str = ""  # 默认的推理档位（只有 OpenRouter 认）；空 = 不发


PROVIDERS = {
    "openrouter": Provider("OPENROUTER", "https://openrouter.ai/api/v1", "cheapest", "~/.openrouter_key",
                           (("X-Title", "ayaka_game"),), "none"),
    "gemma": Provider("GEMMA", "http://127.0.0.1:8000/v1", "heretic", "~/.agentrt/api_key"),
}
BACKENDS = (*PROVIDERS, "mock")
DEFAULT_BACKEND = "openrouter"


def provider_of(backend: str) -> Provider:
    return PROVIDERS.get(backend, PROVIDERS["gemma"])      # mock 不连网，沿用 gemma 的默认值即可


_OR_KEY_RE = re.compile(r"sk-or-v\d+-[A-Za-z0-9]+")


def clean_key(raw: str, prefix: str = "") -> tuple[str, str]:
    """把粘贴进来的 key 规整一下：去空白、引号、「Bearer 」前缀；OpenRouter 的 key 混在别的文字里时把它抠出来。
    返回 (key, 说明)；说明为空表示原样可用。"""
    k = raw.strip().strip("\"'").strip()
    note = ""
    if k.lower().startswith("bearer "):
        k, note = k[7:].strip(), "去掉了 Bearer 前缀"
    if prefix == "OPENROUTER" and not _OR_KEY_RE.fullmatch(k):
        joined = re.sub(r"\s+", "", k)
        found = sorted(set(_OR_KEY_RE.findall(raw)))
        if _OR_KEY_RE.fullmatch(joined):
            k, note = joined, "去掉了中间的换行 / 空格"
        elif len(found) == 1:
            k, note = found[0], "从文件里的其他文字中提取出 sk-or- key"
    return k, note


def read_api_key(prefix: str = "GEMMA", default_file: str = "~/.agentrt/api_key") -> tuple[str, str]:
    """返回 (key, 来源描述)。来源只写变量名或文件路径，不含 key 本身。"""
    raw = os.environ.get(f"{prefix}_API_KEY", "")
    if raw.strip():
        key, note = clean_key(raw, prefix)
        return key, f"{prefix}_API_KEY" + (f"（{note}）" if note else "")
    path = _env(f"{prefix}_API_KEY_FILE", default_file)
    key_file = Path(os.path.expanduser(path))
    if key_file.is_file():
        key, note = clean_key(key_file.read_text(encoding="utf-8", errors="replace"), prefix)
        return key, f"{prefix}_API_KEY_FILE={path}" + (f"（{note}）" if note else "")
    return "", f"没有找到（设 {prefix}_API_KEY，或把 key 写进 {path}）"


@dataclass
class TaskParams:
    temperature: float
    max_tokens: int


TASKS = ("npc", "options", "summary", "review", "compress", "director", "branch", "scene")


@dataclass
class LLMSettings:
    backend: str = DEFAULT_BACKEND              # openrouter | gemma | mock
    base_url: str = PROVIDERS[DEFAULT_BACKEND].base_url
    model: str = PROVIDERS[DEFAULT_BACKEND].model   # OpenRouter 的模型 id；gemma 时是 vLLM 的 --served-model-name
    api_key: str = ""
    key_source: str = ""                        # key 从哪来的（变量名 / 文件路径），给 --check 和报错用
    headers: dict = field(default_factory=dict)
    reasoning_effort: str = ""                  # OpenRouter 的 reasoning.effort；默认 none（关掉推理，省钱也不吃 max_tokens）
    cheapest_vendors: tuple = ()                # model=cheapest 时只在这些厂商里挑；空 = 用 model_picker 的默认名单
    model_note: str = ""                        # 自动选模型时的说明（main 打印）
    free_first: bool = False                    # 免费优先：先试免费模型，失败这一次回落付费（FREE_FIRST=1）
    free_models: tuple = ("nvidia/nemotron-3-super-120b-a12b:free", "google/gemma-4-31b-it:free", "google/gemma-4-26b-a4b-it:free")
    timeout: float = 120.0
    max_retries: int = 3
    merge_system: bool = False                  # 模板不认 system 角色时打开
    structured: str = "json_schema"             # json_schema | json_object | off（不支持时自动降级）
    json_repair_attempts: int = 2               # JSON 不合法时回填错误让模型重写的次数
    strip_thinking: bool = True
    chat_template_kwargs: Optional[dict] = None
    max_context_tokens: int = 32768             # 游戏自己的上下文窗口；gemma 时与 vLLM --max-model-len 对齐
    npc: TaskParams = field(default_factory=lambda: TaskParams(0.8, 700))
    options: TaskParams = field(default_factory=lambda: TaskParams(0.9, 700))
    summary: TaskParams = field(default_factory=lambda: TaskParams(0.3, 700))
    review: TaskParams = field(default_factory=lambda: TaskParams(0.1, 500))
    compress: TaskParams = field(default_factory=lambda: TaskParams(0.2, 400))
    director: TaskParams = field(default_factory=lambda: TaskParams(0.3, 400))
    branch: TaskParams = field(default_factory=lambda: TaskParams(0.0, 200))
    scene: TaskParams = field(default_factory=lambda: TaskParams(0.7, 300))
    # 模型路由：某个任务想走别的模型就设 MODEL_<TASK>（同一个后端下的另一个模型，如 MODEL_REVIEW=openai/gpt-4o-mini）
    routes: dict[str, str] = field(default_factory=dict)


@dataclass
class HarnessSettings:
    """Loop / Context / 终止判断相关的工程参数。"""
    loop_max_steps: int = 4                     # NPC 单轮 ReAct 最多几步（含最终 respond）
    max_repeat: int = 1                         # 同一工具同参数允许重复几次，超过就收紧为只能 respond
    turn_token_budget: int = 40000              # 单轮 NPC 循环的 token 上限（prompt+completion）
    turn_timeout_s: float = 90.0                # 单轮 NPC 循环的墙钟上限
    obs_max_chars: int = 1000                   # 工具观察结果截断长度
    game_token_budget: int = 1500000            # 整局 token 预算，0 = 不限（一整局约 30~60 万，多数命中前缀缓存）
    degrade_ratio: float = 0.15                 # 剩余预算低于该比例时降级（选项静态化、审核只走规则、NPC 不用工具）
    reviewer_mode: str = "llm"                  # llm | rules | off
    review_fail_policy: str = "reject"          # 审核调用失败时：reject（fail-closed）| accept
    stall_limit: int = 10                       # 连续多少轮既无进展、关系也没升温，就收束这次交谈
    player_input_max_chars: int = 1000
    compress_mode: str = "llm"                  # 工作记忆超窗时：llm 摘要 | extractive 抽取 | off 直接丢
    # GOD / tree / ENV 三层的模型部分：都只提议，最后由代码校验执行
    director_mode: str = "llm"                  # 导演（时段切换时调节奏）：llm | rules | off
    director_max_actions: int = 2               # 导演每次最多几个动作
    branch_mode: str = "llm"                    # 分支（自由输入的处置方案 → 预设方案）：llm | rules | off
    scene_mode: str = "llm"                     # 场景（自由描述要查看的东西）：llm | rules | off
    persona_mode: str = "index"                 # 人物卡：index 只放核心 + 索引、细节按话题注入 | full 全量进 system
    # 分支判定：auto = OpenRouter 有 key 用 Jev（便宜），否则用大模型 | jev | llm | off（只用关键词）
    decider_mode: str = "auto"
    decider_model: str = "typesafe/jev-1.13"
    decider_url: str = "https://openrouter.ai/api/alpha/decisions"
    decider_timeout_s: float = 8.0


@dataclass
class GameSettings:
    npc_id: str = "ayaka"
    quest_id: str = "ayaka_main"
    player_name: str = "旅行者"
    max_turns_per_session: int = 20
    history_turns: int = 8                      # 工作记忆窗口：保留原文的最近轮数，更早的压成摘要
    memory_top_k: int = 3                       # 自动注入的记忆条数，更多靠 recall_memory 按需检索
    options_mode: str = "llm"                   # llm | static
    hints: bool = True                          # 菜单里用 💡 标出可能有进展的人、物、地点
    continue_after_ending: bool = True          # 结局 / 截止时间之后继续玩（后日谈）；自动玩家和评测照旧在结局处停
    debug: bool = False
    seed: Optional[int] = None
    data_dir: Path = ROOT / "data"
    content_dir: Path = ROOT / "content"


@dataclass
class Settings:
    llm: LLMSettings
    game: GameSettings
    harness: HarnessSettings = field(default_factory=HarnessSettings)


def _params(prefix: str, default: TaskParams) -> TaskParams:
    return TaskParams(_env_float(f"{prefix}_TEMPERATURE", default.temperature),
                      _env_int(f"{prefix}_MAX_TOKENS", default.max_tokens))


def load_settings(env_file: Optional[Path] = None) -> Settings:
    load_dotenv(env_file or ROOT / ".env")
    backend = _env("LLM_BACKEND", DEFAULT_BACKEND).lower()
    if backend not in BACKENDS:
        raise ValueError(f"LLM_BACKEND={backend!r}，只能是 {' | '.join(BACKENDS)}")
    pv = provider_of(backend)
    P = pv.prefix
    ctk = _env(f"{P}_CHAT_TEMPLATE_KWARGS", "")
    d = LLMSettings()
    key, key_source = read_api_key(P, pv.key_file)
    llm = LLMSettings(
        backend=backend,
        base_url=_env(f"{P}_BASE_URL", pv.base_url),
        model=_env(f"{P}_MODEL", pv.model),
        api_key=key,
        key_source=key_source,
        headers=dict(pv.headers),
        reasoning_effort=_env(f"{P}_REASONING_EFFORT", pv.reasoning_effort) if backend == "openrouter" else "",
        cheapest_vendors=tuple(v.strip() for v in _env(f"{P}_CHEAPEST_VENDORS", "").split(",") if v.strip()),
        free_first=_env_bool("FREE_FIRST", False) and backend == "openrouter",
        free_models=tuple(v.strip() for v in _env("FREE_MODELS", ",".join(LLMSettings.free_models)).split(",") if v.strip()),
        timeout=_env_float(f"{P}_TIMEOUT", d.timeout),
        max_retries=_env_int(f"{P}_MAX_RETRIES", d.max_retries),
        merge_system=_env_bool(f"{P}_MERGE_SYSTEM", False),
        structured=_env(f"{P}_STRUCTURED", d.structured),
        json_repair_attempts=_env_int(f"{P}_JSON_REPAIR_ATTEMPTS", d.json_repair_attempts),
        strip_thinking=_env_bool(f"{P}_STRIP_THINKING", True),
        chat_template_kwargs=json.loads(ctk) if ctk else None,
        max_context_tokens=_env_int(f"{P}_MAX_CONTEXT_TOKENS", d.max_context_tokens),
        npc=_params("NPC", d.npc),
        options=_params("OPTION", d.options),
        summary=_params("SUMMARY", d.summary),
        review=_params("REVIEW", d.review),
        compress=_params("COMPRESS", d.compress),
        director=_params("DIRECTOR", d.director),
        branch=_params("BRANCH", d.branch),
        scene=_params("SCENE", d.scene),
        routes={t: _env(f"MODEL_{t.upper()}", "") for t in TASKS if _env(f"MODEL_{t.upper()}", "")},
    )
    h = HarnessSettings()
    harness = HarnessSettings(
        loop_max_steps=_env_int("LOOP_MAX_STEPS", h.loop_max_steps),
        max_repeat=_env_int("LOOP_MAX_REPEAT", h.max_repeat),
        turn_token_budget=_env_int("TURN_TOKEN_BUDGET", h.turn_token_budget),
        turn_timeout_s=_env_float("TURN_TIMEOUT_S", h.turn_timeout_s),
        obs_max_chars=_env_int("OBS_MAX_CHARS", h.obs_max_chars),
        game_token_budget=_env_int("GAME_TOKEN_BUDGET", h.game_token_budget),
        degrade_ratio=_env_float("DEGRADE_RATIO", h.degrade_ratio),
        reviewer_mode=_env("REVIEWER_MODE", h.reviewer_mode),
        review_fail_policy=_env("REVIEW_FAIL_POLICY", h.review_fail_policy),
        stall_limit=_env_int("STALL_LIMIT", h.stall_limit),
        player_input_max_chars=_env_int("PLAYER_INPUT_MAX_CHARS", h.player_input_max_chars),
        compress_mode=_env("COMPRESS_MODE", h.compress_mode),
        director_mode=_env("DIRECTOR_MODE", h.director_mode),
        director_max_actions=_env_int("DIRECTOR_MAX_ACTIONS", h.director_max_actions),
        branch_mode=_env("BRANCH_MODE", h.branch_mode),
        scene_mode=_env("SCENE_MODE", h.scene_mode),
        persona_mode=_env("PERSONA_MODE", h.persona_mode),
        decider_mode=_env("DECIDER_MODE", h.decider_mode).lower(),
        decider_model=_env("DECIDER_MODEL", h.decider_model),
        decider_url=_env("DECIDER_URL", h.decider_url),
        decider_timeout_s=_env_float("DECIDER_TIMEOUT_S", h.decider_timeout_s),
    )
    seed = _env("GAME_SEED", "")
    game = GameSettings(
        npc_id=_env("GAME_NPC", "ayaka"),
        quest_id=_env("GAME_QUEST", "ayaka_main"),
        player_name=_env("PLAYER_NAME", "旅行者"),
        max_turns_per_session=_env_int("MAX_TURNS_PER_SESSION", 20),
        history_turns=_env_int("HISTORY_TURNS", 8),
        memory_top_k=_env_int("MEMORY_TOP_K", 3),
        options_mode=_env("OPTIONS_MODE", "llm"),
        hints=_env_bool("GAME_HINTS", True),
        continue_after_ending=_env_bool("CONTINUE_AFTER_ENDING", True),
        debug=_env_bool("GAME_DEBUG", False),
        seed=int(seed) if seed else None,
    )
    if _env("GAME_DATA_DIR", ""):
        game.data_dir = Path(_env("GAME_DATA_DIR", ""))
    return Settings(llm=llm, game=game, harness=harness)
