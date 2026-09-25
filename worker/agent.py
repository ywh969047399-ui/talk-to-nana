"""Dev3 Agent — STT → LLM → TTS 分离流水线，音色克隆原生支持。"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    APIConnectionError,
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobExecutorType,
    cli,
    llm,
)
from livekit.agents.types import NOT_GIVEN, DEFAULT_API_CONNECT_OPTIONS  # 阶段 29.1
from livekit.agents import utils  # 阶段 29.1: shortuuid
# 关键：livekit-plugins-cartesia 的 @Plugin 装饰器要求主线程 import
from livekit.plugins import google  # noqa: F401  阶段 28: 保留 google.LLM 备用
from livekit.plugins import cartesia  # noqa: F401  阶段 28: STT 走 Cartesia ink-whisper

from worker.energy_vad import EnergyVAD
from worker.gemini_stt import GeminiSTT
from worker.llm_factory import DeepSeekChatStream, MiniMaxChatStream
from worker.memory_client import MemoryClient, build_memory_context
from worker.memory_recall import build_memory_block  # 阶段 28: 一次性 memory 快照
from worker.moss_tts import MossHttpTTS
from worker.persona import build_system_prompt
from worker.runtime_env import (
    configure_egress_proxy,
    configure_local_no_proxy,
    local_service_env,
)
from worker.tts_factory import build_tts  # 阶段 23：TTS 工厂


class _OpenAICompatLLM(llm.LLM):
    """阶段 29.1 修复: 真正实现 livekit LLM 协议。

    之前 _LivekitShim 把 chat() 写成 async generator，根本没走 livekit 事件流——
    agent_session 收不到 ChatChunk，调 Cartesia TTS 时传 0 token → 5s 后被取消。
    现在的实现：继承 llm.LLM，chat() 返回 LLMStream，_run() 把 OpenAI
    兼容 SSE 流的事件推入 _event_ch。
    """
    def __init__(self, chat_stream):
        super().__init__()
        self._chat_stream = chat_stream  # DeepSeekChatStream / MiniMaxChatStream 实例

    @property
    def model(self) -> str:
        return getattr(self._chat_stream, "model", "unknown")

    @property
    def provider(self) -> str:
        # 阶段 29.1: 必填否则 llm_stream trace 会 KeyError
        return "openai-compat"

    def chat(
        self,
        *,
        chat_ctx,
        tools=None,
        conn_options=None,
        parallel_tool_calls=NOT_GIVEN,
        tool_choice=NOT_GIVEN,
        extra_kwargs=NOT_GIVEN,
    ):
        # 阶段 29.1: 序列化 chat_ctx → OpenAI messages
        # chat_ctx.messages() 是方法（不是 property）——livekit API 微妙处
        messages = []
        chat_messages = chat_ctx.messages() if hasattr(chat_ctx, "messages") else []
        if callable(chat_messages):
            chat_messages = chat_messages()
        for item in chat_messages:
            role = getattr(item, "role", "user")
            content = getattr(item, "text_content", None) or getattr(item, "content", "")
            if callable(content):
                content = content()
            if not content:
                continue
            # 如果 content 是 list（多模态），只取 text part
            if isinstance(content, list):
                texts = []
                for part in content:
                    if isinstance(part, str):
                        texts.append(part)
                content = "\n".join(texts)
            messages.append({"role": str(role), "content": str(content)})
        # 阶段 29.1: conn_options 默认为 None 时给一个
        if conn_options is None:
            conn_options = DEFAULT_API_CONNECT_OPTIONS
        # 阶段 29.1: 必须把原始 chat_ctx 传给 LLMStream（trace 步骤会读）
        return _OpenAICompatLLMStream(self, messages=messages, conn_options=conn_options, chat_ctx=chat_ctx)


class _OpenAICompatLLMStream(llm.LLMStream):
    """阶段 29.1 修复: livekit LLMStream 子类，把 SSE 流推入 _event_ch。"""
    def __init__(self, llm_instance, *, messages, conn_options, chat_ctx):
        # 阶段 29.1: 把原始 chat_ctx 透传给父类（trace 会读）
        super().__init__(
            llm=llm_instance,
            chat_ctx=chat_ctx,
            tools=[],
            conn_options=conn_options,
        )
        self._messages = messages

    _MAX_CONTEXT_MESSAGES = 20

    async def _run(self) -> None:
        """阶段 29.1 修复: 真正流式推 ChatChunk。"""
        request_id = utils.shortuuid()
        # 上下文窗口截断：保留 system prompt + 最近 N 条消息，防止长对话撑爆 token limit
        messages = self._messages
        if len(messages) > self._MAX_CONTEXT_MESSAGES + 1:
            system_msgs = [m for m in messages if m.get("role") == "system"]
            non_system = [m for m in messages if m.get("role") != "system"]
            messages = system_msgs + non_system[-self._MAX_CONTEXT_MESSAGES:]
            print(f"[llm] context truncated: {len(self._messages)} -> {len(messages)} messages", flush=True)
        total_chars = sum(len(m.get("content", "")) for m in messages)
        print(f"[timing] llm_request msgs={len(messages)} chars={total_chars} t={time.time():.3f}", flush=True)
        try:
            async for piece in self._llm._chat_stream.chat(
                messages, temperature=0.7
            ):
                if not piece:
                    continue
                chat_chunk = llm.ChatChunk(
                    id=request_id,
                    delta=llm.ChoiceDelta(
                        role="assistant",
                        content=piece,
                    ),
                )
                # 阶段 29.1 修复: send_nowait 而不是 send（chan 没 close 的时候 send_nowait 是非阻塞的）
                self._event_ch.send_nowait(chat_chunk)
        except Exception as exc:
            print(f"[llm] openai-compat error: {exc!r} (messages={len(messages)})", flush=True)
            raise APIConnectionError(
                f"openai-compat LLM error: {exc!r}",
                retryable=False,
            ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent.parent
for env_name in (".env.local", ".env"):
    env_file = PROJECT_ROOT / env_name
    if env_file.exists():
        load_dotenv(env_file)
        break

# 阶段 20 修复（**修正版**）：
# 关键时序：
#   1. worker 进程启 → livekit-agents worker.py 读 LIVEKIT_URL=ws://127.0.0.1:7880 → **直连**注册到 livekit server
#      → 此时 1087 代理**必须不存在**（ws:// 不走 NO_PROXY 豁免，1087 SOCKS5 拒 ws upgrade）
#   2. worker registered 后 → 接到 dispatch → entrypoint(ctx.connect) → 进房间
#   3. **进房间后**才设 1087 → 让后续 google.LLM / GeminiSTT 调 Google API 走 1087
# 阶段 18 之前：worker 进程手动启 + bash unset → 没 1087 → registered 成功 → 之后调 Google API 卡死（无代理）
# 阶段 20 修法：**不要在 module-level 写 1087**；改在 entrypoint(ctx.connect) 之后设。
configure_local_no_proxy()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
STT_MODEL = os.getenv("STT_MODEL", LLM_MODEL)  # 阶段 25.3 修复：默认复用 LLM_MODEL
STT_PROVIDER = os.getenv("STT_PROVIDER", "gemini")
# 阶段 29: AGENT_NAME 默认按 LLM_PROVIDER 分开，三个 worker 同时跑
#   LLM_PROVIDER=minimax  → AGENT_NAME=talk-to-me-minimax
#   LLM_PROVIDER=deepseek  → AGENT_NAME=talk-to-me-deepseek
#   LLM_PROVIDER=gemini    → AGENT_NAME=talk-to-me-gemini
# 阶段 29 兼容老 env：没设 LLM_PROVIDER 时退化到 talk-to-me-dev3
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower() or "gemini"
_DEFAULT_AGENT_NAME_BY_LLM = {
    "minimax": "talk-to-me-minimax",
    "deepseek": "talk-to-me-deepseek",
    "gemini": "talk-to-me-gemini",
    "google": "talk-to-me-gemini",
}
# 阶段 29: 重要：先读 AGENT_NAME env；没设才按 LLM_PROVIDER 推
# 修正：AGENT_NAME 默认值由 LLM_PROVIDER 决定；只有当 env 显式给了不同 AGENT_NAME 才用 env
_env_agent_name = os.getenv("AGENT_NAME", "").strip()
if _env_agent_name and _env_agent_name != "talk-to-me-dev3":  # 显式设了非默认值
    AGENT_NAME = _env_agent_name
else:
    AGENT_NAME = _DEFAULT_AGENT_NAME_BY_LLM.get(LLM_PROVIDER, _env_agent_name or "talk-to-me-dev3")
os.environ["AGENT_NAME"] = AGENT_NAME  # 阶段 29: 把推出来的 AGENT_NAME 写回 env（web_server 也要读这个）
AGENT_PROMPT = os.getenv("AGENT_INSTRUCTIONS", "").strip()
MOSS_TTS_URL = os.getenv("MOSS_TTS_URL", "http://127.0.0.1:18083/v1/audio/speech").strip()
MOSS_VOICE_PROFILE = os.getenv("MOSS_VOICE_PROFILE", "ye-local").strip()
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "cartesia").strip().lower()  # 阶段 23：cartesia | minimax | moss
MEMORY_SESSION_PREFIX = os.getenv("MEMORY_SESSION_PREFIX", "dev3-pipeline")
_persona_for_memory = os.getenv("PERSONA_NAME", "fengge").strip().lower()
_project_root = Path(__file__).resolve().parent.parent
_default_data_dir = str(_project_root / "data" / "openviking" / "viking" / "default")
OPENVIKING_DATA_DIR = Path(os.getenv("OPENVIKING_DATA_DIR", _default_data_dir))

MEMORY_TOOL_INSTRUCTIONS = """
## 记忆召回规则
- 不要把启动提示当成长时记忆；不确定就说不确定。
- 当用户询问上次/最近聊了什么、某个人是谁、身份事实、偏好、关系、计划、承诺时，**直接依据下面「记忆快照」段落**回答；找不到就说不记得，不要瞎编。
""".strip()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _dedupe_memories(results: list[dict]) -> list[dict]:
    seen: set[str] = set()
    ordered: list[dict] = []
    for item in results:
        uri = str(item.get("uri") or "")
        key = uri or str(item.get("abstract") or item.get("content") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered


def _query_needles(query: str) -> set[str]:
    normalized = re.sub(r"\s+", "", query.strip().lower())
    needles: set[str] = set(re.findall(r"[a-z0-9_]{2,}", normalized))
    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
        needles.add(chunk)
        for size in (2, 3, 4):
            for i in range(len(chunk) - size + 1):
                needles.add(chunk[i : i + size])
    stopwords = {"什么", "什麼", "是谁", "是誰", "记得", "記得", "知道", "上次", "上回", "我们", "我們", "一下"}
    return {n for n in needles if n not in stopwords}


def _lexical_local_memories(query: str, max_files: int = 8) -> list[dict]:
    memories_dir = OPENVIKING_DATA_DIR / "user" / "default" / "memories"
    if not memories_dir.exists():
        return []
    needles = _query_needles(query)
    if not needles:
        return []
    scored: list[tuple[int, float, Path, str]] = []
    for path in memories_dir.rglob("*.md"):
        if path.name.startswith(".") or path.name == ".overview.md":
            continue
        content = _read_text(path)
        if not content:
            continue
        compact = re.sub(r"\s+", "", content.lower())
        score = sum(1 for n in needles if n in compact)
        if score <= 0:
            continue
        scored.append((score, path.stat().st_mtime, path, content))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    results: list[dict] = []
    for score, _mtime, path, content in scored[:max_files]:
        rel = path.relative_to(OPENVIKING_DATA_DIR)
        results.append({
            "uri": f"local://lexical-memory/{rel}",
            "abstract": content,
            "context_type": "lexical_memory_file",
            "score": float(score),
        })
    return results


async def _collect_turn_memories(text: str) -> tuple[str, list[dict]]:
    results: list[dict] = []
    memory = MemoryClient()
    try:
        if await memory.health():
            query_results = await memory.search(text, top_k=8)
            for item in query_results:
                uri = str(item.get("uri") or "")
                if uri.startswith("viking://agent/"):
                    continue
                results.append(item)
    finally:
        await memory.close()
    results.extend(_lexical_local_memories(text, max_files=8))
    return "search", _dedupe_memories(results)


class OpenVikingRecorder:
    def __init__(self, session_id: str, memory: MemoryClient) -> None:
        self.session_id = session_id
        self.memory = memory
        self._lock = asyncio.Lock()
        self._message_count = 0
        self._seen_user_texts: list[str] = []

    @staticmethod
    def _fingerprint(text: str) -> str:
        return re.sub(r"\s+", "", text.strip().lower())

    def _is_duplicate(self, text: str) -> bool:
        fp = self._fingerprint(text)
        if not fp:
            return True
        return fp in self._seen_user_texts[-20:]

    def _remember(self, text: str) -> None:
        fp = self._fingerprint(text)
        if fp and fp not in self._seen_user_texts:
            self._seen_user_texts.append(fp)

    async def add_message(self, role: str, content: str, *, commit: bool = False) -> None:
        text = content.strip()
        if not text or text == "<noise>":
            return
        async with self._lock:
            if role == "user":
                if self._is_duplicate(text):
                    print("[memory] skip duplicate user text", flush=True)
                    return
                self._remember(text)
            await self.memory.add_messages(self.session_id, [{"role": role, "content": text}])
            self._message_count += 1
            print(f"[memory] add_message role={role} count={self._message_count}", flush=True)
            if commit:
                result = await self.memory.commit(self.session_id)
                print(f"[memory] commit ok: {result}", flush=True)

    async def finalize(self) -> None:
        async with self._lock:
            if self._message_count > 0:
                result = await self.memory.commit(self.session_id)
                print(f"[memory] final commit ok: {result}", flush=True)
            await self.memory.close()
            print("[memory] recorder closed", flush=True)


class Dev3Agent(Agent):
    def __init__(self, instructions: str) -> None:
        # 阶段 26.2: STT 改 Deepgram nova-2 (28s 实测首段 2.6s vs gemini 2.5-flash 6s)
        stt_instance = self._build_stt()
        # 阶段 29: LLM 按 LLM_PROVIDER env 选 gemini / deepseek / minimax
        llm_instance = self._build_llm()
        # 阶段 23：TTS 按 TTS_PROVIDER 选实现（cartesia / minimax / moss）
        tts_instance, tts_label = build_tts(TTS_PROVIDER, MOSS_TTS_URL, MOSS_VOICE_PROFILE)
        vad_instance = EnergyVAD(
            speech_threshold=int(os.getenv("VAD_SPEECH_THRESHOLD", "750")),
            silence_threshold=int(os.getenv("VAD_SILENCE_THRESHOLD", "300")),
            min_speech_duration=float(os.getenv("VAD_MIN_SPEECH_DURATION", "0.25")),
            min_silence_duration=float(os.getenv("VAD_MIN_SILENCE_DURATION", "0.30")),
        )
        print(f"[agent] Dev3Agent components: STT={stt_instance.model} LLM={self._llm_label()} TTS={tts_label} VAD={vad_instance.model}", flush=True)

        super().__init__(
            instructions=instructions,
            stt=stt_instance,
            llm=llm_instance,
            tts=tts_instance,
            vad=vad_instance,
        )

    def _build_stt(self):
        """阶段 28: STT_PROVIDER=cartesia 走 ink-whisper; deepgram 保留; gemini 兜底。"""
        prov = STT_PROVIDER.lower()
        if prov == "cartesia":
            from worker.cartesia_stt import build_cartesia_stt
            return build_cartesia_stt()
        if prov == "deepgram":
            from worker.deepgram_stt import DeepgramSTT
            return DeepgramSTT(
                api_key=os.getenv("DEEPGRAM_API_KEY", ""),
                model=os.getenv("DEEPGRAM_MODEL", "nova-2"),
                language=os.getenv("DEEPGRAM_LANGUAGE", "zh"),
                sample_rate=int(os.getenv("DEEPGRAM_SAMPLE_RATE", "22050")),
            )
        return GeminiSTT(api_key=GOOGLE_API_KEY, model=STT_MODEL)

    def _build_llm(self):
        """阶段 29.1 修复: 改用真正继承 llm.LLM 的 _OpenAICompatLLM。

        gemini   → google.LLM（livekit 包装）
        deepseek → _OpenAICompatLLM(DeepSeekChatStream)
        minimax  → _OpenAICompatLLM(MiniMaxChatStream)
        """
        prov = LLM_PROVIDER
        if prov == "deepseek":
            return _OpenAICompatLLM(
                DeepSeekChatStream(
                    api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                    model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                )
            )
        if prov == "minimax":
            return _OpenAICompatLLM(
                MiniMaxChatStream(
                    api_key=os.getenv("MINIMAX_API_KEY", ""),
                    model=os.getenv("MINIMAX_MODEL_NAME", "MiniMax-M2.7-highspeed"),
                    max_tokens=int(os.getenv("MINIMAX_MAX_TOKENS", "150")),
                )
            )
        # gemini / google 默认
        return google.LLM(model=LLM_MODEL, api_key=GOOGLE_API_KEY, temperature=0.7)

    def _llm_label(self) -> str:
        """给日志用的 LLM 标识。"""
        if LLM_PROVIDER == "deepseek":
            return os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        if LLM_PROVIDER == "minimax":
            return os.getenv("MINIMAX_MODEL_NAME", "MiniMax-M2.7-highspeed")
        return LLM_MODEL

    # 阶段 28: 不再用 function_tool 走 recall_memory（auto-recall 直接拼 system prompt，
    # 避免 tool call 拖 1-2s 延迟）。OpenViking recorder 仍在跑，做沉淀。


server = AgentServer(
    job_executor_type=JobExecutorType.THREAD,
    initialize_process_timeout=60.0,
    # 阶段 29: 三个 worker 并行跑；用 port 区分（默认 8081 会冲突）
    #   8081 → minimax, 8082 → deepseek, 8083 → gemini
    port=int(os.getenv("LIVEKIT_WORKER_PORT", "8081")),
)


_PUSH_AUDIO_COUNT = 0
_PUSH_AUDIO_FIRST = None
_PUSH_AUDIO_LAST = None


_ROOM_IO_TRACK_COUNT = 0
_ROOM_IO_SET_PARTICIPANT_COUNT = 0


def _install_push_audio_probe() -> None:
    """monkey patch AgentActivity.push_audio 加探针，确认 audio frame 是否真到 VAD/STT 链路。"""
    from livekit.agents.voice import agent_activity
    from livekit.agents.voice import room_io as _ri_mod
    global _PUSH_AUDIO_COUNT, _PUSH_AUDIO_FIRST, _PUSH_AUDIO_LAST
    global _ROOM_IO_TRACK_COUNT, _ROOM_IO_SET_PARTICIPANT_COUNT

    if not getattr(agent_activity.AgentActivity.push_audio, "_probed", False):
        orig = agent_activity.AgentActivity.push_audio

        def probed(self, frame):
            global _PUSH_AUDIO_COUNT, _PUSH_AUDIO_FIRST, _PUSH_AUDIO_LAST
            _PUSH_AUDIO_COUNT += 1
            if _PUSH_AUDIO_COUNT == 1 or _PUSH_AUDIO_COUNT % 50 == 1:
                _PUSH_AUDIO_FIRST = _PUSH_AUDIO_FIRST or time.time()
                print(
                    f"[agent] push_audio #{_PUSH_AUDIO_COUNT} sr={frame.sample_rate} "
                    f"ch={frame.num_channels} samples={frame.samples_per_channel}",
                    flush=True,
                )
            _PUSH_AUDIO_LAST = time.time()
            return orig(self, frame)

        probed._probed = True
        agent_activity.AgentActivity.push_audio = probed

    # RoomIO set_participant 探针
    if not getattr(_ri_mod.RoomIO.set_participant, "_probed", False):
        orig_sp = _ri_mod.RoomIO.set_participant

        def probed_sp(self, participant_identity):
            global _ROOM_IO_SET_PARTICIPANT_COUNT
            _ROOM_IO_SET_PARTICIPANT_COUNT += 1
            print(
                f"[agent] room_io.set_participant #{_ROOM_IO_SET_PARTICIPANT_COUNT} identity={participant_identity}",
                flush=True,
            )
            return orig_sp(self, participant_identity)

        probed_sp._probed = True
        _ri_mod.RoomIO.set_participant = probed_sp

    # _MossChunkedStream._forward_task 探针：监听 _forward_task 怎么走
    from livekit.agents.voice.room_io import _input as _input_mod
    if not getattr(_input_mod._ParticipantAudioInputStream._forward_task, "_probed", False):
        orig_ft = _input_mod._ParticipantAudioInputStream._forward_task

        async def probed_ft(self, old_task, stream, publication, participant):
            print(
                f"[agent] _forward_task START participant={participant.identity} "
                f"track_sid={publication.track.sid if publication.track else None} "
                f"audio_features={publication.audio_features}",
                flush=True,
            )
            try:
                await orig_ft(self, old_task, stream, publication, participant)
                print(
                    f"[agent] _forward_task END participant={participant.identity}",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[agent] _forward_task ERROR participant={participant.identity}: {e!r}",
                    flush=True,
                )
                raise

        probed_ft._probed = True
        _input_mod._ParticipantAudioInputStream._forward_task = probed_ft

        # _ParticipantAudioInputStream._on_track_available 探针
    from livekit.agents.voice.room_io import _input as _input_mod
    if not getattr(_input_mod._ParticipantInputStream._on_track_available, "_probed", False):
        orig_ota = _input_mod._ParticipantInputStream._on_track_available

        def probed_ota(self, track, publication, participant):
            global _ROOM_IO_TRACK_COUNT
            _ROOM_IO_TRACK_COUNT += 1
            print(
                f"[agent] audio_input._on_track_available #{_ROOM_IO_TRACK_COUNT} "
                f"participant.identity={participant.identity!r} "
                f"publication.source={publication.source} track.sid={track.sid}",
                flush=True,
            )
            try:
                res = orig_ota(self, track, publication, participant)
                print(
                    f"[agent] _on_track_available #{_ROOM_IO_TRACK_COUNT} returned={res}",
                    flush=True,
                )
                return res
            except Exception as e:
                import traceback as _tb
                tb = _tb.format_exc()
                print(
                    f"[agent] _on_track_available #{_ROOM_IO_TRACK_COUNT} RAISED: {e!r}\n{tb}",
                    flush=True,
                )
                raise

        probed_ota._probed = True
        _input_mod._ParticipantInputStream._on_track_available = probed_ota

    # 只 patch audio_input._data_ch.send（不是 aio.Chan.send 全局）
    # 用 set_participant 钩子：等 audio_input 设了 _data_ch 后再 patch
    if not getattr(_input_mod._ParticipantInputStream.set_participant, "_probed", False):
        orig_sp_ia = _input_mod._ParticipantInputStream.set_participant

        def probed_sp_ia(self, participant_identity):
            res = orig_sp_ia(self, participant_identity)
            if self._data_ch is not None and not getattr(self._data_ch.send, "_audio_probed", False):
                orig_data_send = self._data_ch.send

                async def probed_data_send(item):
                    samples = getattr(item, "samples_per_channel", "?")
                    sr = getattr(item, "sample_rate", "?")
                    n = len(item.data) if hasattr(item, "data") and item.data is not None else 0
                    print(
                        f"[agent] _data_ch.send type={type(item).__name__} samples={samples} sr={sr} bytes={n}",
                        flush=True,
                    )
                    return await orig_data_send(item)

                probed_data_send._audio_probed = True
                self._data_ch.send = probed_data_send
            return res

        probed_sp_ia._probed = True
        _input_mod._ParticipantInputStream.set_participant = probed_sp_ia


def _install_wide_accepted_sources_patch() -> None:
    """路径 B 变体修复（**修正版**）：
    根因：livekit-agents 1.5.17 的 _ParticipantAudioInputStream.__init__
        写死 track_source=rtc.TrackSource.SOURCE_MICROPHONE（_input.py:270），
        父类 _ParticipantInputStream.__init__ 拿到后转成单元素 set {_accepted_sources = {1}}。
    后果：Python SDK 的 AudioSource + capture_frame() 发布的 track，publication.source
        是 SOURCE_UNKNOWN (0)，被 `0 not in {1}` 拦掉，_on_track_available return False，
        _create_stream / _forward_task 都不起 → VAD / STT 永远 0 帧。
    修法：把 _ParticipantAudioInputStream.__init__ 包一层，强制 track_source=
        [SOURCE_MICROPHONE, SOURCE_UNKNOWN]，这样父类 _accepted_sources = {1, 0}，UNKNOWN 也能过。
    注意：
      - 只 patch 我们的 _ParticipantAudioInputStream 子类，**不动父类**（避免影响视频流）。
      - 这是 livekit-agents 1.5.17 行为，升级版本后需重新评估。
    """
    from livekit.agents.voice.room_io import _input as _input_mod
    import livekit.rtc as rtc

    if getattr(_input_mod._ParticipantAudioInputStream.__init__, "_wide_sources_patched", False):
        return

    orig_init = _input_mod._ParticipantAudioInputStream.__init__

    def wide_init(self, room, *, sample_rate, num_channels, noise_cancellation,
                  auto_gain_control=True, pre_connect_audio_handler=None, frame_size_ms=50):
        # 直接调父类 _ParticipantInputStream.__init__，绕过原本写死的 SOURCE_MICROPHONE。
        # 其他参数（sample_rate / num_channels / noise_cancellation / frame_size_ms）保持
        # 与原版一致——避免影响 APM / 音频流参数。
        from livekit.agents.voice.room_io._input import _ParticipantInputStream
        from livekit.agents.voice.io import AudioInput
        _ParticipantInputStream.__init__(
            self,
            room=room,
            track_source=[
                rtc.TrackSource.SOURCE_MICROPHONE,
                rtc.TrackSource.SOURCE_UNKNOWN,
            ],
            processor=(
                noise_cancellation if isinstance(noise_cancellation, rtc.FrameProcessor) else None
            ),
        )
        AudioInput.__init__(self, label="RoomIO")
        if frame_size_ms <= 0:
            raise ValueError("frame_size_ms must be greater than 0")
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._frame_size_ms = frame_size_ms
        self._noise_cancellation = noise_cancellation
        self._pre_connect_audio_handler = pre_connect_audio_handler
        self._apm: rtc.AudioProcessingModule | None = None
        if auto_gain_control:
            self._apm = rtc.AudioProcessingModule(auto_gain_control=True)

    wide_init._wide_sources_patched = True
    _input_mod._ParticipantAudioInputStream.__init__ = wide_init
    print(
        "[agent] _ParticipantAudioInputStream.__init__ patched: "
        "track_source widened to {SOURCE_MICROPHONE, SOURCE_UNKNOWN}",
        flush=True,
    )


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    # ctx.connect 走 local_service_env（块内清代理 env）→ 127.0.0.1 直连 livekit
    with local_service_env():
        await ctx.connect()

    # 阶段 20：进房间后**立即**设 1087 代理 → 后续 google.LLM / GeminiSTT 调 Google API 走 1087
    configure_egress_proxy()

    # 阶段 20 探针：装 push_audio 探针，统计 frame 是否真流到 VAD/STT
    _install_push_audio_probe()

    # 路径 B 变体：放宽 _ParticipantAudioInputStream 接受的 track source，让
    # publication.source=SOURCE_UNKNOWN（python SDK 客户端默认）也能进 _forward_task
    _install_wide_accepted_sources_patch()

    recorder: OpenVikingRecorder | None = None
    startup_memory = MemoryClient()
    try:
        if await startup_memory.health():
            label = f"{MEMORY_SESSION_PREFIX}:{ctx.room.name}:{int(time.time())}"
            session_id = await startup_memory.create_session(label=label)
            recorder = OpenVikingRecorder(session_id, startup_memory)
            print(f"[memory] session created: {session_id}", flush=True)
        else:
            await startup_memory.close()
            print("[memory] OpenViking unavailable at startup", flush=True)
    except Exception as exc:
        print(f"[memory] startup error: {exc}", flush=True)
        await startup_memory.close()
        recorder = None

    persona_name = os.getenv("PERSONA_NAME", "yehuiyu").strip().lower()
    instructions = build_system_prompt(persona_name)
    if AGENT_PROMPT:
        instructions = f"{instructions}\n\n## 运行时补充要求\n{AGENT_PROMPT}"

    t0 = time.time()
    memory_block = build_memory_block(max_chars=800)
    print(f"[memory] recall block built chars={len(memory_block)} elapsed={time.time() - t0:.3f}s", flush=True)
    if memory_block:
        memory_owner = {
            "fengge": "峰哥",
            "nana": "娜娜",
            "yehuiyu": "叶会羽",
        }.get(persona_name, persona_name)
        instructions = (
            f"{instructions}\n\n## 关于{memory_owner}的记忆快照\n"
            f"以下是启动时从本地记忆系统拉取的快照。"
            f"回答涉及「我/上次/那个人/那个偏好」时直接用这里的事实；"
            f"找不到再说不记得。\n\n{memory_block}"
        )
    else:
        print("[memory] no memory block available, agent starts without auto-recall", flush=True)

    instructions = f"{instructions}\n\n{MEMORY_TOOL_INSTRUCTIONS}"
    print(f"[agent] runtime prompt: {len(instructions)} chars", flush=True)

    # V3.5.1: preemptive_generation 需要 LLM 支持完整的取消/重生成协议，
    # 我们的 _OpenAICompatLLM 包装层不支持，开启后 agent 卡在 listening 不说话。
    # 要启用需要先改造 _OpenAICompatLLM 实现 LiveKit 的 cancel() 接口。
    session = AgentSession(
        turn_handling={
            "interruption": {"enabled": True},
            "preemptive_generation": {"enabled": False},
        }
    )

    _timing: dict[str, float] = {}

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev) -> None:
        now = time.time()
        old, new = str(ev.old_state), str(ev.new_state)
        parts = [f"[timing] {old}->{new}"]
        if "listening" in old and "thinking" in new:
            _timing["think_start"] = now
            if "vad_end" in _timing:
                parts.append(f"vad_end→think={int((now - _timing['vad_end'])*1000)}ms")
        elif "thinking" in old and "speaking" in new:
            _timing["speak_start"] = now
            if "think_start" in _timing:
                parts.append(f"think→speak={int((now - _timing['think_start'])*1000)}ms")
            if "vad_end" in _timing:
                parts.append(f"vad_end→speak={int((now - _timing['vad_end'])*1000)}ms")
        print(" ".join(parts), flush=True)

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev) -> None:
        now = time.time()
        if ev.is_final:
            _timing["stt_final"] = now
        print(f"[timing] stt_final={ev.is_final} t={now:.3f}: {ev.transcript}", flush=True)
        if recorder is not None and ev.is_final and isinstance(ev.transcript, str):
            asyncio.create_task(recorder.add_message("user", ev.transcript))

    @session.on("conversation_item_added")
    def _on_conversation_item_added(ev) -> None:
        item = ev.item
        role = getattr(item, "role", type(item).__name__)
        text = getattr(item, "text_content", None)
        if callable(text):
            text = text()
        if not text:
            text = getattr(item, "content", "")
        print(f"[agent] conversation item role={role}: {text}", flush=True)
        if recorder is not None and role in {"user", "assistant"} and isinstance(text, str):
            asyncio.create_task(recorder.add_message(role, text, commit=(role == "assistant")))

    @session.on("speech_created")
    def _on_speech_created(ev) -> None:
        print(f"[agent] speech created source={ev.source}", flush=True)

    @session.on("function_tools_executed")
    def _on_function_tools_executed(ev) -> None:
        names = [getattr(call, "name", type(call).__name__) for call in getattr(ev, "function_calls", [])]
        print(f"[agent] function tools executed: {names}", flush=True)

    @session.on("error")
    def _on_error(ev) -> None:
        print(f"[agent] error source={ev.source}: {ev.error}", flush=True)

    @session.on("close")
    def _on_close(ev) -> None:
        print(f"[agent] close reason={ev.reason}", flush=True)
        if recorder is not None:
            asyncio.create_task(recorder.finalize())

    await session.start(agent=Dev3Agent(instructions), room=ctx.room)


def main() -> None:
    cli.run_app(server)


if __name__ == "__main__":
    main()
