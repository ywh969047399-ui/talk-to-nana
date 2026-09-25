"""前端 HTTP 服务 + 用 livekit API 包创建房间、分发 token、显式 dispatch agent。"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from dotenv import load_dotenv  # 阶段 20 修复：web 进程独立拉起时也读到 AGENT_NAME
from livekit import api as lk_api
from livekit.protocol.agent_dispatch import CreateAgentDispatchRequest
from livekit.protocol.room import CreateRoomRequest

from worker.runtime_env import (
    configure_egress_proxy,
    configure_local_no_proxy,
    local_service_env,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / "web"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def safe_print(*args, **kwargs) -> None:
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(arg) for arg in args)
        print(text.encode("utf-8", errors="replace").decode("utf-8"), **kwargs)


# 阶段 20 修复：web 是 nohup 后台拉，**不继承 shell env**，必须自己 load_dotenv
# 否则 AGENT_NAME 走默认值 "talk-to-me-agent"，跟 worker 的 "talk-to-me-dev3" 不匹配，
# dispatch 不会路由到这个 worker → 客户端进房没 agent。
for env_name in (".env.local", ".env"):
    env_file = PROJECT_ROOT / env_name
    if env_file.exists():
        load_dotenv(env_file)
        break

API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://127.0.0.1:7880")
AGENT_NAME = os.getenv("AGENT_NAME", "talk-to-me-dev3")
assert AGENT_NAME, "AGENT_NAME is required"

# 阶段 20 修复：先设 1087 代理再让 localhost 走 NO_PROXY 豁免（与 agent.py 对齐）
configure_egress_proxy()
configure_local_no_proxy()


def create_room_and_token(room_base: str, identity: str, name: str) -> dict:
    """创建房间、确保 agent dispatch 存在，并生成用户 token。

    阶段 29: 按 room 前缀路由到不同 worker 的 agent_name。
    room 命名约定：ttm-<provider>-room-xxxx
      ttm-minimax-room-*  → talk-to-me-minimax
      ttm-deepseek-room-* → talk-to-me-deepseek
      ttm-gemini-room-*   → talk-to-me-gemini
      其他/老 room 名     → 走 AGENT_NAME（兼容）
    """
    host = LIVEKIT_URL.replace("ws://", "http://").replace("wss://", "https://")
    room_name = f"{room_base}-{secrets.token_hex(4)}"

    # 阶段 29: room 前缀 → agent_name 路由
    _PROVIDER_TO_AGENT = {
        "minimax": "talk-to-me-minimax",
        "deepseek": "talk-to-me-deepseek",
        "gemini": "talk-to-me-gemini",
    }
    if "ttm-minimax" in room_base:
        target_agent = _PROVIDER_TO_AGENT["minimax"]
    elif "ttm-deepseek" in room_base:
        target_agent = _PROVIDER_TO_AGENT["deepseek"]
    elif "ttm-gemini" in room_base:
        target_agent = _PROVIDER_TO_AGENT["gemini"]
    else:
        target_agent = AGENT_NAME  # 兼容老 room 名
    safe_print(f"[web] 路由 room='{room_name}' -> agent='{target_agent}'", flush=True)

    async def ensure_room_and_dispatch() -> None:
        with local_service_env():
            lk = lk_api.LiveKitAPI(host, API_KEY, API_SECRET)
            try:
                await lk.room.create_room(CreateRoomRequest(name=room_name))
                safe_print(f"[web] OK 房间 '{room_name}' 已创建")
            except Exception as e:
                err_str = str(e)
                if "already" not in err_str.lower() and "409" not in err_str:
                    safe_print(f"[web] 创建房间异常（非致命）: {e}")
                else:
                    safe_print(f"[web] 房间 '{room_name}' 已存在，复用")

            try:
                await lk.agent_dispatch.create_dispatch(
                    CreateAgentDispatchRequest(agent_name=target_agent, room=room_name)
                )
                safe_print(f"[web] OK 已 dispatch agent: {target_agent} -> {room_name}")
            finally:
                await lk.aclose()

    asyncio.run(ensure_room_and_dispatch())

    user_token = (
        lk_api.AccessToken(API_KEY, API_SECRET)
        .with_identity(identity)
        .with_name(name)
        .with_grants(lk_api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )

    return {
        "token": user_token,
        "room": room_name,
        "identity": identity,
        "livekit_url": LIVEKIT_URL,
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        super().do_GET()

    def do_POST(self):
        if self.path == "/token":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid json"})
                return

            room = data.get("room", "talk-to-me-room")
            identity = data.get("identity", f"user-{secrets.token_hex(4)}")
            name = data.get("name", identity)

            result = create_room_and_token(room, identity, name)
            self._send_json(200, result)
        else:
            self._send_json(404, {"error": "not found"})

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def _send_json(self, status: int, data: dict):
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def log_message(self, format, *args):
        safe_print(f"[web] {args[0]}")


def main():
    port = int(os.getenv("WEB_PORT", "8766"))
    server = HTTPServer(("127.0.0.1", port), Handler)
    safe_print(f"[web] http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        safe_print("\n[web] 已停止")
        server.server_close()


if __name__ == "__main__":
    main()
