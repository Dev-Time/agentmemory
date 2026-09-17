"""
Telemetry plugin for agentmemory.
Captures tool calls, prompts, and agent lifecycles.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError

DEFAULT_BASE_URL = "http://localhost:3111"
TIMEOUT = 5
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
_plaintext_bearer_warned = False

def _preload_agentmemory_dotenv() -> None:
    candidates: list[Path] = []
    home = os.environ.get("HOME")
    if home:
        candidates.append(Path(home) / ".agentmemory" / ".env")
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config:
        candidates.append(Path(xdg_config) / "agentmemory" / ".env")
    for path in candidates:
        try:
            if not path.is_file():
                continue
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    os.environ.setdefault(key, value)
        except (OSError, UnicodeDecodeError):
            continue

_preload_agentmemory_dotenv()

def _validate_url(base: str) -> bool:
    if not base:
        return False
    try:
        parsed = urlparse(base)
        _ = parsed.port
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    return bool(parsed.hostname)

def _uses_plaintext_bearer_auth(base: str, secret: str = "") -> bool:
    if not secret:
        return False
    parsed = urlparse(base)
    return parsed.scheme == "http" and (parsed.hostname or "").lower() not in LOOPBACK_HOSTS

def _plaintext_bearer_auth_message(base: str) -> str:
    return f"agentmemory: AGENTMEMORY_SECRET is configured for plaintext HTTP to {base}. Bearer tokens and memory payloads can be observed on the network; use HTTPS or an SSH tunnel."

def _warn_plaintext_bearer_auth(message: str) -> None:
    print(message, file=sys.stderr)

def _check_plaintext_bearer_guard(
    base: str,
    secret: str = "",
    warn: Callable[[str], None] | None = None,
) -> None:
    global _plaintext_bearer_warned
    if not _uses_plaintext_bearer_auth(base, secret):
        return
    message = _plaintext_bearer_auth_message(base)
    if os.environ.get("AGENTMEMORY_REQUIRE_HTTPS") == "1":
        raise RuntimeError(message)
    if not _plaintext_bearer_warned:
        _plaintext_bearer_warned = True
        (warn or _warn_plaintext_bearer_auth)(message)

def _api(base: str, path: str, body: dict | None = None, method: str = "POST", secret: str = "") -> dict | None:
    if not _validate_url(base):
        return None
    url = f"{base}/agentmemory/{path}"
    headers = {"Content-Type": "application/json"}
    auth = secret or os.environ.get("AGENTMEMORY_SECRET", "")
    _check_plaintext_bearer_guard(base, auth)
    if auth:
        headers["Authorization"] = f"Bearer {auth}"

    data = json.dumps(body).encode() if body else None
    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except (URLError, TimeoutError, json.JSONDecodeError):
        return None

def _api_bg(base: str, path: str, body: dict | None = None) -> None:
    t = threading.Thread(target=_api, args=(base, path, body), daemon=True)
    t.start()

class AgentMemoryTelemetryPlugin:
    def __init__(self):
        self._base = os.environ.get("AGENTMEMORY_URL", DEFAULT_BASE_URL)
        self._project = os.getcwd()

    def on_post_tool_call(self, tool_name: str, args: dict, result: Any = None, task_id: str = "", **kwargs: Any) -> None:
        is_error = False
        error_msg = ""
        output = result

        if isinstance(result, dict) and "error" in result:
            is_error = True
            error_msg = str(result["error"])

        if is_error:
            _api_bg(self._base, "observe", {
                "hookType": "post_tool_failure",
                "sessionId": task_id,
                "project": self._project,
                "cwd": self._project,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "data": {
                    "tool_name": tool_name,
                    "tool_input": json.dumps(args)[:4000],
                    "error": error_msg[:4000],
                },
            })
        else:
            clean_output = output
            if isinstance(output, str):
                clean_output = output[:8000]
            elif isinstance(output, dict):
                clean_output = json.dumps(output)[:8000]

            _api_bg(self._base, "observe", {
                "hookType": "post_tool_use",
                "sessionId": task_id,
                "project": self._project,
                "cwd": self._project,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "data": {
                    "tool_name": tool_name,
                    "tool_input": args,
                    "tool_output": clean_output,
                },
            })

    def on_pre_llm_call(self, task_id: str = "", **kwargs: Any) -> dict | None:
        prompt = kwargs.get("prompt", "")
        if not prompt and "messages" in kwargs:
            messages = kwargs["messages"]
            if messages and isinstance(messages, list):
                last_msg = messages[-1]
                if isinstance(last_msg, dict) and last_msg.get("role") == "user":
                    prompt = str(last_msg.get("content", ""))

        if prompt:
            _api_bg(self._base, "observe", {
                "hookType": "prompt_submit",
                "sessionId": task_id,
                "project": self._project,
                "cwd": self._project,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "data": {"prompt": prompt},
            })
        return None

    def on_subagent_stop(self, task_id: str = "", **kwargs: Any) -> None:
        _api_bg(self._base, "observe", {
            "hookType": "subagent_stop",
            "sessionId": task_id,
            "project": self._project,
            "cwd": self._project,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "data": {
                "agent_id": kwargs.get("agent_id", ""),
                "agent_type": kwargs.get("agent_type", "subagent"),
                "last_message": str(kwargs.get("last_message", ""))[:4000],
            },
        })

def register(ctx: Any) -> None:
    plugin = AgentMemoryTelemetryPlugin()
    ctx.register_hook("post_tool_call", plugin.on_post_tool_call)
    ctx.register_hook("pre_llm_call", plugin.on_pre_llm_call)
    ctx.register_hook("subagent_stop", plugin.on_subagent_stop)
