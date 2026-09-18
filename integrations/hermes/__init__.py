"""
agentmemory memory provider for Hermes Agent.

Drop this folder into ~/.hermes/plugins/agentmemory/
or install via: hermes plugin install agentmemory

Requires agentmemory server running: npx @agentmemory/agentmemory
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import subprocess
from pathlib import Path
from pathlib import PurePath

logger = logging.getLogger(__name__)


def _resolve_project(cwd: str) -> str:
    """Canonical project scope, matching the hooks' resolveProject order:
    AGENTMEMORY_PROJECT_NAME env override, git toplevel basename, cwd basename.
    Keeps Hermes sessions in the same project bucket as every other agent."""
    explicit = os.environ.get("AGENTMEMORY_PROJECT_NAME", "").strip()
    if explicit:
        return explicit
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        if top:
            return PurePath(top).name
    except Exception:
        pass
    return PurePath(cwd).name or cwd
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError

try:
    from agent.memory_provider import MemoryProvider
except ImportError:
    from abc import ABC, abstractmethod

    class MemoryProvider(ABC):
        @property
        @abstractmethod
        def name(self) -> str: ...
        @abstractmethod
        def is_available(self) -> bool: ...
        @abstractmethod
        def initialize(self, session_id: str, **kwargs: Any) -> None: ...
        @abstractmethod
        def get_tool_schemas(self) -> list[dict]: ...
        @abstractmethod
        def handle_tool_call(self, name: str, args: dict) -> str: ...
        def get_config_schema(self) -> list[dict]: return []
        def save_config(self, values: dict, hermes_home: str) -> None: pass
        def system_prompt_block(self) -> str: return ""
        def prefetch(self, query: str, **kwargs: Any) -> str: return ""
        def queue_prefetch(self, query: str, **kwargs: Any) -> None: pass
        def sync_turn(self, user: str, assistant: str, **kwargs: Any) -> None: pass
        def on_session_end(self, messages: list, **kwargs: Any) -> None: pass
        def on_pre_compress(self, messages: list, **kwargs: Any) -> None: pass
        def on_memory_write(self, action: str, target: str, content: str, **kwargs: Any) -> None: pass
        def shutdown(self, **kwargs: Any) -> None: pass


DEFAULT_BASE_URL = "http://localhost:3111"
TIMEOUT = 5
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
_plaintext_bearer_warned = False

# agentmemory's documented runtime config lives at ~/.agentmemory/.env.
# When agentmemory is launched as a systemd user service (or any other
# process manager that loads that file directly), those values never
# reach an interactive shell. `hermes memory status` then reads
# os.environ in the Hermes CLI process, finds AGENTMEMORY_URL /
# AGENTMEMORY_SECRET unset, and reports the plugin as "Missing" even
# though the service is healthy and live sessions can use it (#250).
#
# Preload the file at plugin-import time using os.environ.setdefault so
# we never override anything the user explicitly set in the shell. The
# preload is best-effort and silent on any failure (file absent,
# unreadable, malformed) — the plugin falls back to its existing default
# (http://localhost:3111) and Hermes status reflects that.
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
    # Guarantee AGENTMEMORY_URL is set so `hermes memory status` never
    # reports it as Missing when a user runs agentmemory at the default
    # localhost:3111 (or via systemd with the URL line commented out in
    # ~/.agentmemory/.env because it matches the default). #520.
    os.environ.setdefault("AGENTMEMORY_URL", DEFAULT_BASE_URL)


_preload_agentmemory_dotenv()


def _validate_url(base: str) -> bool:
    if not base:
        return False
    try:
        parsed = urlparse(base)
        # .port raises ValueError on a non-numeric or out-of-range port
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


def _reset_plaintext_bearer_guard_for_tests() -> None:
    global _plaintext_bearer_warned
    _plaintext_bearer_warned = False


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


class AgentMemoryProvider(MemoryProvider):

    @property
    def name(self) -> str:
        return "agentmemory"

    def is_available(self) -> bool:
        # Hermes contract: no network calls in is_available.
        base = os.environ.get("AGENTMEMORY_URL", DEFAULT_BASE_URL)
        return _validate_url(base)

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self._base = os.environ.get("AGENTMEMORY_URL", DEFAULT_BASE_URL)
        self._session_id = session_id
        self._cwd = kwargs.get("cwd", os.getcwd())
        self._project = _resolve_project(self._cwd)
        if os.environ.get("AGENTMEMORY_REQUIRE_HTTPS") == "1":
            _check_plaintext_bearer_guard(self._base, os.environ.get("AGENTMEMORY_SECRET", ""))

        _api(self._base, "session/start", {
            "sessionId": session_id,
            "project": self._project,
            "cwd": self._cwd,
        })

    def get_config_schema(self) -> list[dict]:
        return [
            {
                "key": "url",
                "description": "agentmemory server URL",
                "default": DEFAULT_BASE_URL,
                "env_var": "AGENTMEMORY_URL",
            },
            {
                "key": "secret",
                "description": "agentmemory auth secret (optional)",
                "secret": True,
                "required": False,
                "env_var": "AGENTMEMORY_SECRET",
            },
        ]

    def save_config(self, values: dict, hermes_home: str) -> None:
        config_path = Path(hermes_home) / "agentmemory.json"
        config_path.write_text(json.dumps(values, indent=2))

    def system_prompt_block(self) -> str:
        result = _api(self._base, "context", {
            "sessionId": self._session_id,
            "project": self._project,
        })
        if result and result.get("context"):
            return result["context"]
        return ""

    def prefetch(self, query: str, **kwargs: Any) -> str:
        result = _api(self._base, "smart-search", {
            "query": query,
            "limit": 5,
        })
        if not result or not result.get("results"):
            return ""

        lines = []
        for r in result["results"][:5]:
            obs = r.get("observation", r)
            title = obs.get("title", "")
            narrative = obs.get("narrative", "")
            if title:
                lines.append(f"- {title}: {narrative[:200]}")
        return "\n".join(lines) if lines else ""

    def queue_prefetch(self, query: str, **kwargs: Any) -> None:
        _api_bg(self._base, "smart-search", {"query": query, "limit": 3})

    def get_tool_schemas(self) -> list[dict]:
        return [
            {
                "name": "memory_recall",
                "description": "Search agentmemory for past observations by keyword",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "description": "Max results", "default": 10},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "memory_save",
                "description": "Save an insight, decision, or pattern to long-term memory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "What to remember"},
                        "type": {
                            "type": "string",
                            "enum": ["pattern", "preference", "architecture", "bug", "workflow", "fact"],
                            "description": "Memory type",
                        },
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "memory_search",
                "description": "Hybrid semantic + keyword search across all memories",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            },
        ]

    def handle_tool_call(self, name: str, args: dict) -> str:
        # Hermes stores the return value as the tool result `content` in the
        # session history. Anthropic-protocol providers reject non-string
        # content with a 400 on the next request, so always serialize to a
        # JSON string here — matches what agentmemory's main MCP server does
        # in src/mcp/standalone.ts (`{ type: "text", text: JSON.stringify(...) }`).
        if name == "memory_recall":
            result = _api(self._base, "search", {
                "query": args["query"],
                "limit": args.get("limit", 10),
            })
            if not result:
                return json.dumps({"results": []})
            items = []
            for r in result.get("results", []):
                obs = r.get("observation", r)
                items.append({
                    "title": obs.get("title", ""),
                    "type": obs.get("type", ""),
                    "narrative": obs.get("narrative", ""),
                    "importance": obs.get("importance", 0),
                    "timestamp": obs.get("timestamp", ""),
                })
            return json.dumps({"results": items})

        if name == "memory_save":
            result = _api(self._base, "remember", {
                "content": args["content"],
                "type": args.get("type", "fact"),
            })
            return json.dumps(result or {"success": False})

        if name == "memory_search":
            result = _api(self._base, "smart-search", {
                "query": args["query"],
                "limit": args.get("limit", 5),
            })
            if not result:
                return json.dumps({"results": []})
            items = []
            for r in result.get("results", []):
                obs = r.get("observation", r)
                items.append({
                    "title": obs.get("title", ""),
                    "narrative": obs.get("narrative", "")[:300],
                    "score": r.get("combinedScore", r.get("score", 0)),
                })
            return json.dumps({"results": items})

        return json.dumps({"error": f"Unknown tool: {name}"})

    def sync_turn(self, user: str, assistant: str, **kwargs: Any) -> None:
        _api_bg(self._base, "observe", {
            "hookType": "post_tool_use",
            "sessionId": kwargs.get("session_id", self._session_id),
            "project": self._project,
            "cwd": self._cwd,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "data": {
                "tool_name": "conversation",
                "tool_input": user[:500],
                "tool_output": assistant[:2000],
            },
        })

    # Dual-use: as a MemoryProvider method, memory_manager passes messages
    # positionally; as a ctx.register_hook callback, Hermes fires this with
    # keyword-only turn metadata (completed, failed, ..., session_id) and no
    # messages. Make messages optional so the hook path works.
    def on_session_end(self, messages: list | None = None, **kwargs: Any) -> None:
        _api(self._base, "session/end", {
            "sessionId": kwargs.get("session_id", self._session_id),
        })

    def on_pre_compress(self, messages: list, **kwargs: Any) -> None:
        result = _api(self._base, "context", {
            "sessionId": kwargs.get("session_id", self._session_id),
            "project": self._project,
        })
        if result and result.get("context"):
            messages.insert(0, {
                "role": "user",
                "content": f"[agentmemory context before compaction]\n{result['context']}",
            })

    def on_memory_write(self, action: str, target: str, content: str, **kwargs: Any) -> None:
        if action in ("add", "update") and content:
            _api_bg(self._base, "remember", {
                "content": content,
                "type": "fact",
            })

    # ------------------------------------------------------------------
    # Plugin-context hooks (ctx.register_hook) — feature parity with the
    # Claude Code / Cursor / Copilot adapters, which feed agentmemory's
    # 12 HookTypes. All telemetry is fire-and-forget (_api_bg) so no hook
    # ever blocks or can fail the agent turn.
    # ------------------------------------------------------------------

    def on_session_start(self, **kwargs: Any) -> None:
        # Agentmemory already gets this from initialize(); nothing extra to
        # record here. Declared so the manifest lists every hook implemented.
        pass

    def pre_tool_call(self, tool_name: str = "", args: dict | None = None, **kwargs: Any) -> None:
        _api_bg(self._base, "observe", self._payload("pre_tool_use", {
            "tool_name": tool_name,
            "tool_input": _trunc_json(args),
        }))

    def post_tool_call(self, tool_name: str = "", args: dict | None = None,
                       outcome: Any = None, result: Any = None, **kwargs: Any) -> None:
        # Hermes reports failures either via an {"error": ...} result or an
        # outcome payload; mirror agentmemory's post_tool_use / post_tool_failure.
        payload = outcome if isinstance(outcome, dict) else {}
        if payload.get("status") in ("cancelled", "error") or (
            isinstance(result, dict) and "error" in result
        ):
            error = payload.get("error") or (result.get("error") if isinstance(result, dict) else "")
            _api_bg(self._base, "observe", self._payload("post_tool_failure", {
                "tool_name": tool_name,
                "tool_input": _trunc_json(args),
                "error": str(error)[:4000],
            }))
            return
        output = result if result is not None else payload.get("result")
        _api_bg(self._base, "observe", self._payload("post_tool_use", {
            "tool_name": tool_name,
            "tool_input": args,
            "tool_output": _trunc_str(output, 8000),
        }))

    def pre_llm_call(self, **kwargs: Any) -> None:
        prompt = kwargs.get("prompt", "")
        if not prompt:
            messages = kwargs.get("messages")
            if isinstance(messages, list) and messages:
                last = messages[-1]
                if isinstance(last, dict) and last.get("role") == "user":
                    prompt = str(last.get("content", ""))
        if prompt:
            _api_bg(self._base, "observe", self._payload("prompt_submit", {
                "prompt": _trunc_str(prompt, 8000),
            }))

    def post_llm_call(self, response: Any = None, **kwargs: Any) -> None:
        # Anything agentmemory wants from the response side arrives via
        # sync_turn; observing here would duplicate that capture.
        pass

    def subagent_start(self, **kwargs: Any) -> None:
        _api_bg(self._base, "observe", self._payload("subagent_start", {
            "agent_id": str(kwargs.get("subagent_id", kwargs.get("agent_id", ""))),
            "agent_type": str(kwargs.get("role", kwargs.get("agent_type", "subagent"))),
            "task": _trunc_str(kwargs.get("task", kwargs.get("goal", "")), 4000),
        }))

    def subagent_stop(self, **kwargs: Any) -> None:
        _api_bg(self._base, "observe", self._payload("subagent_stop", {
            "agent_id": str(kwargs.get("subagent_id", kwargs.get("agent_id", ""))),
            "agent_type": str(kwargs.get("role", kwargs.get("agent_type", "subagent"))),
            "last_message": _trunc_str(kwargs.get("summary", kwargs.get("last_message", "")), 4000),
        }))

    def on_session_finalize(self, **kwargs: Any) -> None:
        _api(self._base, "session/end", {
            "sessionId": kwargs.get("session_id", self._session_id),
        })

    def on_session_reset(self, **kwargs: Any) -> None:
        # A /new-style reset kills the old session's context; marks it ended
        # so agentmemory doesn't keep the dead session open.
        _api_bg(self._base, "session/end", {
            "sessionId": kwargs.get("old_session_id", kwargs.get("session_id", self._session_id)),
        })

    def agent_loop_stopped(self, **kwargs: Any) -> None:
        pass

    def _payload(self, hook_type: str, data: dict) -> dict:
        return {
            "hookType": hook_type,
            "sessionId": getattr(self, "_session_id", ""),
            "project": self._project,
            "cwd": getattr(self, "_cwd", os.getcwd()),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "data": data,
        }

    def shutdown(self, **kwargs: Any) -> None:
        pass


def _trunc_str(value: Any, limit: int) -> Any:
    if isinstance(value, str):
        return value[:limit]
    if value is None:
        return ""
    return str(value)[:limit]


def _trunc_json(value: Any, limit: int = 4000) -> Any:
    if value is None:
        return ""
    try:
        return json.dumps(value)[:limit]
    except (TypeError, ValueError):
        return str(value)[:limit]


def register(ctx: Any) -> None:
    provider = AgentMemoryProvider()
    ctx.register_memory_provider(provider)
    # Plugin.yaml's hooks: list is a manifest declaration only — Hermes does not
    # auto-wire it. Each hook below must also be registered as a callback, or
    # invoke_hook() finds no subscriber and the hook never fires. When this same
    # module is also loaded through plugins/memory's provider collector, the
    # fallback-hook dedupe (plugins_ledger._register_fallback_hook) makes those
    # duplicate registrations inert rather than double-firing.
    for _hook in (
        "on_session_start",
        "pre_tool_call",
        "post_tool_call",
        "pre_llm_call",
        "post_llm_call",
        "subagent_start",
        "subagent_stop",
        "on_session_finalize",
        "on_session_reset",
        "agent_loop_stopped",
        "on_session_end",
    ):
        _cb = getattr(provider, _hook, None)
        if callable(_cb):
            try:
                ctx.register_hook(_hook, _cb)
            except Exception as exc:
                logger.debug("agentmemory register_hook(%s) failed: %s", _hook, exc)
