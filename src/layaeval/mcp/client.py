"""MCP stdio client connecting to the Laya MCP server."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Sequence
from typing import Any

from .._version import __version__


class MCPClientError(Exception):
    """Raised when an MCP client or server interaction fails."""


class LayaMCPClient:
    """Manages an MCP stdio subprocess connection to `laya-mcp-server`."""

    def __init__(
        self,
        command: Sequence[str] | None = None,
        timeout: float = 60.0,
        env: dict[str, str] | None = None,
    ):
        """Initialize the client.

        Args:
            command: Command list to launch the MCP server. Defaults to
                `["laya-mcp-server"]` or `[sys.executable, "-m", "laya.mcp.server"]`.
            timeout: Subprocess read/write timeout in seconds.
            env: Custom environment variables to pass to the subprocess.
        """
        self.command = list(command) if command else self._default_command()
        self.timeout = timeout
        self.env = env
        self.process: subprocess.Popen | None = None
        self._msg_id = 0
        self._lock = threading.Lock()

    @staticmethod
    def _default_command() -> list[str]:
        # Try binary on PATH first, otherwise fallback to python module
        if shutil.which("laya-mcp-server"):
            return ["laya-mcp-server"]
        return [sys.executable, "-m", "laya.mcp.server"]

    def start(self) -> None:
        """Start the MCP server subprocess and perform the initialize handshake."""
        if self.process is not None and self.process.poll() is None:
            return  # Already running

        cmd_env = dict(os.environ)
        if self.env:
            cmd_env.update(self.env)
        # Ensure UTF-8 IO
        cmd_env.setdefault("PYTHONIOENCODING", "utf-8")

        try:
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=cmd_env,
            )
        except FileNotFoundError as exc:
            raise MCPClientError(
                f"Failed to execute command '{' '.join(self.command)}'. "
                "Ensure `laya[mcp]` is installed (`pip install 'layaeval[laya]'` or "
                "`pip install 'laya[mcp]'`)."
            ) from exc

        # Perform MCP initialize handshake
        self._initialize()

    def stop(self) -> None:
        """Gracefully terminate the subprocess."""
        if self.process is None:
            return
        try:
            if self.process.stdin and not self.process.stdin.closed:
                self.process.stdin.close()
            self.process.terminate()
            self.process.wait(timeout=2.0)
        except Exception:
            if self.process.poll() is None:
                self.process.kill()
        finally:
            self.process = None

    def __enter__(self) -> LayaMCPClient:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    def _send_request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send JSON-RPC request and wait for the response."""
        if self.process is None:
            self.start()

        if self.process is None or self.process.poll() is not None:
            raise MCPClientError("MCP server process is not running.")

        req_id = self._next_id()
        msg: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
        }
        if params is not None:
            msg["params"] = params

        payload = json.dumps(msg) + "\n"
        with self._lock:
            try:
                assert self.process.stdin is not None
                self.process.stdin.write(payload)
                self.process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                stderr_text = ""
                if self.process.stderr:
                    try:
                        stderr_text = self.process.stderr.read()
                    except Exception:
                        pass
                raise MCPClientError(
                    f"Failed writing to MCP server stdin: {exc}. Server stderr: {stderr_text}"
                ) from exc

            # Read response lines until we get the matching ID
            while True:
                assert self.process.stdout is not None
                line = self.process.stdout.readline()
                if not line:
                    stderr_msg = ""
                    if self.process.stderr:
                        try:
                            stderr_msg = self.process.stderr.read()
                        except Exception:
                            pass
                    raise MCPClientError(
                        f"MCP server closed stdout unexpectedly. Stderr: {stderr_msg}"
                    )
                line = line.strip()
                if not line:
                    continue
                try:
                    resp = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if resp.get("id") == req_id:
                    if "error" in resp:
                        raise MCPClientError(f"MCP server error: {resp['error']}")
                    return resp.get("result", {})

    def _send_notification(self, method: str, params: dict[str, Any] | None = None) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        msg: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            msg["params"] = params
        payload = json.dumps(msg) + "\n"
        with self._lock:
            try:
                assert self.process.stdin is not None
                self.process.stdin.write(payload)
                self.process.stdin.flush()
            except Exception:
                pass

    def _initialize(self) -> None:
        """Execute MCP initialize sequence."""
        init_params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "layaeval",
                "version": __version__,
            },
        }
        self._send_request("initialize", init_params)
        self._send_notification("notifications/initialized")

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Invoke an MCP tool by name and return its parsed output."""
        params = {
            "name": name,
            "arguments": arguments,
        }
        result = self._send_request("tools/call", params)
        if result.get("isError"):
            content = result.get("content", [])
            err_text = content[0].get("text", "Unknown tool error") if content else "Error"
            raise MCPClientError(f"Tool {name} failed: {err_text}")

        content = result.get("content", [])
        if not content:
            return result

        first = content[0]
        if first.get("type") == "text":
            text = first.get("text", "")
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        return content

    def predict(
        self,
        state: dict[str, Any],
        questions: dict[str, Any],
        model: str = "auto",
        **kwargs,
    ) -> dict[str, Any]:
        """Convenience method calling the `laya_predict` MCP tool."""
        args: dict[str, Any] = {
            "state": state,
            "questions": questions,
            "model": model,
        }
        args.update(kwargs)
        res = self.call_tool("laya_predict", args)
        if isinstance(res, dict) and "answers" in res:
            return res["answers"]
        return res


class DemoMCPClient:
    """Lightweight simulated client for offline demonstrations, testing, and notebooks."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def start(self) -> None:
        """No-op for demo client."""
        pass

    def stop(self) -> None:
        """No-op for demo client."""
        pass

    def __enter__(self) -> DemoMCPClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def predict(
        self,
        state: dict[str, Any],
        questions: dict[str, Any],
        model: str = "auto",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Simulate realistic calibrated neural answers for demo execution."""
        answers: dict[str, Any] = {}
        resp = str(state.get("response", state.get("final_result", ""))).lower()
        ctx = str(state.get("retrieved_context", state.get("agent_trace", ""))).lower()

        # Simple semantic overlap heuristic to provide natural outputs
        has_overlap = (
            any(w in ctx for w in resp.replace(".", " ").replace(",", " ").split() if len(w) > 3)
            if (resp and ctx)
            else True
        )

        for q_key, q_spec in questions.items():
            q_type = q_spec.get("type", "score")
            is_risk = any(
                k in q_key.lower()
                for k in ("risk", "contradiction", "conflict", "injection", "hallucination")
            )

            if q_type == "score":
                score = round(3.8 if has_overlap else 1.8, 2)
                answers[q_key] = {
                    "score": score,
                    "confidence": 0.94,
                    "answer_confidence": 0.91,
                }
            elif q_type == "noul":
                noul = round(0.06 if is_risk else (0.94 if has_overlap else 0.25), 3)
                answers[q_key] = {
                    "noul": noul,
                    "confidence": 0.95,
                    "answer_confidence": 0.92,
                }
            elif q_type == "choice":
                choices = q_spec.get("criteria", q_spec.get("choices", ["default"]))
                answers[q_key] = {
                    "choice": choices[0] if choices else "selected",
                    "confidence": 0.96,
                }
        return answers
