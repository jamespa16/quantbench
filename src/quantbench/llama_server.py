"""Lifecycle management and HTTP client for a `llama-server` subprocess."""

from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests


class LlamaServerError(RuntimeError):
    """Raised when llama-server fails to start, become healthy, or generate."""


@dataclass(frozen=True)
class GenerateResult:
    text: str
    completion_tokens: int
    elapsed_s: float


def _free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class LlamaServer:
    """One `llama-server` process serving a single GGUF model.

    Logs are written to `log_path` rather than captured via `subprocess.PIPE`:
    a benchmark run can produce many megabytes of per-request server output,
    and a pipe nobody drains fills its OS buffer and deadlocks the child
    process's writes -- silently hanging the "serial" benchmark step.
    """

    model_path: str
    binary: str = "llama-server"
    gpu_layers: str = "all"
    ctx_size: int | None = None
    host: str = "127.0.0.1"
    port: int = field(default_factory=_free_port)
    log_path: str | None = None
    startup_timeout: float = 300.0
    request_timeout: float = 600.0
    _process: subprocess.Popen | None = field(default=None, init=False, repr=False)
    _log_file: object = field(default=None, init=False, repr=False)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        if self.log_path is None:
            fd, self.log_path = tempfile.mkstemp(prefix="quantbench-llama-server-", suffix=".log")
            os.close(fd)
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.log_path, "w")

        cmd = [
            self.binary,
            "-m", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "-ngl", str(self.gpu_layers),
        ]
        if self.ctx_size is not None:
            cmd += ["--ctx-size", str(self.ctx_size)]

        try:
            self._process = subprocess.Popen(cmd, stdout=self._log_file, stderr=subprocess.STDOUT)
        except FileNotFoundError as e:
            raise LlamaServerError(
                f"could not launch {self.binary!r} -- is llama-server installed and on PATH?"
            ) from e
        self._wait_healthy()

    def _wait_healthy(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                code = self._process.returncode
                self.stop()
                raise LlamaServerError(
                    f"llama-server exited early (code {code}) while loading "
                    f"{self.model_path}; see {self.log_path}"
                )
            try:
                resp = requests.get(f"{self.base_url}/health", timeout=2)
                if resp.status_code == 200 and resp.json().get("status") == "ok":
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        self.stop()
        raise LlamaServerError(
            f"llama-server did not become healthy within {self.startup_timeout}s; see {self.log_path}"
        )

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> GenerateResult:
        """POST a chat completion; the server applies the GGUF's chat template."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.monotonic()
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={"messages": messages, "temperature": temperature, "max_tokens": max_tokens},
            timeout=self.request_timeout,
        )
        elapsed_s = time.monotonic() - start
        resp.raise_for_status()
        body = resp.json()
        return GenerateResult(
            text=body["choices"][0]["message"]["content"],
            completion_tokens=body["usage"]["completion_tokens"],
            elapsed_s=elapsed_s,
        )

    def stop(self, *, timeout: float = 15.0) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def __enter__(self) -> "LlamaServer":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
