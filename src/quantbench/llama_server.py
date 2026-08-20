"""Lifecycle management and HTTP client for a `llama-server` subprocess."""

from __future__ import annotations

import contextlib
import io
import os
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import requests


class LlamaServerError(RuntimeError):
    """Raised when llama-server fails to start, become healthy, or generate."""


@dataclass(frozen=True)
class GenerateResult:
    text: str
    completion_tokens: int
    elapsed_s: float


def _free_port() -> int:
    # NOTE: There is a brief race window between binding the socket (which
    # reserves the port) and returning it. Another process could grab the
    # same port if it starts before the caller actually binds to the
    # returned value. In practice this only matters when multiple
    # quantbench instances run concurrently on the same host.
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
    ctx_size: int | None = None
    host: str = "127.0.0.1"
    port: int = field(default_factory=_free_port)
    log_path: str | None = None
    startup_timeout: float = 300.0
    # Per-request timeout. The orchestrator scales this with max_tokens: the
    # flat 600s default would kill legitimate long generations on slow or
    # CPU-only machines (32000 tokens at a mere 5 tok/s takes ~107 minutes).
    request_timeout: float = 600.0
    # llama-server --parallel slot count; lets the server batch concurrent
    # requests so the client can issue them in parallel. KV cache memory
    # scales with this (slots x ctx-size).
    parallel: int | None = None
    # Extra attempts after the first for transient failures (connection
    # errors, timeouts, HTTP 5xx). 4xx client errors are never retried.
    max_retries: int = 2
    _process: subprocess.Popen | None = field(default=None, init=False, repr=False)
    _log_file: io.TextIOWrapper | None = field(default=None, init=False, repr=False)
    gpu_layers: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        if self.log_path is None:
            fd, self.log_path = tempfile.mkstemp(prefix="quantbench-llama-server-", suffix=".log")
            os.close(fd)
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        self._log_file = cast("io.TextIOWrapper", open(self.log_path, "w"))  # noqa: SIM115

        cmd = [
            self.binary,
            "-m", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
        ]
        if self.gpu_layers is not None:
            cmd += ["-ngl", str(self.gpu_layers)]
        if self.ctx_size is not None:
            cmd += ["--ctx-size", str(self.ctx_size)]
        if self.parallel is not None:
            cmd += ["-np", str(self.parallel)]

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
            process = self._process
            if process is None or process.poll() is not None:
                code = process.returncode if process is not None else None
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
        max_tokens: int = 32000,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> GenerateResult:
        """POST a chat completion; the server applies the GGUF's chat template.

        Transient failures (connection errors, read timeouts, HTTP 5xx) are
        retried up to `max_retries` extra times with linear backoff, so one
        flaky request out of ~164xN doesn't abort an entire quant.
        `elapsed_s` covers the whole call including any retries, so callers
        can use it for throughput accounting.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        # Typed for requests' JsonType (mypy rejects dict[str, object]).
        body: dict[str, str | int | float | list[dict[str, str]]] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            body["seed"] = seed

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            start = time.monotonic()
            try:
                resp = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=body,
                    timeout=self.request_timeout,
                )
            except (requests.ConnectionError, requests.Timeout) as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                raise LlamaServerError(
                    f"request to llama-server failed after {attempt + 1} attempts: {e}"
                ) from e
            if 500 <= resp.status_code < 600:
                last_error = LlamaServerError(f"llama-server returned HTTP {resp.status_code}")
                if attempt < self.max_retries:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                raise last_error
            resp.raise_for_status()  # 4xx client errors are not retried
            data = resp.json()
            return GenerateResult(
                text=data["choices"][0]["message"]["content"],
                completion_tokens=data["usage"]["completion_tokens"],
                elapsed_s=time.monotonic() - start,
            )
        raise LlamaServerError(f"request to llama-server failed: {last_error}")

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

    def __enter__(self) -> LlamaServer:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
