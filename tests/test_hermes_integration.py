"""End-to-end against a real Hermes checkout. Skipped when hermes_cli isn't importable.

No real network: /v1/models and /v1/credits come from a local server, chat goes to
Hermes' fake LLM (tests/fakes/fake_llm_provider.py).
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

hermes_cli = pytest.importorskip("hermes_cli")

from conftest import CATALOG, PLUGIN_DIR, RUNNABLE_IDS  # noqa: E402

HERMES_ROOT = Path(hermes_cli.__file__).resolve().parents[1]
API_KEY = "sk-lr-e2e-not-a-real-key"


@pytest.fixture
def home(tmp_path):
    home = tmp_path / "hermes-home"
    shutil.copytree(PLUGIN_DIR, home / "plugins" / "lowrouter", ignore=shutil.ignore_patterns("__pycache__"))
    (home / ".env").write_text(f"LOWROUTER_API_KEY={API_KEY}\n", encoding="utf-8", newline="\n")
    return home


def _env(home: Path) -> dict:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("OPENAI", "ANTHROPIC", "LOWROUTER", "OPENROUTER", "HERMES_"))}
    env.update(HERMES_HOME=str(home), PYTHONPATH=str(HERMES_ROOT), NO_COLOR="1")
    return env


def _python(home: Path, code: str):
    proc = subprocess.run([sys.executable, "-c", code], env=_env(home), cwd=str(home.parent),
                          capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_hermes_discovers_the_installed_plugin(home):
    found = _python(home, """
import json
from providers import get_provider_profile
from hermes_cli.auth import PROVIDER_REGISTRY
from hermes_cli.config import OPTIONAL_ENV_VARS
from hermes_cli.models import list_available_providers
profile = get_provider_profile("lowrouter")
row = PROVIDER_REGISTRY.get("lowrouter")
picker = [p for p in list_available_providers() if p["id"] == "lowrouter"]
print(json.dumps({"cls": type(profile).__name__, "base_url": profile.base_url, "env": list(profile.env_vars),
                  "row_url": row.inference_base_url, "row_env": list(row.api_key_env_vars),
                  "setup_lists_key": "LOWROUTER_API_KEY" in OPTIONAL_ENV_VARS, "picker": picker}))
""")
    assert found == {"cls": "LowRouterProfile", "base_url": "https://api.lowrouter.ai/v1", "env": ["LOWROUTER_API_KEY"],
                     "row_url": "https://api.lowrouter.ai/v1", "row_env": ["LOWROUTER_API_KEY"],
                     "setup_lists_key": True,
                     "picker": [{"id": "lowrouter", "label": "LowRouter", "aliases": [], "authenticated": True}]}


CREDITS = {"data": {"total_credits": 115, "total_usage": 112.879761, "currency": "EUR"}}


class _CatalogServer:

    def __init__(self):
        seen = self.seen = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):  # noqa: N802
                seen.append((self.path, self.headers.get("Authorization", "")))
                body = json.dumps(CREDITS if self.path.endswith("/credits") else CATALOG).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def test_the_model_list_and_context_windows_come_from_the_catalog(home):
    server = _CatalogServer()
    try:
        found = _python(home, f"""
import json
import hermes_cli.auth as auth
from providers import get_provider_profile
from agent.model_metadata import get_model_context_length
# Hermes has no base URL override for lowrouter; the key still comes from .env
resolve = auth.resolve_api_key_provider_credentials
auth.resolve_api_key_provider_credentials = lambda provider: {{**resolve(provider), "base_url": "{server.base_url}"}}
profile = get_provider_profile("lowrouter")
models = profile.fetch_models(api_key="{API_KEY}", base_url="{server.base_url}")
context = {{m: get_model_context_length(m, base_url=profile.base_url, api_key="{API_KEY}", provider="lowrouter")
           for m in ("auto/mistralai/mistral-large-2512", "auto/qwen/qwen3-coder-30b-a3b-instruct")}}
print(json.dumps({{"models": models, "context": context}}))
""")
    finally:
        server.close()
    assert found["models"] == RUNNABLE_IDS
    # without the plugin Hermes says 256000 and 262144 here
    assert found["context"] == {"auto/mistralai/mistral-large-2512": 256000,
                                "auto/qwen/qwen3-coder-30b-a3b-instruct": 128000}
    assert server.seen == [("/v1/models", f"Bearer {API_KEY}")]


def test_usage_shows_the_credit_balance_in_euros(home):
    server = _CatalogServer()
    try:
        lines = _python(home, f"""
import json
from agent.account_usage import fetch_account_usage, render_account_usage_lines
snapshot = fetch_account_usage("lowrouter", base_url="{server.base_url}", api_key="{API_KEY}")
print(json.dumps(render_account_usage_lines(snapshot)))
""")
    finally:
        server.close()
    assert lines == ["📈 Account balance", "Provider: lowrouter",
                     "Credits balance: €2.12", "Purchased €115.00 • used €112.88"]
    assert server.seen == [("/v1/credits", f"Bearer {API_KEY}")]


def _fake_llm_module():
    path = HERMES_ROOT / "tests" / "fakes" / "fake_llm_provider.py"
    if not path.is_file():
        pytest.skip("this Hermes checkout has no tests/fakes/fake_llm_provider.py")
    spec = importlib.util.spec_from_file_location("hermes_fake_llm_provider", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_a_real_turn_goes_out_with_the_routed_model_id_and_the_key(home):
    fake = _fake_llm_module()
    model = "auto/mistralai/mistral-large-2512"
    with fake.FakeLLMServer([fake.Text("pong")], api_key=API_KEY) as llm:
        (home / "config.yaml").write_text(
            "model:\n  provider: lowrouter\n"
            f"  base_url: {llm.base_url}\n  default: {model}\n  context_length: 256000\n"
            "agent:\n  api_max_retries: 1\n",
            encoding="utf-8", newline="\n")
        proc = subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", "chat", "-q", "Reply with: pong", "--oneshot", "-Q", "-t", "todo"],
            env=_env(home), cwd=str(home.parent), capture_output=True, text=True, timeout=240, check=False)
        assert proc.returncode == 0, (proc.stdout[-1000:], proc.stderr[-2000:])
        assert "pong" in proc.stdout
        main = llm.main_requests()
    assert main, "the turn made no chat request"
    assert main[0]["model"] == model
    assert main[0]["tools"], "Hermes sent no tool schemas"
