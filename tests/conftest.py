"""Loads the plugin against stub `providers` modules, so unit tests run without Hermes."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR = ROOT / "lowrouter"
CATALOG = json.loads((Path(__file__).resolve().parent / "lowrouter_models.json").read_text(encoding="utf-8"))
CATALOG_IDS = [item["id"] for item in CATALOG["data"]]

# expected picker for the fixture
RUNNABLE_IDS = [
    "auto/mistralai/mistral-large-2512",
    "auto/qwen/qwen3-coder-30b-a3b-instruct",
    "auto/anthropic/claude-haiku-4.5",
    "auto/moonshotai/kimi-k2.6",
    "alias/coder",
]


class StubProfile:
    def __init__(self, **fields):
        self.__dict__.update(fields)


def load_plugin():
    registered: list = []
    providers = types.ModuleType("providers")
    providers.register_provider = registered.append
    base = types.ModuleType("providers.base")
    base.ProviderProfile = StubProfile
    providers.base = base
    saved = {name: sys.modules.get(name) for name in ("providers", "providers.base")}
    sys.modules.update({"providers": providers, "providers.base": base})
    try:
        spec = importlib.util.spec_from_file_location(
            "lowrouter_provider_under_test", PLUGIN_DIR / "__init__.py", submodule_search_locations=[str(PLUGIN_DIR)])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module, registered


plugin, registered_profiles = load_plugin()
