"""Unit tests, no Hermes needed."""

from __future__ import annotations

import sys
import types

import pytest
from conftest import CATALOG, RUNNABLE_IDS, plugin, registered_profiles


@pytest.fixture(autouse=True)
def empty_catalog():
    plugin.catalog.store([])
    yield
    plugin.catalog.store([])


@pytest.fixture(autouse=True)
def no_configured_key(monkeypatch):
    monkeypatch.setattr(plugin, "configured_credentials", lambda: (None, plugin.BASE_URL))


@pytest.fixture
def requests(monkeypatch):
    seen: list = []

    def fake_fetch(url, *, api_key, timeout):
        seen.append((url, api_key))
        return list(CATALOG["data"])

    monkeypatch.setattr(plugin, "fetch_catalog", fake_fetch)
    return seen


def _item(**fields):
    return {"id": "auto/lab/model", "modality": "chat", "supported_parameters": ["tools"],
            "context_length": 128000, **fields}


def test_registers_one_lowrouter_profile():
    assert registered_profiles == [plugin.lowrouter]
    profile = plugin.lowrouter
    assert profile.name == "lowrouter"
    assert profile.base_url == "https://api.lowrouter.ai/v1"
    assert profile.env_vars == ("LOWROUTER_API_KEY",)
    assert profile.auth_type == "api_key"


def test_fallback_and_aux_models_are_routed_ids():
    profile = plugin.lowrouter
    assert profile.fallback_models
    assert all(model.startswith("auto/") for model in (*profile.fallback_models, profile.default_aux_model))


@pytest.mark.parametrize("item, kept", [
    (_item(), True),
    (_item(context_length=None), True),                  # unknown: let Hermes decide
    (_item(context_length=64000), True),                 # exactly the minimum
    (_item(context_length=40960), False),                # too small
    (_item(id="alias/coder"), True),                     # account alias
    (_item(id="mistral/mistralai/mistral-large-2512"), False),  # pinned route
    (_item(modality="embedding"), False),
    (_item(modality="realtime"), False),
    (_item(supported_parameters=None), False),           # no tools
    (_item(supported_parameters=["tool_choice"]), False),
    ("auto/lab/model", False),
    (None, False),
])
def test_runs_in_hermes(item, kept):
    assert plugin.runs_in_hermes(item) is kept


def test_fetch_models_returns_the_runnable_models_in_catalog_order(requests):
    models = plugin.lowrouter.fetch_models(api_key="sk-lr-test", base_url=None)
    assert models == RUNNABLE_IDS
    assert requests == [("https://api.lowrouter.ai/v1/models", "sk-lr-test")]


def test_fetch_models_follows_a_configured_base_url(requests):
    plugin.lowrouter.fetch_models(api_key="sk-lr-test", base_url="http://127.0.0.1:9080/v1/")
    assert requests == [("http://127.0.0.1:9080/v1/models", "sk-lr-test")]


def test_a_failed_fetch_returns_none_so_hermes_uses_the_fallback_list(monkeypatch):
    monkeypatch.setattr(plugin, "fetch_catalog", lambda url, *, api_key, timeout: None)
    assert plugin.lowrouter.fetch_models(api_key="sk-lr-test") is None


@pytest.mark.parametrize("model, context", [
    ("auto/mistralai/mistral-large-2512", 256000),
    ("auto/qwen/qwen3-coder-30b-a3b-instruct", 128000),
    ("auto/moonshotai/kimi-k2.6", None),                 # no window in catalog
    ("auto/unknown/model", None),
])
def test_context_length_comes_from_the_catalog(requests, model, context):
    assert plugin.lowrouter.get_model_context_length(model) == context


def test_context_lookup_fetches_the_catalog_once_without_a_key_when_none_is_configured(requests):
    for _ in range(3):
        plugin.lowrouter.get_model_context_length("auto/mistralai/mistral-large-2512")
    assert requests == [("https://api.lowrouter.ai/v1/models", None)]


def test_context_lookup_uses_the_configured_key_and_base_url(requests, monkeypatch):
    monkeypatch.setattr(plugin, "configured_credentials", lambda: ("sk-lr-test", "http://127.0.0.1:9080/v1"))
    assert plugin.lowrouter.get_model_context_length("alias/coder") == 128000
    assert requests == [("http://127.0.0.1:9080/v1/models", "sk-lr-test")]


def test_context_lookup_reuses_the_picker_fetch(requests):
    plugin.lowrouter.fetch_models(api_key="sk-lr-test")
    assert plugin.lowrouter.get_model_context_length("auto/anthropic/claude-haiku-4.5") == 200000
    assert len(requests) == 1


def test_context_lookup_refetches_after_the_ttl(requests, monkeypatch):
    plugin.lowrouter.get_model_context_length("auto/mistralai/mistral-large-2512")
    monkeypatch.setattr(plugin, "CATALOG_TTL_SECONDS", 0.0)
    plugin.lowrouter.get_model_context_length("auto/mistralai/mistral-large-2512")
    assert len(requests) == 2


def test_context_lookup_returns_none_when_the_catalog_is_unreachable(monkeypatch):
    monkeypatch.setattr(plugin, "fetch_catalog", lambda url, *, api_key, timeout: None)
    assert plugin.lowrouter.get_model_context_length("auto/mistralai/mistral-large-2512") is None


CREDITS = {"data": {"total_credits": 115, "total_usage": 112.879761, "currency": "EUR"}}


@pytest.mark.parametrize("body, lines", [
    (CREDITS, ["Credits balance: €2.12", "Purchased €115.00 • used €112.88"]),
    ({"data": {"total_credits": 10, "total_usage": 12.5, "currency": "EUR"}},
     ["Credits balance: €0.00", "Purchased €10.00 • used €12.50"]),
    ({"data": {"total_credits": 1500, "total_usage": 0, "currency": "chf"}},
     ["Credits balance: 1,500.00 CHF", "Purchased 1,500.00 CHF • used 0.00 CHF"]),
    ({"data": {"total_credits": 115, "total_usage": 112.88}}, []),          # no currency
    ({"data": {"total_credits": "115", "total_usage": 1, "currency": "EUR"}}, []),
    ({"data": {"total_credits": True, "total_usage": 1, "currency": "EUR"}}, []),
    ({"error": "unauthorized"}, []),
    (None, []),
])
def test_credit_lines_carry_the_currency_lowrouter_declares(body, lines):
    assert plugin.credit_lines(body) == lines


@pytest.fixture
def account_usage_module(monkeypatch):
    module = types.ModuleType("agent.account_usage")
    module.AccountUsageSnapshot = lambda **fields: fields
    monkeypatch.setitem(sys.modules, "agent", types.ModuleType("agent"))
    monkeypatch.setitem(sys.modules, "agent.account_usage", module)


def test_account_usage_shows_the_balance_in_euros(monkeypatch, account_usage_module):
    seen = []
    monkeypatch.setattr(plugin, "get_json", lambda url, *, api_key, timeout: seen.append((url, api_key)) or CREDITS)
    snapshot = plugin.lowrouter.fetch_account_usage(base_url=None, api_key="sk-lr-test")
    assert seen == [("https://api.lowrouter.ai/v1/credits", "sk-lr-test")]
    assert snapshot["provider"] == "lowrouter"
    assert snapshot["title"] == "Account balance"
    assert snapshot["details"] == ("Credits balance: €2.12", "Purchased €115.00 • used €112.88")


def test_account_usage_needs_a_key_and_a_usable_answer(monkeypatch, account_usage_module):
    monkeypatch.setattr(plugin, "get_json", lambda url, *, api_key, timeout: None)
    assert plugin.lowrouter.fetch_account_usage(api_key=None) is None
    assert plugin.lowrouter.fetch_account_usage(api_key="sk-lr-test") is None
