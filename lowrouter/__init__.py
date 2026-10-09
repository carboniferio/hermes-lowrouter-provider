"""LowRouter model provider for Hermes.

Filters the picker down to models Hermes can run, takes context windows from our
catalog, and shows the EUR balance in /usage. No per-request cost: Hermes counts
in USD and we bill in EUR.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)

API_KEY_ENV = "LOWROUTER_API_KEY"
BASE_URL = "https://api.lowrouter.ai/v1"

MIN_CONTEXT = 64_000  # Hermes won't start below this
CATALOG_TTL_SECONDS = 3600.0
LISTED_PREFIXES = ("auto/", "alias/")
CURRENCY_SYMBOLS = {"EUR": "€", "USD": "$", "GBP": "£"}


def runs_in_hermes(item: Any) -> bool:
    """Chat model, takes tools, window not known to be too small."""
    if not isinstance(item, dict):
        return False
    if not str(item.get("id") or "").startswith(LISTED_PREFIXES):
        return False
    if item.get("modality") != "chat":
        return False
    if "tools" not in (item.get("supported_parameters") or []):
        return False
    context = item.get("context_length")
    return not (isinstance(context, int) and 0 < context < MIN_CONTEXT)


class _Catalog:
    """Last /v1/models answer, by model id."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, dict] = {}
        self._fetched_at = 0.0

    def store(self, items: list) -> None:
        with self._lock:
            self._items = {item["id"]: item for item in items if isinstance(item, dict) and "id" in item}
            self._fetched_at = time.monotonic()

    def fresh(self) -> bool:
        with self._lock:
            return bool(self._items) and time.monotonic() - self._fetched_at < CATALOG_TTL_SECONDS

    def context_length(self, model: str) -> int | None:
        with self._lock:
            context = (self._items.get(model) or {}).get("context_length")
        return context if type(context) is int and context > 0 else None


catalog = _Catalog()


def get_json(url: str, *, api_key: str | None, timeout: float) -> Any:
    from hermes_cli.urllib_security import open_credentialed_url

    request = urllib.request.Request(url)
    request.add_header("Accept", "application/json")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with open_credentialed_url(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except Exception as exc:  # noqa: BLE001
        logger.debug("lowrouter: GET %s failed: %s", url, exc)
        return None


def fetch_catalog(url: str, *, api_key: str | None, timeout: float) -> list | None:
    data = get_json(url, api_key=api_key, timeout=timeout)
    items = data.get("data") if isinstance(data, dict) else data
    return items if isinstance(items, list) else None


def configured_credentials() -> tuple[str | None, str]:
    try:
        from hermes_cli.auth import resolve_api_key_provider_credentials

        creds = resolve_api_key_provider_credentials("lowrouter")
    except Exception as exc:  # noqa: BLE001 - not set up yet
        logger.debug("lowrouter: no credentials: %s", exc)
        return None, BASE_URL
    return (creds.get("api_key") or None), (creds.get("base_url") or BASE_URL)


def format_money(amount: float, currency: str) -> str:
    symbol = CURRENCY_SYMBOLS.get(currency)
    return f"{symbol}{amount:,.2f}" if symbol else f"{amount:,.2f} {currency}"


def credit_lines(credits: Any) -> list[str]:
    data = credits.get("data") if isinstance(credits, dict) else None
    if not isinstance(data, dict):
        return []
    total, used = data.get("total_credits"), data.get("total_usage")
    currency = str(data.get("currency") or "").strip().upper()
    if not currency or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (total, used)):
        return []  # no currency, no number
    return [
        f"Credits balance: {format_money(max(0.0, float(total) - float(used)), currency)}",
        f"Purchased {format_money(float(total), currency)} • used {format_money(float(used), currency)}",
    ]


class LowRouterProfile(ProviderProfile):
    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        items = fetch_catalog((base_url or self.base_url).rstrip("/") + "/models", api_key=api_key, timeout=timeout)
        if items is None:
            return None
        catalog.store(items)
        return [item["id"] for item in items if runs_in_hermes(item)]

    def get_model_context_length(self, model: str) -> int | None:
        if not catalog.fresh():
            # with the key, so aliases are in there
            api_key, base_url = configured_credentials()
            items = fetch_catalog(base_url.rstrip("/") + "/models", api_key=api_key, timeout=5.0)
            if items is not None:
                catalog.store(items)
        return catalog.context_length(model)

    def fetch_account_usage(self, *, base_url: str | None = None, api_key: str | None = None) -> Any:
        if not api_key:
            return None
        lines = credit_lines(get_json((base_url or self.base_url).rstrip("/") + "/credits",
                                      api_key=api_key, timeout=8.0))
        if not lines:
            return None
        from agent.account_usage import AccountUsageSnapshot

        return AccountUsageSnapshot(provider=self.name, source="credits_api", fetched_at=datetime.now(timezone.utc),
                                    title="Account balance", details=tuple(lines))


lowrouter = LowRouterProfile(
    name="lowrouter",
    display_name="LowRouter",
    description="EU-hosted model router (OpenAI-compatible)",
    signup_url="https://lowrouter.ai/app/keys",
    env_vars=(API_KEY_ENV,),
    base_url=BASE_URL,
    auth_type="api_key",
    default_aux_model="auto/mistralai/mistral-small-2603",
    fallback_models=(
        "auto/mistralai/mistral-large-2512",
        "auto/mistralai/mistral-small-2603",
        "auto/qwen/qwen3-coder-30b-a3b-instruct",
        "auto/anthropic/claude-haiku-4.5",
    ),
)

register_provider(lowrouter)
