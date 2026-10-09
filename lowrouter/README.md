# LowRouter provider for Hermes

Adds [LowRouter](https://lowrouter.ai) as a provider in [Hermes Agent](https://github.com/NousResearch/hermes-agent).
Pick LowRouter in `hermes model`, or run
`hermes chat --provider lowrouter -m auto/mistralai/mistral-large-2512`.

LowRouter is an OpenAI-compatible router run from the EU. An `auto/<creator>/<model>` ID can be
served by several European or global hosts, and LowRouter fails over between them. Maintained by
the LowRouter team.

## Why not a custom endpoint?

You can point a custom endpoint at `https://api.lowrouter.ai/v1` and it works. What you get with
the plugin instead (Hermes v0.21.5, live API, 2026-10-09):

| | Custom endpoint | Plugin |
|---|---|---|
| Model list | 719 IDs, including every pinned route, embedding and realtime models, and models without tool calling | the 180 `auto/` chat models with tool calling, plus your own `alias/` models |
| Context window | read from `/v1/models` | read from `/v1/models`. A named provider without this lookup gets Hermes' 256K fallback, or a family default: 262,144 for `auto/qwen/qwen3-coder-30b-a3b-instruct`, which has 128,000 |
| `/usage` | nothing | your credit balance in euros |

## Install

```sh
hermes plugins install lowrouter
```

Get a key at [lowrouter.ai/app/keys](https://lowrouter.ai/app/keys) and put it in `~/.hermes/.env`:

```sh
LOWROUTER_API_KEY=sk-lr-...
```

Then pick LowRouter in `hermes model`, or in `~/.hermes/config.yaml`:

```yaml
model:
  provider: lowrouter
  default: auto/mistralai/mistral-large-2512
```

No need to set `context_length`.

## Models

The picker shows the entries from `/v1/models` that Hermes can run:

- `auto/<creator>/<model>` IDs and your account's `alias/<name>` models. Pinned routes
  (`<host>/<creator>/<model>`) are skipped since the `auto/` ID already covers them. An alias has
  the capabilities of the model it points to.
- Chat models with tool calling. Hermes does everything through tools.
- At least 64,000 tokens of context, Hermes' minimum. If we don't publish a window for a model, it
  stays in the list and Hermes looks the window up itself.

Tested on 2026-10-09, one turn each with a tool call:

| Model | Input tokens | Output tokens |
|---|---|---|
| `auto/mistralai/mistral-large-2512` | 37,342 | 70 |
| `auto/mistralai/mistral-small-2603` | 12,836 | 81 |
| `auto/qwen/qwen3-coder-30b-a3b-instruct` | 34,453 | 152 |
| `auto/anthropic/claude-haiku-4.5` | 53,764 | 226 |

Hermes falls back to these four when it can't reach the catalog.
`auto/mistralai/mistral-small-2603` is the auxiliary model (context compression and other
background work).

The plugin doesn't touch reasoning effort.

## Cost

LowRouter bills in euros. A routed model has several routes (hosts, regions) with different
prices, and you pay for the route that served the request. Each response has the amount in
`usage.cost`, with `usage.currency: "EUR"`. Price ranges per model are on
[lowrouter.ai/models](https://lowrouter.ai/models).

Hermes tracks cost in US dollars (`/insights`, the dashboard). The plugin doesn't convert, so
Hermes shows LowRouter sessions with an unknown cost. `/usage` shows your balance in euros, from
`/v1/credits`.

## Security and footprint

- Registers one provider, `lowrouter`. No tools, hooks, commands or background threads.
- Chat requests go to `https://api.lowrouter.ai/v1` (or your `model.base_url`) through Hermes' own
  client, with `LOWROUTER_API_KEY`.
- The plugin's own requests go to the same base URL with the same key, through Hermes'
  `open_credentialed_url`: `GET /v1/models` for the model list and context windows (cached for an
  hour), and `GET /v1/credits` on `/usage`. Without a key the catalog is fetched anonymously; it's
  public.
- LowRouter forwards your prompts, tool schemas and tool results to the host serving the model.
  Hosts per model are listed on [lowrouter.ai/models](https://lowrouter.ai/models).
- No files read or written, no processes, no dependencies beyond Hermes. Hermes reads
  `LOWROUTER_API_KEY`; the plugin asks Hermes for it
  (`resolve_api_key_provider_credentials`) only to look up context windows.

## Compatibility

Hermes 0.21.5 or newer. CI tests against v2026.9.24 (0.21.5) and `main` daily.

## Changelog

### 0.1.0

- First release: filtered model list with aliases, context windows from the catalog, EUR balance
  in `/usage`.

## License

[Apache-2.0](https://github.com/carboniferio/hermes-lowrouter-provider/blob/main/LICENSE)
