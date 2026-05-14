# workers-ai

see it in action:
https://patrick-ring-motive.github.io/workers-ai/

OpenAI-compatible chat completions proxy built on Cloudflare Workers AI. Auto-selects the best available free model, falls back gracefully on rate limits.

## What It Does

Exposes a single Worker endpoint that:

- Accepts OpenAI-style `POST /v1/chat/completions` requests
- Routes to the best available Cloudflare Workers AI model automatically
- Falls back to the next best model on rate limit errors
- Returns OpenAI-compatible responses (streaming SSE or JSON)
- Scrapes live rate limit data from CF docs on cold start

## Endpoints

### `GET /`

Returns current model selection and full tier map.

```json
{
  "model": "@cf/meta/llama-3.1-8b-instruct",
  "source": "auto",
  "tiers": [
    {
      "rateLimit": 100,
      "models": [
        { "name": "@cf/...", "score": 47.2, "rateLimited": false }
      ]
    }
  ]
}
```

### `POST /`

OpenAI-compatible chat completions.

**Request:**

```json
{
  "messages": [
    { "role": "system", "content": "You are helpful." },
    { "role": "user", "content": "Hello!" }
  ],
  "temperature": 0.7,
  "stream": true
}
```

**Response (non-streaming):** Standard `chat.completion` object.  
**Response (streaming):** SSE stream of `chat.completion.chunk` events, terminated with `data: [DONE]`.

## Model Selection

On cold start the worker:

1. Scrapes `developers.cloudflare.com/workers-ai/platform/limits` for live rate limits
1. Fetches all `Text Generation` beta models from the CF AI API
1. Scores and ranks them
1. Groups into rate limit tiers

### Scoring Heuristic

|Signal                           |Points|
|---------------------------------|------|
|Parameter count                  |0–30  |
|Context window (per 1k tokens)   |0–25  |
|Function calling support         |+10   |
|LoRA support                     |+5    |
|Instruct/chat fine-tune          |+5    |
|Cloudflare-native source         |+3    |
|Recency (decays over months)     |0–3   |
|Specialized (SQL/code/math/embed)|−15   |

Only **non-deprecated beta models** are candidates.

### Fallback Strategy

On each request, the worker builds an ordered list:

1. `CF_MODEL` env override (if set)
1. Highest-scoring currently non-rate-limited model
1. Remaining models, lowest rate limit tier first

On a `429` / rate limit error, the model is marked exhausted for the current 60-second window and the next candidate is tried. Non-rate-limit errors abort immediately.

## Configuration

|Variable       |Required|Description                                |
|---------------|--------|-------------------------------------------|
|`CF_ACCOUNT_ID`|Yes     |Cloudflare account ID                      |
|`CF_API_TOKEN` |Yes     |API token with Workers AI read access      |
|`CF_MODEL`     |No      |Pin a specific model, bypassing auto-select|
|`AI`           |Yes     |Workers AI binding (set in `wrangler.toml`)|

### `wrangler.toml` Example

```toml
name = "workers-ai-proxy"
main = "worker.js"
compatibility_date = "2024-01-01"

[ai]
binding = "AI"

[vars]
CF_ACCOUNT_ID = "your-account-id"
CF_API_TOKEN = "your-api-token"
```

## Using with OpenAI-Compatible Clients

Point any OpenAI SDK or tool at your Worker URL:

**JavaScript:**

```js
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://your-worker.workers.dev",
  apiKey: "unused", // required by SDK but ignored
});

const res = await client.chat.completions.create({
  model: "auto", // ignored — worker picks
  messages: [{ role: "user", content: "Hello!" }],
  stream: true,
});
```

**Continue (VS Code extension):** Set `apiBase` to your Worker URL in `~/.continue/config.json`.

**LangChain:**

```js
import { ChatOpenAI } from "@langchain/openai";

const model = new ChatOpenAI({
  configuration: { baseURL: "https://your-worker.workers.dev" },
  apiKey: "unused",
});
```

## Chat Tester UI

`index.html` is a standalone dark-mode chat interface for testing the worker directly in a browser. No build step — just open it.

- Saves endpoint URL to `localStorage`
- Streams responses token by token
- Maintains full conversation history per session
- Displays active model name from the `GET /` response

## Limitations

- Model quality varies significantly across the free tier
- Rate limit data is scraped from docs at startup — may drift if CF changes the page format
- In-memory rate limit tracking resets on Worker restart/cold start
- No auth on the proxy itself — anyone with the URL can use it
