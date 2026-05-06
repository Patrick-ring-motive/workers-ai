
const DEFAULT_MODEL = "@cf/google/gemma-7b-it-lora";
globalThis.env ??= {};

let resolvedModel = null;
let modelTiers = null;
let modelLimits = null;

// Tracks per-model rate limit windows: { modelName: { count, windowStart } }
const rateLimitTracker = {};
const RATE_LIMIT_WINDOW_MS = 60_000;

const CORS_HEADERS = {
  "access-control-allow-origin": "*",
  "access-control-allow-methods": "POST,OPTIONS",
  "access-control-allow-headers": "authorization,content-type",
};

function json(data, init = {}) {
  return new Response(JSON.stringify(data), {
    ...init,
    headers: {
      "content-type": "application/json",
      ...CORS_HEADERS,
      ...init.headers,
    },
  });
}

const isArray = (val) => Array.isArray(val) || val instanceof Array;
const isString = (val) => typeof val === "string" || val instanceof String;

function extractAssistantText(result) {
  if (isString(result)) return result;
  if (!result) return "";

  if (isString(result.response)) return result.response;
  if (isString(result.output_text)) return result.output_text;
  if (isString(result.text)) return result.text;
  if (isString(result.result?.response)) return result.result.response;
  if (isString(result.result?.output_text)) return result.result.output_text;

  const maybeChoice = result.choices?.[0]?.message?.content;
  if (isString(maybeChoice)) return maybeChoice;

  return JSON.stringify(result);
}

function toOpenAIChatResponse({ id, model, content }) {
  return {
    id,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content,
        },
        finish_reason: "stop",
      },
    ],
  };
}

function cfStreamToOpenAIStream(cfStream, { id, model, created }) {
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  let buffer = "";

  return cfStream.pipeThrough(
    new TransformStream({
      transform(chunk, controller) {
        buffer += decoder.decode(chunk, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data:")) continue;
          const data = trimmed.slice(5).trim();

          if (data === "[DONE]") {
            const finishChunk = {
              id, object: "chat.completion.chunk", created, model,
              choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
            };
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(finishChunk)}\n\ndata: [DONE]\n\n`));
            return;
          }

          let parsed;
          try { parsed = JSON.parse(data); } catch { continue; }

          const token = parsed.response ?? parsed.token ?? parsed.text ?? "";
          if (!token) continue;

          const openaiChunk = {
            id, object: "chat.completion.chunk", created, model,
            choices: [{ index: 0, delta: { content: token }, finish_reason: null }],
          };
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(openaiChunk)}\n\n`));
        }
      },
    }),
  );
}

function isBetaModel(model) {
  if (model.beta === true) return true;
  if (isArray(model.properties)) {
    return model.properties.some((p) => p.property_id === "beta" && p.value === "true");
  }
  return false;
}

function isDeprecated(model) {
  if (model.deprecated === true) return true;
  if (isArray(model.properties)) {
    return model.properties.some((p) => (p.property_id === "deprecated" && p.value === "true") || (p.property_id === "planned_deprecation_date" && new Date(p.value).getTime() < new Date().getTime()));
  }
  return false;
}

function paramScore(name) {
  const SUFFIXES = { t: 1000, b: 1, m: 0.001, k: 0.000001 };
  const matches = [...name.matchAll(/(\d+(?:\.\d+)?)\s*x?\s*(\d+(?:\.\d+)?)?\s*([tbmk])\b/gi)];
  if (!matches.length) return 0;
  return Math.max(...matches.map((m) => {
    const num = Number.parseFloat(m[1]);
    const mult = m[2] ? Number.parseFloat(m[2]) : 1;
    const scale = SUFFIXES[m[3].toLowerCase()] ?? 1;
    return num * mult * scale;
  }));
}

function getProps(model) {
  const props = {};
  if (isArray(model.properties)) {
    for (const p of model.properties) props[p.property_id] = p.value;
  }
  return props;
}

const SPECIALIZED_KEYWORDS = ["sql", "code", "coder", "math", "embed", "rerank"];

function scoreModel(model) {
  const props = getProps(model);
  let score = 0;

  // Parameter count (0–30): larger models are generally more capable
  const params = paramScore(model.name);
  score += Math.min(params * 4, 30);

  // Context window (0–25): 1 pt per 1k tokens, capped at 25
  const ctx = Number.parseInt(props.context_window || "0", 10);
  score += Math.min(ctx / 1000, 25);

  // Function calling support (+10) – strong capability signal
  if (props.function_calling === "true") score += 10;

  // LoRA adapter support (+5) – adds flexibility
  if (props.lora === "true") score += 5;

  // Instruct / chat fine-tune bonus (+5)
  const lower = model.name.toLowerCase();
  if (/instruct|chat/.test(lower)) score += 5;

  // Cloudflare-native source gets a small integration bonus (+3)
  if (model.source === 1) score += 3;

  // Penalise specialised models that aren't great for general chat (–15)
  const lowerDesc = (model.description || "").toLowerCase();
  if (SPECIALIZED_KEYWORDS.some((kw) => lower.includes(kw) || lowerDesc.includes(kw))) {
    score -= 15;
  }

  // Slight recency bonus: newer models get up to +3
  const created = new Date(model.created_at).getTime();
  const ageMonths = (Date.now() - created) / (1000 * 60 * 60 * 24 * 30);
  score += Math.max(3 - ageMonths * 0.1, 0);

  return score;
}

function rankModels(candidates) {
  return candidates
    .map((m) => ({ ...m, _score: scoreModel(m) }))
    .sort((a, b) => b._score - a._score);
}

function getModelRateLimit(modelName, taskName, limits) {
  if (limits.models[modelName]) return limits.models[modelName].limit;
  if (taskName && limits.tasks[taskName]) return limits.tasks[taskName];
  return limits.tasks["Text Generation"] ?? 300;
}

function buildModelTiers(rankedModels, limits) {
  const tierMap = new Map(); // limit -> [models]
  for (const m of rankedModels) {
    const taskName = m.task?.name || "Text Generation";
    const limit = getModelRateLimit(m.name, taskName, limits);
    if (!tierMap.has(limit)) tierMap.set(limit, []);
    tierMap.get(limit).push({ ...m, _rateLimit: limit });
  }
  // Sort tiers: lowest limit first, models within each tier already ranked by score
  return [...tierMap.entries()]
    .sort(([a], [b]) => a - b)
    .map(([limit, models]) => ({ limit, models }));
}

function isRateLimited(modelName, limit) {
  const now = Date.now();
  const entry = rateLimitTracker[modelName];
  if (!entry || now - entry.windowStart >= RATE_LIMIT_WINDOW_MS) return false;
  return entry.count >= limit;
}

function trackRequest(modelName) {
  const now = Date.now();
  const entry = rateLimitTracker[modelName];
  if (!entry || now - entry.windowStart >= RATE_LIMIT_WINDOW_MS) {
    rateLimitTracker[modelName] = { count: 1, windowStart: now };
  } else {
    entry.count++;
  }
}

function markRateLimited(modelName) {
  const now = Date.now();
  const entry = rateLimitTracker[modelName];
  if (!entry || now - entry.windowStart >= RATE_LIMIT_WINDOW_MS) {
    rateLimitTracker[modelName] = { count: Infinity, windowStart: now };
  } else {
    entry.count = Infinity;
  }
}

function pickAvailableModel(tiers) {
  // Always prefer the highest-scoring model across all tiers
  let best = null;
  for (const tier of tiers) {
    for (const m of tier.models) {
      if (!isRateLimited(m.name, tier.limit) && (!best || m._score > best._score)) {
        best = m;
      }
    }
  }
  return best?.name ?? null;
}

async function scrapeModelLimits() {
  const response = await fetch('https://developers.cloudflare.com/workers-ai/platform/limits/index.md');
  const mdContent = await response.text();
  
  const result = {
    tasks: {},   // Default limits for the category
    models: {}   // Specific overrides
  };

  const lines = mdContent.split('\n');
  let currentTask = null;

  for (const line of lines) {
    // 1. Identify the Task (### [Task Name])
    const taskMatch = line.match(/^###\s+\[(.*?)\]/);
    if (taskMatch) {
      currentTask = taskMatch[1];
      continue;
    }

    // 2. Identify a Model Override (* [@model] is X...)
    const modelMatch = line.match(/\*\s+\[(@[\w\-\.\/]+)\]\(.*?\)\s+is\s+(\d+)/);
    if (modelMatch) {
      const modelName = modelMatch[1];
      const limit = parseInt(modelMatch[2], 10);
      result.models[modelName] = {
        limit,
        task: currentTask
      };
      continue;
    }

    // 3. Identify Task Default (* X requests per minute)
    // Only captures if it's a bullet point and NOT a model override
    const defaultMatch = line.match(/^\*\s+(\d+)\s+requests\s+per\s+minute/);
    if (defaultMatch && currentTask) {
      result.tasks[currentTask] = parseInt(defaultMatch[1], 10);
    }
  }

  return result;
}

async function buildTieredModels(env, limits) {
  if (!env.CF_ACCOUNT_ID || !env.CF_API_TOKEN) return null;
  try {
    const url =
      `https://api.cloudflare.com/client/v4/accounts/${encodeURIComponent(env.CF_ACCOUNT_ID)}` +
      `/ai/models/search?task=Text+Generation&per_page=100`;
    const res = await fetch(url, {
      headers: { authorization: `Bearer ${env.CF_API_TOKEN}` },
    });
    if (!res.ok) return null;
    const { result } = await res.json();
    if (!isArray(result) || result.length === 0) return null;
    const candidates = result.filter((m) => isBetaModel(m) && !isDeprecated(m));
    const ranked = rankModels(candidates);
    const tiers = buildModelTiers(ranked, limits);
    console.log("Model tiers:", tiers.map((t) =>
      `[${t.limit} rpm: ${t.models.map((m) => `${m.name}(${m._score.toFixed(1)})`).join(", ")}]`
    ).join(" → "));
    return tiers;
  } catch(e) {
    console.log(e);
    return null;
  }
}

const stringify = x =>{
  if(isString(x)){
    return String(x);
  }
  try{
    return String(JSON.stringify(x));
  }catch{
    return String(x);
  }
};

export default {
  async fetch(request, env) {
    env.CF_ACCOUNT_ID ??= '';
    env.CF_API_TOKEN ??= '';
    if (request.method === "OPTIONS") {
      return new Response(null, { headers: CORS_HEADERS });
    }

    // Lazy-init

    modelLimits ??= scrapeModelLimits();
    if(modelLimits instanceof Promise){
      modelLimits = await modelLimits;
    }

    modelTiers ??= buildTieredModels(env, modelLimits);
    if(modelTiers instanceof Promise){
      modelTiers = await modelTiers;
    }

    // Resolve current best available model from tiers
    resolvedModel = (modelTiers && pickAvailableModel(modelTiers)) || DEFAULT_MODEL;

    if (request.method === "GET") {
      return json({
        model: env.CF_MODEL || resolvedModel,
        source: env.CF_MODEL ? "env_override" : "auto",
        tiers: modelTiers?.map((t) => ({
          rateLimit: t.limit,
          models: t.models.map((m) => ({
            name: m.name,
            score: m._score,
            rateLimited: isRateLimited(m.name, t.limit),
          })),
        })),
      });
    }

    let text;
    let body;
    try {
      text = (await request.text()).trim();
      body = JSON.parse(text);
    } catch {
      if(!text){
        body = Object.fromEntries(new URL(request.url).searchParams.entries());
        body = {...Object.fromEntries(request.headers.entries(),...body);
      }
    }

    let messages = body?.messages;
    if(body && !messages?.length){
      messages = Object.entries(Object(body)).map(stringify);
    }
    if(!messages?.length){
      messages = stringify(text).split('\n');
    }

    const stream = Boolean(body?.stream);
    const requestId = `chatcmpl-${crypto.randomUUID()}`;
    const created = Math.floor(Date.now() / 1000);

    const aiInput = { messages };
    for (const key of ["temperature", "top_p", "max_tokens", "stop"]) {
      if (body[key] !== undefined) aiInput[key] = body[key];
    }

    // Build ordered list of models to try:
    // 1. env override, 2. highest-scoring model, 3. tiers low→high limit
    const modelsToTry = [];
    if (env.CF_MODEL) {
      modelsToTry.push(env.CF_MODEL);
    }
    if (modelTiers) {
      // Find the globally highest-scoring available model and try it first
      let bestModel = null;
      for (const tier of modelTiers) {
        for (const m of tier.models) {
          if (!isRateLimited(m.name, tier.limit) && (!bestModel || m._score > bestModel._score)) {
            bestModel = m;
          }
        }
      }
      if (bestModel && !modelsToTry.includes(bestModel.name)) {
        modelsToTry.push(bestModel.name);
      }
      // Then fill in remaining models tier by tier (lowest limit → highest)
      for (const tier of modelTiers) {
        for (const m of tier.models) {
          if (!modelsToTry.includes(m.name) && !isRateLimited(m.name, tier.limit)) {
            modelsToTry.push(m.name);
          }
        }
      }
    }
    if (modelsToTry.length === 0) modelsToTry.push(DEFAULT_MODEL);

    let lastError = null;
    for (const model of modelsToTry) {
      try {
        trackRequest(model);

        if (stream) {
          const streamInput = { ...aiInput, stream: true };
          const cfStream = await env.AI.run(model, streamInput);
          return new Response(
            cfStreamToOpenAIStream(cfStream, { id: requestId, model, created }),
            {
              headers: {
                "content-type": "text/event-stream; charset=utf-8",
                "cache-control": "no-cache, no-transform",
                connection: "keep-alive",
                ...CORS_HEADERS,
              },
            },
          );
        }

        const result = await env.AI.run(model, aiInput);
        const content = extractAssistantText(result);
        return json(toOpenAIChatResponse({ id: requestId, model, content }), {
          headers: { "cache-control": "no-store" },
        });
      } catch (error) {
        lastError = error;
        const msg = String(error?.message || "").toLowerCase();
        if (msg.includes("rate limit") || msg.includes("429") || msg.includes("too many")) {
          markRateLimited(model);
          console.log(`Rate limited on ${model}, trying next…`);
          continue;
        }
        // Non-rate-limit error — don't retry with a different model
        break;
      }
    }

    return json(
      {
        error: "Cloudflare AI request failed. " + String(lastError?.message),
        detail: lastError instanceof Error ? lastError.message : String(lastError),
      },
      { status: 502 },
    );
  },
};
