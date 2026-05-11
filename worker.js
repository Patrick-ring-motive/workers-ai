

let test = `ok`;
const DEFAULT_MODEL = "@cf/google/gemma-7b-it-lora";
globalThis.env ??= {};

let resolvedModel = null;
let modelTiers = null;

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

const parse = x =>{
  try{
    return Object(JSON.parse(x));
  }catch{
    return Object(x);
  }
};

const fetchResponse = async(...args)=>{
  try{
    return await fetch(...args);
  }catch(e){
    return new Response(String(e),{status:500,statusText:String(e)});
  }
};



// Tracks per-model rate limit windows: { modelName: { count, windowStart } }
const rateLimitTracker = {};
const RATE_LIMIT_WINDOW_MS = 60_000;

const CORS_HEADERS = {
  "access-control-allow-origin": "*",
  "access-control-allow-methods": "GET,POST,OPTIONS",
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

      const canParse = x =>{
        try{
          JSON.parse(x);
          return true;
        }catch{
          return false;
        }
      };

const isArray = (val) => Array.isArray(val) || val instanceof Array;
const isString = (val) => typeof val === "string" || val instanceof String;
function removeJsonBlocks(str) {
  let result = str;
  let prev;
  do {
    prev = result;
    result = result.replace(/\{[^{}]*\}/g, (match) => {
      try {
       // JSON.parse(match);
        return '';
      } catch {
        return match;
      }
    });
  } while (result !== prev);
  return result.trim();
}
function extractAssistantText(result) {
  let eh = (()=>{
    if (!result) return "";
  if (isString(result)) {
    try {
      const parsed = JSON.parse(result);
      if (parsed && typeof parsed === 'object') return extractAssistantText(parsed);
    } catch {}
    return result;
  }

  if (isString(result.response)) return result.response;
  if (isString(result.output_text)) return result.output_text;
  if (isString(result.text)) return result.text;
  if (isString(result.content)) return result.content;
  if (isString(result.generated_text)) return result.generated_text;
  if (isString(result.result?.response)) return result.result.response;
  if (isString(result.result?.output_text)) return result.result.output_text;

  let maybeChoice = result.choices?.[0]?.message?.content;
  if (isString(maybeChoice)){
    if(canParse(maybeChoice)){
      return extractAssistantText(parse(maybeChoice));
    }
    return maybeChoice;
  }

  if (result.response && typeof result.response === 'object') return extractAssistantText(result.response);
  if (result.result && typeof result.result === 'object') return extractAssistantText(result.result);

  return removeJsonBlocks(stringify(result));
})();
   if(canParse(eh)){
      eh = extractAssistantText(parse(eh));
    }
    return (eh);
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
          const data = trimmed.replace(/^data:/i,'').trim();

          if (data === "[DONE]") {
            const finishChunk = {
              id, object: "chat.completion.chunk", created, model,
              choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
            };
            controller.enqueue(encoder.encode(`data: ${stringify(finishChunk)}\n\ndata: [DONE]\n\n`));
            return;
          }

          let parsed;
          try { parsed = JSON.parse(data); } catch { }

          let token = parsed?.response ?? parsed?.token ?? parsed?.text ?? data;
          if (!token) continue;

          if(canParse(token)){
            const parsed2 = JSON.parse(token);
            if(parsed2.choices?.[0]?.delta?.content){
              token = parsed2.choices?.[0]?.delta?.content;
              model = parsed2.model ?? model;
            }
          }

          const openaiChunk = {
            id, object: "chat.completion.chunk", created, model,
            choices: [{ index: 0, delta: { role:"assistant",content: token }, finish_reason: null }],
          };
          controller.enqueue(encoder.encode(`data: ${stringify(openaiChunk)}\n\n`));
        }
      },
    }),
  );
}

/* ── Ollama-format helpers ────────────────────────────────── */

function toOllamaChatResponse({ model, content }) {
  return {
    model,
    created_at: new Date().toISOString(),
    message: { role: "assistant", content },
    done: true,
    total_duration: 0,
    load_duration: 0,
    prompt_eval_count: 0,
    eval_count: 0,
  };
}

function toOllamaGenerateResponse({ model, content }) {
  return {
    model,
    created_at: new Date().toISOString(),
    response: content,
    done: true,
    total_duration: 0,
    load_duration: 0,
    prompt_eval_count: 0,
    eval_count: 0,
  };
}

function cfStreamToOllamaStream(cfStream, { model, isChat }) {
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
          const data = trimmed.replace(/^data:/i, '').trim();

          if (data === "[DONE]") {
            const done = isChat
              ? { model, created_at: new Date().toISOString(), message: { role: "assistant", content: "" }, done: true, total_duration: 0, eval_count: 0 }
              : { model, created_at: new Date().toISOString(), response: "", done: true, total_duration: 0, eval_count: 0 };
            controller.enqueue(encoder.encode(stringify(done) + "\n"));
            return;
          }

          let parsed;
          try { parsed = JSON.parse(data); } catch { continue; }

          const token = parsed?.response ?? parsed?.token ?? parsed?.text ?? "";
          if (!token) continue;

          const ollamaChunk = isChat
            ? { model, created_at: new Date().toISOString(), message: { role: "assistant", content: token }, done: false }
            : { model, created_at: new Date().toISOString(), response: token, done: false };
          controller.enqueue(encoder.encode(stringify(ollamaChunk) + "\n"));
        }
      },
    }),
  );
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

async function getModelTiers(){
  try{
    const res = await fetchResponse('https://text-generation-models.language-models-aggregate.workers.dev/');
    const text = await res.text();
    const obj = parse(text);
    obj.summarizer = res.headers.get('summarizer');
    obj.summarizerType = res.headers.get('summarizer-type');
    return obj;
  }catch{
    return [];
  }
}

const longestArray = (...args)=>{
  let longest = [];
  for(const arg of args){
    if(arg.length > longest.length){
      longest = arg;
    }
  }
  return longest;
};

async function runAI(AI, model, aiInput,summarizer,summarizerType){
  if(aiInput.messages[0]?.role !== 'system'){
    aiInput.messages.unshift({role:'system',content:`Current DateTime: ${new Date().toISOString()}`});
  }
  try{
    return await AI.run(model, aiInput);
  }catch(e){
    if(!e.message.includes('5021')){
      throw e;
    }
    try{
      return await AI.run('@cf/ibm-granite/granite-4.0-h-micro',aiInput);
    }catch{}
    const tokens = (+e.message.match(/tokens\s*\((\d+)\)/)[1]||0);
    const limit = Math.floor((+e.message.match(/limit\s*\((\d+)\)/)[1]||0) * 0.8);
    let text = aiInput.messages.map(x=>x.content).join('\n');
    try{
      if(!summarizer){
        throw summarizer;
      }
      if(summarizerType === 'Summarization'){
        const {summary} = await AI.run(summarizer, {
          input_text: text,
          max_length: limit
        });
        text = summary;
      }else{
        text = [...new Set(text.split('\n'))].join('\n');
        text = [...new Set(text.split('.'))].join('.');
        text = [...new Set(text.split(' '))].join(' ');
      }   
    }catch{
      const over = tokens - limit;
      const overPercent = over / tokens;
      const cut = Math.floor(overPercent * text.length);
      text = text.slice(cut);
      if(summarizer === 'thanos'){
        text = text.slice(Math.round(text.length/2));
      }
    }

    const messages = longestArray(text.split('\n'),[...new Intl.Segmenter("en", { granularity: "sentence" }).segment(text)].map(x=>x.segment.trim()).filter(Boolean));
    const oldMessages = aiInput.messages.slice(aiInput.messages.length - messages.length);
    aiInput.messages = messages.map((x,i)=>({role:oldMessages[i]?.role||'user',content:x}));
    aiInput.messages.push({role:String(oldMessages[oldMessages.length - 1]?.role),content:String([...oldMessages].map(x=>x.content).join('\n').split('\n').pop())});
    if(summarizer){
      return runAI(AI,model, aiInput);
    }else{
      return runAI(AI,model,aiInput,'thanos');
    }
  }
}

export default {
  async fetch(request, env) {

    if (request.method === "OPTIONS") {
      return new Response(null, { headers: CORS_HEADERS });
    }

    // Lazy-init

    modelTiers ??= getModelTiers();
    if(modelTiers instanceof Promise){
      modelTiers = await modelTiers;
    }
    const summarizer= modelTiers?.summarizer;
    const summarizerType= modelTiers?.summarizerType;

    // Resolve current best available model from tiers
    resolvedModel = (modelTiers && pickAvailableModel(modelTiers)) || DEFAULT_MODEL;
    const currentModel = env.CF_MODEL || resolvedModel;

    const url = new URL(request.url);
    const path = url.pathname.replace(/\/+$/, '') || '/';

    /* ── Ollama API routes ────────────────────────────────── */

    // Ollama health check: GET / must return plain text "Ollama is running"
    if (request.method === 'GET' && path === '/') {
      return new Response('Ollama is running', { headers: { 'content-type': 'text/plain; charset=utf-8', ...CORS_HEADERS } });
    }

    if (path === '/api/version') {
      return json({ version: "0.6.2" });
    }

    if (path === '/api/tags' && request.method === 'GET') {
      const models = [];
      if (modelTiers) {
        for (const tier of modelTiers) {
          for (const m of tier.models) {
            models.push({
              name: m.name,
              model: m.name,
              modified_at: new Date().toISOString(),
              size: 0,
              digest: "",
              details: { format: "gguf", family: "", parameter_size: "", quantization_level: "" },
            });
          }
        }
      }
      if (models.length === 0) {
        models.push({ name: currentModel, model: currentModel, modified_at: new Date().toISOString(), size: 0, digest: "", details: {} });
      }
      return json({ models });
    }

    if (path === '/api/show' && request.method === 'POST') {
      let showBody; try { showBody = await request.clone().json(); } catch { showBody = {}; }
      return json({
        modelfile: `FROM ${showBody?.name || currentModel}`,
        parameters: "",
        template: "",
        details: { format: "gguf", family: "", parameter_size: "", quantization_level: "" },
      });
    }

    if (path === '/api/pull' && request.method === 'POST') {
      return json({ status: "success" });
    }

    const isOllamaChat = path === '/api/chat' && request.method === 'POST';
    const isOllamaGenerate = path === '/api/generate' && request.method === 'POST';
    const isOllama = isOllamaChat || isOllamaGenerate;

    /* ── Model info endpoint ────────────────────────────── */

    if (request.method === "GET" && !request.url.includes('test')) {
      return json({
        model: currentModel,
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
        body = {...Object.fromEntries(request.headers.entries()),...body};
      }
    }

    let messages = body?.messages;

    // Ollama /api/generate uses "prompt" instead of "messages"
    if (isOllamaGenerate && !messages?.length) {
      const prompt = body?.prompt || "";
      messages = [];
      if (body?.system) messages.push({ role: "system", content: body.system });
      messages.push({ role: "user", content: prompt });
    }

    if(body && !messages?.length){
      messages = Object.entries(Object(body)).map(([key,value])=>({role:String(key),content:stringify(value)}));
    }
    if(!messages?.length){
      messages = stringify(text).split('\n').map(x=>({role:"user",content:x}));
    }

    if(request.url.includes('test')){
      messages = stringify(test).split('\n').map(x=>({role:"user",content:x}));
    }

    // Ollama defaults stream to true; OpenAI defaults to false
    const stream = isOllama ? (body?.stream !== false) : Boolean(body?.stream);
    const requestId = `chatcmpl-${crypto.randomUUID()}`;
    const created = Math.floor(Date.now() / 1000);

    const aiInput = { messages };
    // Ollama puts params in "options", OpenAI puts them top-level
    const paramSource = isOllama ? (body?.options || {}) : body;
    for (const key of ["temperature", "top_p", "max_tokens", "stop"]) {
      if (paramSource?.[key] !== undefined) aiInput[key] = paramSource[key];
    }
    // Ollama uses num_predict for max tokens
    if (isOllama && body?.options?.num_predict !== undefined) {
      aiInput.max_tokens = body.options.num_predict;
    }

    // Build ordered list of models to try:
    // 1. env override, 2. highest-scoring model, 3. tiers low→high limit
    const modelsToTry = [];
    if (env.CF_MODEL) {
      modelsToTry.push(env.CF_MODEL);
    }
    if (modelTiers) {
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
          const cfStream = await runAI(env.AI, model, streamInput,summarizer,summarizerType);
          if (isOllama) {
            return new Response(
              cfStreamToOllamaStream(cfStream, { model, isChat: isOllamaChat }),
              {
                headers: {
                  "content-type": "application/x-ndjson",
                  "cache-control": "no-cache, no-transform",
                  ...CORS_HEADERS,
                },
              },
            );
          }

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

        const result = await runAI(env.AI, model, aiInput,summarizer,summarizerType);
        const content = extractAssistantText(result);

        if (isOllamaChat) {
          return json(toOllamaChatResponse({ model, content }), { headers: { "cache-control": "no-store" } });
        }
        if (isOllamaGenerate) {
          return json(toOllamaGenerateResponse({ model, content }), { headers: { "cache-control": "no-store" } });
        }

        return json(toOpenAIChatResponse({ id: requestId, model, content }), {
          headers: { "cache-control": "no-store" },
        });
      } catch (error) {
        lastError = error;
        const msg = String(error?.message || error).toLowerCase();
        if (msg.includes("rate limit") || msg.includes("429") || msg.includes("too many")) {
          markRateLimited(model);
          console.log(`Rate limited on ${model}, trying next…`);
          continue;
        }
        break;
      }
    }

    const errPayload = isOllama
      ? { error: "request failed: " + String(lastError?.message) }
      : { error: "Cloudflare AI request failed. " + String(lastError?.message), detail: lastError instanceof Error ? lastError.message : String(lastError), aiInput };

    return json(errPayload, { status: 502 });
  },
};
