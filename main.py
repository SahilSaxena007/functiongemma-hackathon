import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, re, time
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types

# ── Toggle: set False before submitting ───────────────────────────────
DEBUG = True

def _debug(*args):
    if DEBUG:
        print("[DBG]", *args)


# ═══════════════════════════════════════════════════════════════════════
#  Core generators
# ═══════════════════════════════════════════════════════════════════════

def generate_cactus(messages, tools, system_prompt=None):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = cactus_init(functiongemma_path)

    cactus_tools = [{"type": "function", "function": t} for t in tools]

    if system_prompt is None:
        system_prompt = (
            "You are a function-calling assistant. "
            "You MUST respond with exactly one function call. "
            "Pick the best matching function and extract the arguments "
            "from the user message. Only output the function call, nothing else."
        )

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": system_prompt}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        _debug("  cactus JSON FAIL:", repr(raw_str[:200]) if raw_str else "None")
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0}

    result = {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }
    _debug(f"  cactus → calls={result['function_calls']}  conf={result['confidence']:.3f}  {result['total_time_ms']:.0f}ms")
    return result


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]
    start_time = time.time()

    gemini_response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config=types.GenerateContentConfig(tools=gemini_tools),
    )

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    _debug(f"  cloud → calls={function_calls}  {total_time_ms:.0f}ms")
    return {"function_calls": function_calls, "total_time_ms": total_time_ms}


# ═══════════════════════════════════════════════════════════════════════
#  Dynamic tool-keyword extraction  (zero hardcoded tool names)
# ═══════════════════════════════════════════════════════════════════════

_STOPWORDS = {
    "a", "an", "the", "for", "to", "of", "in", "on", "is", "it",
    "and", "or", "with", "by", "at", "from", "that", "this", "be",
    "are", "was", "were", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "can", "shall", "given", "type", "object", "string",
    "integer", "number", "boolean", "current", "specific", "based",
    "optional", "required", "default", "value", "example", "return",
}


def _extract_tool_keywords(tool):
    """Build keyword set from a tool's own definition at runtime."""
    keywords = set()

    name_parts = tool["name"].lower().replace("-", "_").split("_")
    keywords.update(p for p in name_parts if len(p) > 2 and p not in _STOPWORDS)
    keywords.add(" ".join(name_parts))

    desc = tool.get("description", "")
    desc_words = re.findall(r'[a-z]+', desc.lower())
    keywords.update(w for w in desc_words if w not in _STOPWORDS and len(w) > 2)

    props = tool.get("parameters", {}).get("properties", {})
    for pname, pdef in props.items():
        pname_parts = pname.lower().replace("-", "_").split("_")
        keywords.update(p for p in pname_parts if len(p) > 2 and p not in _STOPWORDS)
        pdesc = pdef.get("description", "")
        pdesc_words = re.findall(r'[a-z]+', pdesc.lower())
        keywords.update(w for w in pdesc_words if w not in _STOPWORDS and len(w) > 2)

    return keywords


def _score_tool_match(text, tool, tool_keywords):
    text_lower = text.lower()
    score = 0

    name_parts = tool["name"].lower().replace("-", "_").split("_")
    name_phrase = " ".join(name_parts)
    if len(name_parts) > 1 and name_phrase in text_lower:
        score += 10

    for part in name_parts:
        if len(part) > 2 and re.search(r'\b' + re.escape(part) + r'\b', text_lower):
            score += 3

    for kw in tool_keywords:
        if len(kw) > 3 and kw in text_lower:
            score += len(kw)

    return score


def _match_best_tool(text, tools, kw_cache):
    best_name, best_score = None, 0
    for tool in tools:
        s = _score_tool_match(text, tool, kw_cache[tool["name"]])
        if s > best_score:
            best_score = s
            best_name = tool["name"]
    return best_name if best_score > 0 else None


def _build_keyword_cache(tools):
    cache = {t["name"]: _extract_tool_keywords(t) for t in tools}
    for name, kws in cache.items():
        _debug(f"  kw[{name}] = {sorted(kws)}")
    return cache


# ═══════════════════════════════════════════════════════════════════════
#  Intent splitting  (tool-agnostic)
# ═══════════════════════════════════════════════════════════════════════

_ACTION_VERBS = (
    r"(?:check|get|set|send|text|play|remind|find|look|search|create|make|"
    r"add|remove|delete|open|close|start|stop|turn|show|list|call|book|"
    r"schedule|cancel|update|read|write|fetch|run|launch|enable|disable|"
    r"order|buy|reserve|track|locate|navigate|calculate|convert|translate|"
    r"record|save|load|upload|download|forward|reply|share|post|submit|"
    r"lock|unlock|dim|brighten|mute|unmute|pause|resume|skip|rewind|"
    r"wake|message|tell|ask|query|request|ping|alert|notify)"
)

_SPLIT_PATTERNS = [
    rf',\s*and\s+(?={_ACTION_VERBS})',
    rf'\s+and\s+(?={_ACTION_VERBS})',
    rf',\s+(?={_ACTION_VERBS})',
    r'\.\s+',
]


def _split_user_message(text):
    fragments = [text]
    for pattern in _SPLIT_PATTERNS:
        new = []
        for frag in fragments:
            parts = re.split(pattern, frag, flags=re.IGNORECASE)
            new.extend(p.strip() for p in parts if p.strip())
        fragments = new

    if len(fragments) == 1:
        parts = re.split(r'\s+and\s+', text, flags=re.IGNORECASE)
        if len(parts) > 1:
            fragments = [p.strip() for p in parts if p.strip()]

    return fragments


def _detect_multi_intent(text, tools, kw_cache):
    matched = {}
    for tool in tools:
        s = _score_tool_match(text, tool, kw_cache[tool["name"]])
        if s >= 3:
            matched[tool["name"]] = s

    _debug(f"  intent scores: {matched}")
    if len(matched) <= 1:
        _debug(f"  → single intent")
        return None

    fragments = _split_user_message(text)
    _debug(f"  split: {fragments}")

    if len(fragments) < len(matched):
        fragments = [f.strip() for f in re.split(r'\s+and\s+', text, flags=re.IGNORECASE) if f.strip()]
        _debug(f"  re-split (bare 'and'): {fragments}")

    assignments, used = [], set()
    for frag in fragments:
        tn = _match_best_tool(frag, tools, kw_cache)
        if tn and tn not in used:
            assignments.append((frag, tn))
            used.add(tn)

    for tn in matched:
        if tn not in used:
            for frag in fragments:
                if _match_best_tool(frag, tools, kw_cache) == tn:
                    assignments.append((frag, tn))
                    used.add(tn)
                    break

    _debug(f"  assignments: {[(f[:50], t) for f, t in assignments]}")
    return assignments if len(assignments) >= 2 else None


# ═══════════════════════════════════════════════════════════════════════
#  Validation
# ═══════════════════════════════════════════════════════════════════════

def _tools_for(tools, name):
    return [t for t in tools if t["name"] == name]


def _validate_call(call, tools):
    if not isinstance(call, dict) or "name" not in call:
        return False
    for t in tools:
        if t["name"] == call["name"]:
            args = call.get("arguments", {})
            if not isinstance(args, dict):
                return False
            ok = all(r in args for r in t["parameters"].get("required", []))
            if not ok:
                _debug(f"  VALIDATE FAIL: {call['name']} missing required args. got={list(args.keys())} need={t['parameters'].get('required',[])}")
            return ok
    _debug(f"  VALIDATE FAIL: {call.get('name','?')} not in available tools {[t['name'] for t in tools]}")
    return False


# ═══════════════════════════════════════════════════════════════════════
#  Targeted prompts
# ═══════════════════════════════════════════════════════════════════════

def _focused_prompt(tool):
    """When we know the tool, tell the model exactly what to extract."""
    lines = []
    for pname, pdef in tool.get("parameters", {}).get("properties", {}).items():
        lines.append(f"- {pname} ({pdef.get('type','string')}): {pdef.get('description','')}")
    params = "\n".join(lines)
    return (
        f"You are a function-calling assistant. "
        f"Call the function '{tool['name']}'. "
        f"Extract these parameters from the user message:\n{params}\n"
        f"Output only the function call."
    )


def _general_prompt(tools):
    """List available tools explicitly."""
    names = ", ".join(t["name"] for t in tools)
    return (
        f"You are a function-calling assistant. "
        f"Available functions: {names}. "
        f"Call exactly one function. Pick the best match and extract arguments. "
        f"Output only the function call."
    )


# ═══════════════════════════════════════════════════════════════════════
#  HYBRID STRATEGY
# ═══════════════════════════════════════════════════════════════════════

def generate_hybrid(messages, tools, confidence_threshold=0.99):
    user_msg = ""
    for m in messages:
        if m["role"] == "user":
            user_msg = m["content"]

    _debug(f"\n{'='*70}")
    _debug(f"USER: {user_msg}")
    _debug(f"TOOLS: {[t['name'] for t in tools]}")

    kw_cache = _build_keyword_cache(tools)

    # ── Step 1: multi-intent decomposition ────────────────────────────
    multi = _detect_multi_intent(user_msg, tools, kw_cache)

    if multi and len(multi) >= 2:
        _debug(f"▶ MULTI-INTENT: {len(multi)} sub-tasks")
        all_calls, total_time, all_local = [], 0, True

        for i, (sub_text, tool_name) in enumerate(multi):
            sub_tools = _tools_for(tools, tool_name)
            if not sub_tools:
                _debug(f"  [{i}] SKIP: no tool def for '{tool_name}'")
                continue

            prompt = _focused_prompt(sub_tools[0])
            sub_msgs = [{"role": "user", "content": sub_text}]

            _debug(f"  [{i}] sub='{sub_text[:60]}' → tool={tool_name}")
            local = generate_cactus(sub_msgs, sub_tools, system_prompt=prompt)
            total_time += local.get("total_time_ms", 0)

            valid = [c for c in local.get("function_calls", []) if _validate_call(c, sub_tools)]
            _debug(f"  [{i}] valid={len(valid)}")

            if valid:
                all_calls.extend(valid)
                _debug(f"  [{i}] ✓ ACCEPT local")
            else:
                _debug(f"  [{i}] ✗ local failed → cloud fallback for sub-task")
                try:
                    cloud = generate_cloud(sub_msgs, sub_tools)
                    total_time += cloud.get("total_time_ms", 0)
                    cc = cloud.get("function_calls", [])
                    all_calls.extend(cc)
                    all_local = False
                    _debug(f"  [{i}] cloud got {len(cc)} calls")
                except Exception as e:
                    _debug(f"  [{i}] cloud ERROR: {e}")
                    all_local = False

        src = "on-device" if all_local else "on-device (partial cloud)"
        _debug(f"▶ MULTI RESULT: {len(all_calls)} total calls, src={src}, {total_time:.0f}ms")
        _debug(f"  calls: {all_calls}")
        return {
            "function_calls": all_calls,
            "total_time_ms": total_time,
            "source": src,
            "confidence": 1.0 if all_local else 0.5,
        }

    # ── Step 2: single-intent – local with all tools ─────────────────
    _debug(f"▶ SINGLE-INTENT")
    prompt = _general_prompt(tools)
    local = generate_cactus(messages, tools, system_prompt=prompt)
    calls = local.get("function_calls", [])
    conf = local.get("confidence", 0)
    valid = [c for c in calls if _validate_call(c, tools)]

    _debug(f"  step2: valid={len(valid)}/{len(calls)}  conf={conf:.3f}")

    if valid and conf >= 0.1:
        _debug(f"▶ ACCEPT local (step 2)")
        return {
            "function_calls": valid,
            "total_time_ms": local.get("total_time_ms", 0),
            "source": "on-device",
            "confidence": conf,
        }

    # ── Step 3: retry with best-matching tool + focused prompt ───────
    best = _match_best_tool(user_msg, tools, kw_cache)
    _debug(f"  step2 rejected → step3: best keyword match = {best}")

    if best:
        focused = _tools_for(tools, best)
        if focused:
            prompt2 = _focused_prompt(focused[0])
            local2 = generate_cactus(messages, focused, system_prompt=prompt2)
            calls2 = local2.get("function_calls", [])
            valid2 = [c for c in calls2 if _validate_call(c, focused)]
            t2 = local.get("total_time_ms", 0) + local2.get("total_time_ms", 0)

            _debug(f"  step3: valid={len(valid2)}/{len(calls2)}  conf={local2.get('confidence',0):.3f}")

            if valid2:
                _debug(f"▶ ACCEPT focused retry (step 3)")
                return {
                    "function_calls": valid2,
                    "total_time_ms": t2,
                    "source": "on-device",
                    "confidence": local2.get("confidence", 0),
                }

    # ── Step 4: cloud last resort ────────────────────────────────────
    _debug(f"▶ CLOUD FALLBACK (step 4)")
    try:
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback)"
        cloud["local_confidence"] = conf
        cloud["total_time_ms"] += local.get("total_time_ms", 0)
        _debug(f"  cloud result: {cloud['function_calls']}")
        return cloud
    except Exception as e:
        _debug(f"  cloud ERROR: {e}")
        return {
            "function_calls": valid,
            "total_time_ms": local.get("total_time_ms", 0),
            "source": "on-device",
            "confidence": conf,
        }


# ═══════════════════════════════════════════════════════════════════════

def print_result(label, result):
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"],
        },
    }]

    messages = [{"role": "user", "content": "What is the weather in San Francisco?"}]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)