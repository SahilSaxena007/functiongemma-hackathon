import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, re, time
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = cactus_init(functiongemma_path)

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


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

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


# ---------------------------------------------------------------------------
# Dynamic tool-matching helpers (zero hardcoded tool names/keywords)
# ---------------------------------------------------------------------------

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

    # From name: "get_weather" -> {"get", "weather", "get weather"}
    name_parts = tool["name"].lower().replace("-", "_").split("_")
    keywords.update(p for p in name_parts if len(p) > 2 and p not in _STOPWORDS)
    keywords.add(" ".join(name_parts))

    # From description
    desc = tool.get("description", "")
    desc_words = re.findall(r'[a-z]+', desc.lower())
    keywords.update(w for w in desc_words if w not in _STOPWORDS and len(w) > 2)

    # From parameter names and descriptions
    props = tool.get("parameters", {}).get("properties", {})
    for pname, pdef in props.items():
        pname_parts = pname.lower().replace("-", "_").split("_")
        keywords.update(p for p in pname_parts if len(p) > 2 and p not in _STOPWORDS)
        pdesc = pdef.get("description", "")
        pdesc_words = re.findall(r'[a-z]+', pdesc.lower())
        keywords.update(w for w in pdesc_words if w not in _STOPWORDS and len(w) > 2)

    return keywords


def _score_tool_match(text, tool, tool_keywords):
    """Score how well a text fragment matches a tool. Higher = better."""
    text_lower = text.lower()
    score = 0

    # Multi-word phrase from tool name (strongest signal)
    name_parts = tool["name"].lower().replace("-", "_").split("_")
    name_phrase = " ".join(name_parts)
    if len(name_parts) > 1 and name_phrase in text_lower:
        score += 10

    # Individual name parts
    for part in name_parts:
        if len(part) > 2 and re.search(r'\b' + re.escape(part) + r'\b', text_lower):
            score += 3

    # Keywords from description/params
    for kw in tool_keywords:
        if len(kw) > 3 and kw in text_lower:
            score += len(kw)

    return score


def _match_best_tool(text, tools, tool_kw_cache):
    """Find best-matching tool for a text fragment. Returns tool name or None."""
    best_name = None
    best_score = 0
    for tool in tools:
        s = _score_tool_match(text, tool, tool_kw_cache[tool["name"]])
        if s > best_score:
            best_score = s
            best_name = tool["name"]
    return best_name if best_score > 0 else None


def _build_keyword_cache(tools):
    return {t["name"]: _extract_tool_keywords(t) for t in tools}


# ---------------------------------------------------------------------------
# Intent splitting (fully tool-agnostic)
# ---------------------------------------------------------------------------

# Broad set of action verbs that signal a new intent boundary
_ACTION_VERBS = (
    r"(?:check|get|set|send|text|play|remind|find|look|search|create|make|"
    r"add|remove|delete|open|close|start|stop|turn|show|list|call|book|"
    r"schedule|cancel|update|read|write|fetch|run|launch|enable|disable|"
    r"order|buy|reserve|track|locate|navigate|calculate|convert|translate|"
    r"record|save|load|upload|download|forward|reply|share|post|submit|"
    r"lock|unlock|dim|brighten|mute|unmute|pause|resume|skip|rewind)"
)

SPLIT_PATTERNS = [
    rf',\s*and\s+(?={_ACTION_VERBS})',  # ", and <verb>"
    rf'\s+and\s+(?={_ACTION_VERBS})',   # " and <verb>"
    rf',\s+(?={_ACTION_VERBS})',        # ", <verb>"
    r'\.\s+',                            # sentence boundary
]


def _split_user_message(text):
    """Split a multi-intent user message into sub-request fragments."""
    fragments = [text]
    for pattern in SPLIT_PATTERNS:
        new_frags = []
        for frag in fragments:
            parts = re.split(pattern, frag, flags=re.IGNORECASE)
            new_frags.extend([p.strip() for p in parts if p.strip()])
        fragments = new_frags

    # Fallback: bare " and " split
    if len(fragments) == 1:
        parts = re.split(r'\s+and\s+', text, flags=re.IGNORECASE)
        if len(parts) > 1:
            fragments = [p.strip() for p in parts if p.strip()]

    return fragments


def _detect_multi_intent(text, tools, tool_kw_cache):
    """
    Detect if message needs multiple tool calls.
    Returns list of (sub_text, tool_name) or None.
    """
    # How many distinct tools are signaled?
    matched_full = set()
    for tool in tools:
        s = _score_tool_match(text, tool, tool_kw_cache[tool["name"]])
        if s >= 3:
            matched_full.add(tool["name"])

    if len(matched_full) <= 1:
        return None

    fragments = _split_user_message(text)

    if len(fragments) < len(matched_full):
        fragments = re.split(r'\s+and\s+', text, flags=re.IGNORECASE)
        fragments = [f.strip() for f in fragments if f.strip()]

    # Assign fragments to tools
    assignments = []
    used = set()
    for frag in fragments:
        tn = _match_best_tool(frag, tools, tool_kw_cache)
        if tn and tn not in used:
            assignments.append((frag, tn))
            used.add(tn)

    # Catch any matched tools we missed
    for tn in matched_full:
        if tn not in used:
            for frag in fragments:
                if _match_best_tool(frag, tools, tool_kw_cache) == tn:
                    assignments.append((frag, tn))
                    used.add(tn)
                    break

    return assignments if len(assignments) >= 2 else None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _get_tools_for_name(tools, name):
    return [t for t in tools if t["name"] == name]


def _validate_call(call, tools):
    """Check if a function call is structurally valid."""
    if not call or not isinstance(call, dict) or "name" not in call:
        return False
    tool_names = {t["name"] for t in tools}
    if call["name"] not in tool_names:
        return False
    for t in tools:
        if t["name"] == call["name"]:
            required = t["parameters"].get("required", [])
            args = call.get("arguments", {})
            if not isinstance(args, dict):
                return False
            return all(r in args for r in required)
    return False


# ---------------------------------------------------------------------------
# Main hybrid strategy
# ---------------------------------------------------------------------------

def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    Optimized hybrid: decompose multi-intent, focus tools, validate locally,
    cloud only as last resort. All tool matching is dynamic from definitions.
    """
    user_msg = ""
    for m in messages:
        if m["role"] == "user":
            user_msg = m["content"]

    tool_kw_cache = _build_keyword_cache(tools)

    # --- Step 1: Multi-intent decomposition ---
    multi = _detect_multi_intent(user_msg, tools, tool_kw_cache)

    if multi and len(multi) >= 2:
        all_calls = []
        total_time = 0
        all_local = True

        for sub_text, tool_name in multi:
            sub_messages = [{"role": "user", "content": sub_text}]
            sub_tools = _get_tools_for_name(tools, tool_name)
            if not sub_tools:
                continue

            local = generate_cactus(sub_messages, sub_tools)
            total_time += local.get("total_time_ms", 0)

            calls = local.get("function_calls", [])
            valid = [c for c in calls if _validate_call(c, sub_tools)]

            if valid:
                all_calls.extend(valid)
            else:
                try:
                    cloud = generate_cloud(sub_messages, sub_tools)
                    total_time += cloud.get("total_time_ms", 0)
                    cc = cloud.get("function_calls", [])
                    if cc:
                        all_calls.extend(cc)
                    all_local = False
                except Exception:
                    all_local = False

        return {
            "function_calls": all_calls,
            "total_time_ms": total_time,
            "source": "on-device" if all_local else "on-device (partial cloud)",
            "confidence": 1.0 if all_local else 0.5,
        }

    # --- Step 2: Single-intent, try local with all tools ---
    local = generate_cactus(messages, tools)
    calls = local.get("function_calls", [])
    confidence = local.get("confidence", 0)
    valid_calls = [c for c in calls if _validate_call(c, tools)]

    if valid_calls and confidence >= 0.1:
        return {
            "function_calls": valid_calls,
            "total_time_ms": local.get("total_time_ms", 0),
            "source": "on-device",
            "confidence": confidence,
        }

    # --- Step 3: Retry with only the best-matching tool ---
    best_tool = _match_best_tool(user_msg, tools, tool_kw_cache)
    if best_tool:
        focused = _get_tools_for_name(tools, best_tool)
        if focused:
            local2 = generate_cactus(messages, focused)
            calls2 = local2.get("function_calls", [])
            valid2 = [c for c in calls2 if _validate_call(c, focused)]
            t2 = local.get("total_time_ms", 0) + local2.get("total_time_ms", 0)

            if valid2:
                return {
                    "function_calls": valid2,
                    "total_time_ms": t2,
                    "source": "on-device",
                    "confidence": local2.get("confidence", 0),
                }

    # --- Step 4: Cloud fallback ---
    try:
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback)"
        cloud["local_confidence"] = confidence
        cloud["total_time_ms"] += local.get("total_time_ms", 0)
        return cloud
    except Exception:
        return {
            "function_calls": valid_calls,
            "total_time_ms": local.get("total_time_ms", 0),
            "source": "on-device",
            "confidence": confidence,
        }


def print_result(label, result):
    """Pretty-print a generation result."""
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


############## Example usage ##############

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)