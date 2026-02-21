
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time, re
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
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
        model="gemini-2.5-flash",
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
#  Helpers for generate_hybrid
# ---------------------------------------------------------------------------

def _decompose_query(content):
    """Split a multi-action query into individual single-action sub-queries."""
    content = content.strip().rstrip('.')

    # First try comma-separated clauses: "do X, do Y, and do Z"
    parts = re.split(r',\s+and\s+|,\s+', content)

    if len(parts) == 1:
        # Try " and " / " then " / " after that " but only before action verbs
        verbs = (r'(?:set|send|get|check|play|remind|find|look|text|search|'
                 r'wake|start|stop|cancel|create|make|open|close|turn|call|'
                 r'add|remove|delete|update|book|order|schedule|tell|ask|show)')
        parts = re.split(
            r'(?:\s+and\s+|\s+then\s+|\s+after\s+that\s+|\s+also\s+)(?=' + verbs + r')',
            content,
            flags=re.IGNORECASE,
        )

    return [p.strip() for p in parts if p.strip()]


def _resolve_pronouns(text, name):
    """Replace him/her/them with a known name from prior sub-query results."""
    text = re.sub(r'\bhim\b', name, text, count=1, flags=re.IGNORECASE)
    text = re.sub(r'\bher\b', name, text, count=1, flags=re.IGNORECASE)
    text = re.sub(r'\bthem\b', name, text, count=1, flags=re.IGNORECASE)
    return text


def _extract_name(call):
    """Extract a proper-noun name from a function call's arguments."""
    args = call.get("arguments", {})
    for key in ["query", "recipient", "name", "contact"]:
        val = args.get(key)
        if isinstance(val, str) and val and val[0].isupper():
            return val
    return None


def _coerce_types(calls, tools):
    """Convert argument values to match the tool schema types (e.g. str→int)."""
    tool_map = {t["name"]: t for t in tools}
    for call in calls:
        td = tool_map.get(call["name"])
        if not td:
            continue
        props = td["parameters"].get("properties", {})
        for k, v in list(call.get("arguments", {}).items()):
            if k not in props:
                continue
            expected_type = props[k].get("type")
            if expected_type == "integer" and not isinstance(v, int):
                try:
                    call["arguments"][k] = int(float(str(v)))
                except (ValueError, TypeError):
                    pass
            elif expected_type == "string" and not isinstance(v, str):
                call["arguments"][k] = str(v)
    return calls


def _validate_calls(calls, tools):
    """Check every call references a real tool and has all required fields filled."""
    if not calls:
        return False
    tool_map = {t["name"]: t for t in tools}
    for call in calls:
        td = tool_map.get(call["name"])
        if not td:
            return False
        for field in td["parameters"].get("required", []):
            val = call.get("arguments", {}).get(field)
            if val is None or val == "" or val == "unknown":
                return False
    return True


# ---------------------------------------------------------------------------
#  Main hybrid router
# ---------------------------------------------------------------------------

def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    Query-decomposition hybrid routing strategy.

    Core idea: decompose multi-action queries into single-action sub-queries,
    run each through the local model individually (reusing the model handle),
    and only fall back to cloud if local fails to produce valid calls.

    Layers:
      1. Decompose — split "do X, do Y, and do Z" into ["do X", "do Y", "do Z"]
      2. Pronoun resolution — replace "send him" with "send Tom" using prior results
      3. Local per sub-query — confidence_threshold=0.0, tool_rag_top_k=2, force_tools
      4. Type coercion — "10" → 10 for integer params
      5. Validation — all calls must reference real tools with required fields filled
      6. Cloud fallback — only if local produces no/invalid output
    """
    content = messages[-1]["content"]
    sub_queries = _decompose_query(content)

    # Init model once, reuse across sub-queries
    model = cactus_init(functiongemma_path)
    cactus_tools = [{"type": "function", "function": t} for t in tools]

    all_calls = []
    total_time = 0
    local_failed = False
    last_name = None

    try:
        for i, sq in enumerate(sub_queries):
            if i > 0:
                cactus_reset(model)

            # Resolve pronouns ("send him" → "send Tom")
            if last_name and re.search(r'\b(him|her|them)\b', sq, re.IGNORECASE):
                sq = _resolve_pronouns(sq, last_name)

            raw_str = cactus_complete(
                model,
                [{"role": "user", "content": sq}],
                tools=cactus_tools,
                force_tools=True,
                max_tokens=256,
                stop_sequences=["<|im_end|>", "<end_of_turn>"],
                tool_rag_top_k=2,
                confidence_threshold=0.0,
            )

            try:
                raw = json.loads(raw_str)
            except json.JSONDecodeError:
                local_failed = True
                break

            total_time += raw.get("total_time_ms", 0)
            calls = raw.get("function_calls", [])

            if calls:
                call = calls[0]  # one call per sub-query
                all_calls.append(call)
                # Track name for pronoun resolution in subsequent sub-queries
                name = _extract_name(call)
                if name:
                    last_name = name
            else:
                local_failed = True
                break
    finally:
        cactus_destroy(model)

    # Post-process and validate local results
    if not local_failed:
        all_calls = _coerce_types(all_calls, tools)
        if _validate_calls(all_calls, tools):
            return {
                "function_calls": all_calls,
                "total_time_ms": total_time,
                "source": "on-device",
            }

    # Cloud fallback
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["total_time_ms"] += total_time
    return cloud


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
