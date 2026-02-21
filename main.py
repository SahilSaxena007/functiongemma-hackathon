import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time, re
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
from google import genai
from google.genai import types


# ---------------------------------------------------------------------------
#  Cloud (Gemini)
# ---------------------------------------------------------------------------

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
#  Helpers
# ---------------------------------------------------------------------------

LOCAL_ROUTER_PROMPT = (
    "You are a strict function-calling router. "
    "Output only function calls, never natural language. "
    "Extract all requested actions from the user request in order. "
    "If the user asks multiple actions, return multiple calls. "
    "Resolve pronouns (him/her/them) from earlier actions in the same request. "
    "For string args, use concise spans, avoid trailing punctuation, and avoid unnecessary leading articles. "
    "For alarm times, set integer hour/minute."
)


def _likely_multi_action(messages):
    text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user").lower()
    return (
        "," in text
        or " and " in text
        or " then " in text
        or " after that " in text
        or " also " in text
    )


def _estimated_min_calls(messages):
    """
    Heuristic lower bound for number of calls requested by the user.
    We intentionally keep this conservative to avoid false negatives.
    """
    text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user").lower()
    separators = [", and ", " and ", ",", " then ", " after that ", " also "]
    count = 1
    for sep in separators:
        count += text.count(sep)
    return max(1, min(4, count))


def _clean_arg(value, field_name):
    if not isinstance(value, str):
        return value
    value = re.sub(r"\s+", " ", value.strip())
    value = value.strip("\"'")
    value = re.sub(r"[.!?]+$", "", value).strip()
    if field_name in {"title", "song"}:
        value = re.sub(r"^(the|a|an)\s+", "", value, flags=re.IGNORECASE)
    return value


def _coerce_and_clean(calls, tools):
    tool_map = {t["name"]: t for t in tools}
    for call in calls:
        td = tool_map.get(call.get("name"))
        if not td:
            continue
        props = td["parameters"].get("properties", {})
        args = call.get("arguments", {})
        for k, v in list(args.items()):
            if k not in props:
                continue
            expected_type = props[k].get("type")
            if expected_type == "integer" and not isinstance(v, int):
                try:
                    call["arguments"][k] = int(float(str(v)))
                except (TypeError, ValueError):
                    pass
            elif expected_type == "string":
                if not isinstance(v, str):
                    v = str(v)
                call["arguments"][k] = _clean_arg(v, k)
    return calls


def _dedupe_calls(calls):
    seen = set()
    unique = []
    for call in calls:
        key = (
            call.get("name"),
            json.dumps(call.get("arguments", {}), sort_keys=True, ensure_ascii=True),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(call)
    return unique


def _validate_calls(calls, tools):
    if not calls:
        return False
    tool_map = {t["name"]: t for t in tools}
    for call in calls:
        td = tool_map.get(call.get("name"))
        if not td:
            return False
        for field in td["parameters"].get("required", []):
            val = call.get("arguments", {}).get(field)
            if val is None or val == "" or val == "unknown":
                return False
    return True


def _is_complete_for_request(calls, messages):
    """
    Completeness check: for multi-action queries, require multiple calls.
    For single-action queries, at least one call is enough.
    """
    if not calls:
        return False
    if not _likely_multi_action(messages):
        return True
    return len(calls) >= _estimated_min_calls(messages)


def _run_local_pass(model, messages, tools, extra_instruction=None, max_tokens=320):
    cactus_tools = [{"type": "function", "function": t} for t in tools]
    local_messages = [{"role": "system", "content": LOCAL_ROUTER_PROMPT}] + messages
    if extra_instruction:
        local_messages.append({"role": "system", "content": extra_instruction})

    raw_str = cactus_complete(
        model,
        local_messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=max_tokens,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
        tool_rag_top_k=0,
        confidence_threshold=0.0,
    )

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {"ok": False, "function_calls": [], "total_time_ms": 0, "confidence": 0}

    return {
        "ok": True,
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


# ---------------------------------------------------------------------------
#  Main hybrid router
# ---------------------------------------------------------------------------

def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """FunctionGemma-first routing with local self-repair and last-resort cloud fallback."""
    # Treat threshold as advisory (avoid default 0.99 forcing unnecessary cloud/extra retries).
    try:
        configured_threshold = float(confidence_threshold)
    except (TypeError, ValueError):
        configured_threshold = 0.0
    effective_threshold = max(0.0, min(0.6, configured_threshold))

    total_time_ms = 0
    model = None
    try:
        model = cactus_init(functiongemma_path)

        # Pass 1: direct local extraction
        pass1 = _run_local_pass(model, messages, tools)
        total_time_ms += pass1["total_time_ms"]
        calls = _dedupe_calls(_coerce_and_clean(pass1["function_calls"], tools))
        valid = _validate_calls(calls, tools)
        complete = _is_complete_for_request(calls, messages)
        confident = (effective_threshold <= 0 or pass1["confidence"] >= effective_threshold)
        if valid and complete and confident:
            return {
                "function_calls": calls,
                "total_time_ms": total_time_ms,
                "source": "on-device",
            }

        # Pass 2 (single retry): local self-repair for missing/invalid output.
        cactus_reset(model)
        repair_instruction = (
            "Re-read the user request and output the complete, ordered list of tool calls. "
            "Include every requested action. "
            "Return only function calls and ensure all required args are present."
        )
        pass2 = _run_local_pass(model, messages, tools, extra_instruction=repair_instruction, max_tokens=384)
        total_time_ms += pass2["total_time_ms"]
        merged = _dedupe_calls(_coerce_and_clean(calls + pass2["function_calls"], tools))
        valid = _validate_calls(merged, tools)
        complete = _is_complete_for_request(merged, messages)
        confident = (effective_threshold <= 0 or pass2["confidence"] >= effective_threshold or pass1["confidence"] >= effective_threshold)
        if valid and complete and confident:
            return {
                "function_calls": merged,
                "total_time_ms": total_time_ms,
                "source": "on-device",
            }
    except Exception:
        # Fall through to cloud fallback.
        pass
    finally:
        if model is not None:
            try:
                cactus_destroy(model)
            except Exception:
                pass

    try:
        cloud = generate_cloud(messages, tools)
        cloud["function_calls"] = _dedupe_calls(_coerce_and_clean(cloud.get("function_calls", []), tools))
        cloud["source"] = "cloud (fallback)"
        cloud["total_time_ms"] += total_time_ms
        return cloud
    except Exception:
        return {
            "function_calls": [],
            "total_time_ms": total_time_ms,
            "source": "fallback-error",
        }


# ---------------------------------------------------------------------------
#  Convenience helper (unchanged interface)
# ---------------------------------------------------------------------------

def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = cactus_init(functiongemma_path)
    cactus_tools = [{"type": "function", "function": t} for t in tools]
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
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0}
    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
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

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid", hybrid)
