
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time, re
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


LOCAL_ROUTER_PROMPT = (
    "You are a strict tool-calling router. "
    "Output only tool calls and no natural language. "
    "Extract every requested action from the user request in order. "
    "If there are multiple actions, output multiple calls. "
    "Resolve pronouns like him/her/them using earlier actions in the same request."
)


def _likely_multi_action(messages):
    text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user").lower()
    return any(sep in text for sep in [",", " and ", " then ", " after that ", " also "])


def _estimated_min_calls(messages):
    text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user").lower()
    count = 1
    for sep in [", and ", " and ", ",", " then ", " after that ", " also "]:
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
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0}
    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    Local-first router:
    1) Local extraction pass
    2) Local repair pass (single retry if invalid/incomplete)
    3) Cloud fallback as last resort
    """
    total_time_ms = 0
    model = None
    try:
        model = cactus_init(functiongemma_path)

        pass1 = _run_local_pass(model, messages, tools)
        total_time_ms += pass1["total_time_ms"]
        calls = _dedupe_calls(_coerce_and_clean(pass1["function_calls"], tools))
        if _validate_calls(calls, tools) and _is_complete_for_request(calls, messages):
            return {
                "function_calls": calls,
                "total_time_ms": total_time_ms,
                "source": "on-device",
                "confidence": pass1["confidence"],
            }

        cactus_reset(model)
        repair_instruction = (
            "Re-read the request and return the complete ordered list of tool calls. "
            "Include all requested actions with required arguments. "
            "Output only tool calls."
        )
        pass2 = _run_local_pass(model, messages, tools, extra_instruction=repair_instruction, max_tokens=384)
        total_time_ms += pass2["total_time_ms"]
        merged = _dedupe_calls(_coerce_and_clean(calls + pass2["function_calls"], tools))
        if _validate_calls(merged, tools) and _is_complete_for_request(merged, messages):
            return {
                "function_calls": merged,
                "total_time_ms": total_time_ms,
                "source": "on-device",
                "confidence": max(pass1["confidence"], pass2["confidence"]),
            }
    except Exception:
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
