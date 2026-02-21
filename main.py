import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time, re
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types

DEBUG = True


def debug_log(*args):
    if DEBUG:
        print("[DEBUG]", *args)


# -------------------------------
# Local Generation
# -------------------------------

def generate_cactus(messages, tools):
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
        debug_log("Local JSON decode failed")
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0}

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


# -------------------------------
# Cloud Generation
# -------------------------------

def generate_cloud(messages, tools):
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


# -------------------------------
# Hybrid Intelligence
# -------------------------------

def extract_user_text(messages):
    return " ".join(m["content"].lower() for m in messages if m["role"] == "user")


def has_multi_intent(user_text):
    return bool(re.search(r"\band\b|,", user_text))


def score_tool_relevance(user_text, tool):
    score = 0

    name_tokens = tool["name"].lower().split("_")
    desc_tokens = tool["description"].lower().split()
    param_tokens = tool["parameters"]["properties"].keys()

    for token in name_tokens:
        if token in user_text:
            score += 2

    for token in desc_tokens:
        if token in user_text:
            score += 1

    for token in param_tokens:
        if token.lower() in user_text:
            score += 1

    return score


def best_tool_by_text(user_text, tools):
    scored = [(t["name"], score_tool_relevance(user_text, t)) for t in tools]
    debug_log("Tool relevance scores:", scored)

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0] if scored else None


def missing_required_params(call, tools):
    tool_map = {t["name"]: t for t in tools}
    schema = tool_map.get(call["name"])
    if not schema:
        debug_log("Tool not found in schema:", call["name"])
        return True

    required = schema["parameters"].get("required", [])
    args = call.get("arguments", {})

    missing = [r for r in required if r not in args]
    if missing:
        debug_log("Missing params for", call["name"], ":", missing)

    return len(missing) > 0


def dynamic_threshold(user_text, tool_count):
    base = 0.70

    if has_multi_intent(user_text):
        base += 0.15

    if tool_count > 3:
        base += 0.10

    return min(base, 0.95)


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    debug_log("\n--- HYBRID CALL START ---")

    local = generate_cactus(messages, tools)
    user_text = extract_user_text(messages)

    debug_log("User text:", user_text)
    debug_log("Local confidence:", local["confidence"])
    debug_log("Local calls:", local["function_calls"])

    calls = local["function_calls"]

    # ----- Hard rejection rules -----

    if len(calls) == 0:
        debug_log("Rejecting local → no function calls")
        return fallback_to_cloud(local, messages, tools, reason="no_calls")

    if has_multi_intent(user_text) and len(calls) == 1:
        debug_log("Rejecting local → multi-intent underprediction")
        return fallback_to_cloud(local, messages, tools, reason="multi_intent_underprediction")

    expected_tool = best_tool_by_text(user_text, tools)

    if expected_tool and all(c["name"] != expected_tool for c in calls):
        debug_log("Rejecting local → tool mismatch. Expected:", expected_tool)
        return fallback_to_cloud(local, messages, tools, reason="tool_mismatch")

    for call in calls:
        if missing_required_params(call, tools):
            debug_log("Rejecting local → missing required params")
            return fallback_to_cloud(local, messages, tools, reason="missing_params")

    # ----- Confidence decision -----

    threshold = dynamic_threshold(user_text, len(tools))
    debug_log("Dynamic threshold:", threshold)

    if local["confidence"] >= threshold:
        debug_log("Accepting local prediction")
        local["source"] = "on-device"
        return local

    debug_log("Rejecting local → low confidence")
    return fallback_to_cloud(local, messages, tools, reason="low_confidence")


def fallback_to_cloud(local, messages, tools, reason="fallback"):
    debug_log("Falling back to cloud. Reason:", reason)

    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local.get("confidence", 0)
    cloud["total_time_ms"] += local.get("total_time_ms", 0)
    cloud["fallback_reason"] = reason
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