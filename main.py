
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time
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


def _classify_query(messages, tools):
    """Pre-route before calling any model. Returns 'easy_local', 'medium_local', or 'hard_cloud'."""
    content = messages[-1]["content"].lower()
    num_tools = len(tools)

    multi_call_signals = [
        " and ", " also ", " plus ", " as well", "both",
        ", and", "then ", "after that",
    ]
    signal_count = sum(1 for s in multi_call_signals if s in content)

    action_keywords = [
        "set", "send", "check", "play", "remind", "find",
        "look up", "text", "wake", "search", "get", "call",
        "schedule", "add", "start", "stop", "cancel",
    ]
    action_count = sum(1 for kw in action_keywords if kw in content)

    if action_count >= 2 and signal_count >= 1:
        return "hard_cloud"
    if signal_count >= 2:
        return "hard_cloud"
    if num_tools >= 4:
        return "medium_local"
    return "easy_local"


def _validate_local(result, tools, query_type):
    """Returns True if local result should be trusted, False to escalate to cloud."""
    calls = result.get("function_calls", [])
    confidence = result.get("confidence", 0)
    decode_tps = result.get("decode_tps", 0)

    if not calls:
        return False
    if decode_tps == 0.0:
        return False
    if confidence < 0.55:
        return False

    tool_map = {t["name"]: t for t in tools}
    for call in calls:
        tool_def = tool_map.get(call["name"])
        if not tool_def:
            return False
        required = tool_def["parameters"].get("required", [])
        args = call.get("arguments", {})
        for field in required:
            if field not in args:
                return False
            val = args[field]
            if val is None or val == "" or val == "unknown":
                return False

    if query_type == "medium_local" and confidence < 0.70:
        return False

    return True


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    Three-layer hybrid routing strategy:
      Layer 1 — Pre-route: classify query complexity before calling anything.
      Layer 2 — Tool RAG: pass tool_rag_top_k=2 to reduce noise for local.
      Layer 3 — Output validation: validate local result before trusting it.
    """
    query_type = _classify_query(messages, tools)

    # Hard (multi-action) queries go straight to cloud — skip local entirely
    if query_type == "hard_cloud":
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback)"
        return cloud

    # Try local with Tool RAG enabled
    model = cactus_init(functiongemma_path)
    cactus_tools = [{"type": "function", "function": t} for t in tools]
    try:
        raw_str = cactus_complete(
            model,
            [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
            tools=cactus_tools,
            force_tools=True,
            max_tokens=256,
            stop_sequences=["<|im_end|>", "<end_of_turn>"],
            tool_rag_top_k=2,
            confidence_threshold=0.99,
        )
    finally:
        cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        raw = {"function_calls": [], "total_time_ms": 0, "confidence": 0, "decode_tps": 0}

    local = {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
        "decode_tps": raw.get("decode_tps", 0),
    }

    if _validate_local(local, tools, query_type):
        local["source"] = "on-device"
        return local

    # Local failed validation — escalate to cloud
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local["confidence"]
    cloud["total_time_ms"] += local["total_time_ms"]
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
