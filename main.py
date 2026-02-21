import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import atexit
import json, os, re, time
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types

DEBUG = True


def _debug(*args):
    if DEBUG:
        print("[DBG]", *args, flush=True)


# =====================================================================
#  GLOBAL MODEL CACHE (large latency win)
# =====================================================================

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        _debug("Initializing FunctionGemma...")
        _MODEL = cactus_init(functiongemma_path)
    return _MODEL


def _destroy_model():
    global _MODEL
    if _MODEL is not None:
        try:
            cactus_destroy(_MODEL)
            _debug("Destroyed FunctionGemma model")
        except Exception as exc:
            _debug("Model destroy error:", str(exc))
        finally:
            _MODEL = None


atexit.register(_destroy_model)


# =====================================================================
#  Generation - ON DEVICE
# =====================================================================

def generate_cactus(messages, tools, system_prompt):
    model = _get_model()
    cactus_tools = [{"type": "function", "function": t} for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": system_prompt}] + messages,
        tools=cactus_tools,
        force_tools=True,
        tool_rag_top_k=2,
        confidence_threshold=0.25,
        max_tokens=128,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    try:
        raw = json.loads(raw_str)
    except Exception:
        _debug("CACTUS JSON FAIL:", raw_str[:240])
        return {"function_calls": [], "confidence": 0, "total_time_ms": 0, "cloud_handoff": False}

    _debug(
        f"cactus -> handoff={raw.get('cloud_handoff')} "
        f"calls={raw.get('function_calls')} "
        f"conf={raw.get('confidence', 0):.3f} "
        f"time={raw.get('total_time_ms', 0):.0f}ms"
    )

    return raw


# =====================================================================
#  Generation - CLOUD
# =====================================================================

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
                        k: types.Schema(type=v["type"].upper())
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]
    start = time.time()

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(tools=gemini_tools),
    )

    total_time_ms = (time.time() - start) * 1000
    calls = []
    for cand in resp.candidates:
        for part in cand.content.parts:
            if part.function_call:
                calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    _debug(f"cloud -> calls={calls} time={total_time_ms:.0f}ms")
    return {"function_calls": calls, "total_time_ms": total_time_ms}


# =====================================================================
#  Prompts
# =====================================================================

def _strict_prompt(tools):
    names = ", ".join(t["name"] for t in tools)
    return (
        "You are a function calling engine.\\n"
        f"Available functions: {names}\\n"
        "Rules:\\n"
        "1. Output ONLY valid JSON\\n"
        "2. Do NOT output natural language\\n"
        "3. You MUST call exactly one function\\n"
        "4. Response format:\\n"
        "{ \"function_calls\": [ {\"name\": \"...\", \"arguments\": {...}} ] }\\n"
    )


# =====================================================================
#  Validation / normalization
# =====================================================================

def _clean_string_arg(value):
    if not isinstance(value, str):
        return value
    value = re.sub(r"\\s+", " ", value.strip())
    value = value.strip("\"'")
    value = re.sub(r"[.!?]+$", "", value)
    return value


def _normalize_calls(calls, tools):
    tool_map = {t["name"]: t for t in tools}
    out = []
    for call in calls:
        name = call.get("name")
        args = call.get("arguments", {})
        td = tool_map.get(name)
        if not td:
            continue
        props = td["parameters"].get("properties", {})
        fixed = {}
        for k, v in args.items():
            if k not in props:
                continue
            typ = props[k].get("type")
            if typ == "integer" and not isinstance(v, int):
                try:
                    fixed[k] = int(float(str(v)))
                except (TypeError, ValueError):
                    fixed[k] = v
            elif typ == "string":
                fixed[k] = _clean_string_arg(str(v))
            else:
                fixed[k] = v
        out.append({"name": name, "arguments": fixed})
    return out


def _validate_call(call, tools):
    if not isinstance(call, dict) or "name" not in call:
        return False
    for t in tools:
        if t["name"] == call["name"]:
            required = t["parameters"].get("required", [])
            args = call.get("arguments", {})
            return all((r in args and args[r] not in (None, "", "unknown")) for r in required)
    return False


# =====================================================================
#  HYBRID STRATEGY
# =====================================================================

def generate_hybrid(messages, tools, confidence_threshold=0.99):
    user_msg = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")

    _debug("\\n" + "=" * 70)
    _debug("USER:", user_msg)
    _debug("TOOLS:", [t["name"] for t in tools])

    prompt = _strict_prompt(tools)

    local = generate_cactus(messages, tools, prompt)

    calls = _normalize_calls(local.get("function_calls", []), tools)
    conf = local.get("confidence", 0)
    handoff = local.get("cloud_handoff", False)

    valid = [c for c in calls if _validate_call(c, tools)]

    _debug(
        "LOCAL CHECK:",
        {
            "raw_calls": len(local.get("function_calls", [])),
            "valid_calls": len(valid),
            "confidence": round(conf, 4),
            "handoff": handoff,
            "threshold_arg": confidence_threshold,
        },
    )

    # Primary acceptance path: prefer local if any valid call exists.
    if valid:
        _debug("ACCEPT LOCAL")
        return {
            "function_calls": valid,
            "total_time_ms": local.get("total_time_ms", 0),
            "source": "on-device",
            "confidence": conf,
        }

    if handoff:
        _debug("CACTUS RECOMMENDS CLOUD")

    _debug("FALLBACK -> CLOUD (reason: no valid local calls)")

    cloud = generate_cloud(messages, tools)

    return {
        "function_calls": cloud["function_calls"],
        "total_time_ms": cloud["total_time_ms"] + local.get("total_time_ms", 0),
        "source": "cloud (fallback)",
        "confidence": conf,
        "local_confidence": conf,
    }


# =====================================================================

def print_result(label, result):
    print(f"\\n=== {label} ===")
    print("Source:", result.get("source"))
    print("Confidence:", round(result.get("confidence", 0), 4))
    print("Time:", round(result.get("total_time_ms", 0), 2), "ms")
    for call in result.get("function_calls", []):
        print(call)


if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"],
        },
    }]

    messages = [{"role": "user", "content": "What is the weather in San Francisco?"}]

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid", hybrid)
