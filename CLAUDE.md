# FunctionGemma Hackathon — Master Reference

> For Claude Code / Codex / any AI agent working on this project.
> Last updated: Day-of hackathon. Do not modify this file during hacking — it is a reference only.

---

## 1. What This Project Actually Is

A Python-only competition. You modify **one function** in `main.py` called `generate_hybrid`.
The function receives a user message and a list of tools. It must return the correct function call(s).
It can use either the **local FunctionGemma model** (via Cactus) or **Gemini cloud API** as fallback.

You are scored on three things simultaneously:

- **F1 correctness** (50% of score) — did you call the right function with the right arguments?
- **Speed** (25%) — faster is better, capped at 500ms baseline
- **On-device ratio** (25%) — more local calls = better score

Difficulty weights: easy = 20%, medium = 30%, **hard = 50%**. Win or lose on hard cases.

---

## 2. File Structure

```
functiongemma-hackathon/          ← the repo you cloned and work in
│
├── main.py                       ← THE ONLY FILE YOU EDIT
├── benchmark.py                  ← run this to test your score (never edit)
├── submit.py                     ← run this to submit to leaderboard (never edit)
├── INSTRUCTIONS.md               ← this file
│
└── cactus/                       ← cloned separately: git clone cactus-compute/cactus
    └── python/src/               ← Cactus Python bindings (auto-added to sys.path)
        └── cactus/
    └── weights/
        └── functiongemma-270m-it/  ← downloaded model weights
```

**Critical rule:** `benchmark.py` imports `generate_hybrid` directly from `main.py`.
The function signature must never change:

```python
def generate_hybrid(messages, tools):
    ...
    return {
        "function_calls": [...],
        "total_time_ms": ...,
        "source": "on-device" | "cloud (fallback)"
    }
```

---

## 3. The Scoring Formula (Read This Carefully)

From `benchmark.py` — exact formula:

```python
level_score = (0.50 * avg_f1) + (0.25 * time_score) + (0.25 * on_device_ratio)
time_score = max(0, 1 - avg_time / 500)  # 500ms = baseline, under = full marks
total = 0.20 * easy_score + 0.30 * medium_score + 0.50 * hard_score
```

**Implication:** Hard cases dominate. A perfect easy/medium score with bad hard performance loses.
Hard cases are ALL multi-call: the model must return 2-3 function calls from one message.

**Implication:** Time under 500ms gets full time marks. Cloud calls are typically 800-1500ms.
So cloud calls hurt both speed AND on-device ratio. Use local whenever it's correct.

**Implication:** F1 scoring is partial credit. Getting the right function name but wrong args
scores 0 for that call. Getting 2/3 calls right scores ~0.67 F1. Every call matters.

---

## 4. The Benchmark Cases (Know These Cold)

### Easy (10 cases) — 1 tool, 1 call required

All mobile assistant style: weather, alarm, message, timer, reminder, music, search contacts.
Local FunctionGemma handles these well. Target: 100% local, F1 ≥ 0.95.

### Medium (10 cases) — 2-5 tools provided, must pick the right 1

Same action types but model must discriminate. E.g., "Set an alarm" when alarm + weather + music tools exist.
Local model struggles when tools are similar. Target: mostly local, F1 ≥ 0.85.

### Hard (10 cases) — 2-5 tools, must return 2-3 calls

Examples:

- "Send Bob hi AND get weather in London" → 2 calls
- "Set alarm 6:45, remind me medicine 7am" → 2 calls
- "Text Emma, check Chicago weather, alarm 5am" → 3 calls
- "Search Jake, send him let's meet, check Seattle weather" → 3 calls

FunctionGemma base accuracy on multi-call is low. Cloud likely needed for hard cases.
Target: F1 ≥ 0.80, accept cloud fallback for hard.

---

## 5. The Cactus API Reference

### Init / Destroy

```python
from cactus import cactus_init, cactus_complete, cactus_destroy

model = cactus_init("cactus/weights/functiongemma-270m-it")
# ... use model ...
cactus_destroy(model)   # ALWAYS call this or you leak memory
```

### Complete

```python
raw_str = cactus_complete(
    model,
    messages,                    # list of {"role": ..., "content": ...}
    tools=cactus_tools,          # list of {"type": "function", "function": {...}}
    force_tools=True,            # constrain output to tool call format — ALWAYS use this
    max_tokens=256,
    stop_sequences=["<|im_end|>", "<end_of_turn>"],
    tool_rag_top_k=2,            # only show top-2 most relevant tools — HUGE accuracy boost
    confidence_threshold=0.99,   # set high so cloud_handoff fires rarely (we handle routing ourselves)
)
raw = json.loads(raw_str)
```

### Response Fields (all always present)

```python
raw["success"]              # bool
raw["cloud_handoff"]        # bool — model's own recommendation (don't rely on this alone)
raw["response"]             # str — text response (usually empty for tool calls)
raw["function_calls"]       # list of {"name": ..., "arguments": {...}}
raw["confidence"]           # float 0-1 — model's self-assessed certainty
raw["time_to_first_token_ms"]
raw["total_time_ms"]
raw["prefill_tps"]          # tokens/sec prefill
raw["decode_tps"]           # tokens/sec decode — if 0.0, model stopped early (bad sign)
raw["prefill_tokens"]
raw["decode_tokens"]
raw["total_tokens"]
raw["ram_usage_mb"]
```

### Tool Format for Cactus

```python
# Wrap each tool dict in {"type": "function", "function": tool}
cactus_tools = [{"type": "function", "function": t} for t in tools]
```

### Reset Between Calls

```python
# If reusing a model handle across multiple calls, reset KV cache between them
cactus_reset(model)
```

### Whisper (for voice demo — Rubric 3)

```python
whisper = cactus_init("cactus/weights/whisper-small")
prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
result = json.loads(cactus_transcribe(whisper, "audio.wav", prompt=prompt))
text = result["response"]
cactus_destroy(whisper)
```

---

## 6. The Gemini API Reference

### Setup

```python
import os
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
```

### Tool Format for Gemini (DIFFERENT from Cactus — uppercase types)

```python
gemini_tools = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name=t["name"],
            description=t["description"],
            parameters=types.Schema(
                type="OBJECT",                           # UPPERCASE
                properties={
                    k: types.Schema(
                        type=v["type"].upper(),          # string→STRING, integer→INTEGER
                        description=v.get("description", "")
                    )
                    for k, v in t["parameters"]["properties"].items()
                },
                required=t["parameters"].get("required", []),
            ),
        )
        for t in tools
    ])
]
```

### Call

```python
import time

start = time.time()
response = client.models.generate_content(
    model="gemini-2.0-flash-001",    # use this exact string
    contents=[m["content"] for m in messages if m["role"] == "user"],
    config=types.GenerateContentConfig(tools=gemini_tools),
)
total_time_ms = (time.time() - start) * 1000
```

### Extract Function Calls

```python
function_calls = []
for candidate in response.candidates:
    for part in candidate.content.parts:
        if part.function_call:
            function_calls.append({
                "name": part.function_call.name,
                "arguments": dict(part.function_call.args),
            })
```

---

## 7. The Routing Strategy — What to Build

### Baseline (already in main.py — do NOT keep this)

```python
# Baseline: just uses confidence threshold. Score ~40-50%.
if local["confidence"] >= 0.99:
    return local
else:
    return cloud
```

### Strategy Layer 1 — Pre-routing (before calling local)

Classify the query BEFORE calling anything. This saves latency on cases local can't handle.

```python
def classify_query(messages, tools):
    """
    Returns: "easy_local", "medium_local", or "hard_cloud"
    """
    content = messages[-1]["content"].lower()
    num_tools = len(tools)

    # Hard signals — go straight to cloud
    multi_call_signals = [
        " and ", " also ", " plus ", "as well", "both",
        "check the weather", "send a message",   # often paired
    ]
    signal_count = sum(1 for s in multi_call_signals if s in content)

    # Count how many distinct actions are mentioned
    action_keywords = ["set", "send", "check", "play", "remind", "find", "look up", "text", "wake"]
    action_count = sum(1 for kw in action_keywords if kw in content)

    if action_count >= 2 or signal_count >= 2:
        return "hard_cloud"   # almost certainly multi-call → go to cloud

    if num_tools >= 4:
        return "medium_local"  # try local but validate carefully

    return "easy_local"  # try local, trust if confidence is decent
```

### Strategy Layer 2 — Output Validation (after local call)

Don't trust local just because it returned something. Validate the output quality.

```python
def validate_local_result(result, tools, query_type):
    """
    Returns True if local result should be trusted, False if we should escalate.
    """
    calls = result.get("function_calls", [])
    confidence = result.get("confidence", 0)
    decode_tps = result.get("decode_tps", 0)

    # No calls produced at all
    if not calls:
        return False

    # Model stopped generating almost immediately — likely confused
    if decode_tps == 0.0:
        return False

    # Very low confidence
    if confidence < 0.55:
        return False

    # Validate each call has all required fields filled
    tool_map = {t["name"]: t for t in tools}
    for call in calls:
        tool_def = tool_map.get(call["name"])
        if not tool_def:
            return False   # called a tool that doesn't exist

        required = tool_def["parameters"].get("required", [])
        args = call.get("arguments", {})

        for field in required:
            if field not in args:
                return False   # missing required field
            val = args[field]
            if val is None or val == "" or val == "unknown":
                return False   # empty or placeholder value

    # For medium queries with many tools, require higher confidence
    if query_type == "medium_local" and confidence < 0.70:
        return False

    return True
```

### Strategy Layer 3 — Tool RAG (reduce noise for local)

Always pass `tool_rag_top_k=2` to Cactus. This filters the tools to the 2 most relevant ones
before the model sees them. Dramatically improves accuracy when 4-5 tools are present.

```python
raw_str = cactus_complete(
    model, messages,
    tools=cactus_tools,
    tool_rag_top_k=2,    # ← this is the magic parameter
    force_tools=True,
    ...
)
```

### Full generate_hybrid Implementation

```python
def generate_hybrid(messages, tools):
    # Layer 1: Pre-route based on query complexity
    query_type = classify_query(messages, tools)

    if query_type == "hard_cloud":
        # Multi-action query — skip local, go straight to cloud
        result = generate_cloud(messages, tools)
        result["source"] = "cloud (fallback)"
        return result

    # Try local
    local = generate_cactus(messages, tools)

    # Layer 2: Validate local output
    if validate_local_result(local, tools, query_type):
        local["source"] = "on-device"
        return local

    # Local failed validation — escalate to cloud
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local.get("confidence", 0)
    cloud["total_time_ms"] += local.get("total_time_ms", 0)
    return cloud
```

---

## 8. Work Division (2-Person Team)

### Person A — Mac owner (runs all code)

Responsibilities:

1. Complete Mac setup (Steps 1-10 from README)
2. Run `python benchmark.py` after every change — report scores to Person B
3. Run `python submit.py` once per hour when score improves
4. Implement `generate_cactus` tweaks (prompting, tool_rag_top_k tuning)
5. If time permits: build voice demo using `cactus_transcribe`
6. Handle all terminal work

### Person B — Windows (Sahil — writes the logic)

Responsibilities:

1. Write and iterate `classify_query` function — tune the signals
2. Write and iterate `validate_local_result` function — tune the thresholds
3. Write the full `generate_hybrid` function body
4. Monitor leaderboard at `https://cactusevals.ngrok.app`
5. Analyze failed benchmark cases — identify patterns in what local gets wrong
6. If time permits: write the voice demo script

### Coordination method

Use one shared file. Person B writes code in VS Code, Person A pulls via Git or shared folder.
Simplest workflow: Person B writes → pastes new `generate_hybrid` into shared chat (WhatsApp/Slack) → Person A replaces in their `main.py` → runs benchmark → reports score back.

**Do NOT use Git branches.** One person, one file, one `main.py`. Too much coordination overhead.

### Time schedule

- 10:00 AM — Person A starts setup. Person B reads benchmark.py carefully, notes all hard cases.
- 11:00 AM — Setup should be complete. First baseline run. Identify weak spots.
- 11:30 AM — Person B has `classify_query` written. Person A tests it.
- 12:30 PM — Person B has `validate_local_result` written. Full strategy tested.
- 2:00 PM — Iterate on thresholds based on benchmark feedback.
- 3:30 PM — Submit best version. Begin voice demo if score is competitive.
- 5:00 PM — Final submit. Prepare demo explanation.

---

## 9. Setup Commands (Mac — Person A runs these)

```bash
# Step 1: Clone repos
git clone https://github.com/cactus-compute/functiongemma-hackathon
git clone https://github.com/cactus-compute/cactus
cd functiongemma-hackathon

# Step 2: Setup Cactus
cd cactus && source ./setup && cd ..
# If new terminal: cd cactus && source ./setup && cd ..

# Step 3: Build Python bindings
cactus build --python

# Step 4: Download model
cactus download google/functiongemma-270m-it --reconvert

# Step 5: Auth
# Get key from https://cactuscompute.com/dashboard/api-keys
cactus auth   # paste key when prompted

# Step 6: Install Gemini SDK
pip install google-genai

# Step 7: Set Gemini key
export GEMINI_API_KEY="your-key-here"
# Get key from https://aistudio.google.com/api-keys

# Step 8: Test everything works
python main.py

# Step 9: Run baseline benchmark
python benchmark.py
```

If `source ./setup` says "command not found": try `. ./setup` instead.
If `cactus build --python` fails: run `xcode-select --install` first.

---

## 10. Testing Workflow

```bash
# After every change to main.py:
python benchmark.py

# Look for these in output:
# - F1 per case (0.00 = wrong, 1.00 = perfect)
# - Source: on-device or cloud (fallback)
# - TOTAL SCORE at bottom

# Submit when score improves (max 1x per hour):
python submit.py --team "YourTeamName" --location "London"

# View live leaderboard:
# https://cactusevals.ngrok.app
```

---

## 11. Known Gotchas

**Cactus model path** — must be exact:

```python
functiongemma_path = "cactus/weights/functiongemma-270m-it"
# NOT "weights/functiongemma-270m-it" — the cactus folder is a sibling, not inside the repo
```

**Always destroy model** — or you run out of RAM:

```python
model = cactus_init(...)
try:
    result = cactus_complete(model, ...)
finally:
    cactus_destroy(model)
```

**Gemini model string** — use `"gemini-2.0-flash-001"` not `"gemini-2.0-flash"` (deprecated):

```python
model="gemini-2.0-flash-001"
```

**Type conversion for Gemini** — integer fields must be `INTEGER` not `STRING`:

```python
type=v["type"].upper()   # "integer" → "INTEGER", "string" → "STRING"
```

**F1 scoring is case-insensitive and strips whitespace** — from benchmark.py:

```python
def _normalize(v):
    if isinstance(v, str):
        return v.strip().lower()
    return v
```

So "Good Morning" matches "good morning". Don't worry about casing in arguments.

**benchmark.py checks argument VALUES not just names** — if location is "San Francisco" but you return "SF", it scores 0 for that argument. Local model must extract the exact entity.

**source field must be exactly** `"on-device"` or `"cloud (fallback)"` — benchmark.py uses this string to count on-device ratio.

---

## 12. Winning Strategy Summary

The leaderboard formula rewards:

1. Getting hard multi-call cases right (50% weight)
2. Staying local wherever possible (25% weight)
3. Being fast (25% weight, easy to max out if local works)

So the win condition is: **correctly classify which cases local can handle, stay local for those, use cloud for the rest**.

The baseline in main.py uses confidence threshold alone — naive and loses local calls it could win.
Your strategy adds: pre-routing + output validation + tool RAG.

If you nail that, you're in top 10. Then for qualitative judging, a voice demo (cactus_transcribe → generate_hybrid → print executed action) running live is more impressive than any amount of architecture explanation.

---

## 13. Voice Demo (Rubric 3 — Build If Time Permits)

```python
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import json
from cactus import cactus_init, cactus_transcribe, cactus_destroy

TOOLS = [
    # ... same tools as benchmark ...
]

def record_audio(seconds=5, sample_rate=16000):
    print(f"Recording for {seconds} seconds...")
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    wav.write("input.wav", sample_rate, audio)
    print("Done recording.")

def voice_to_action():
    # Step 1: Record
    record_audio(seconds=5)

    # Step 2: Transcribe locally
    whisper = cactus_init("cactus/weights/whisper-small")
    prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
    raw = json.loads(cactus_transcribe(whisper, "input.wav", prompt=prompt))
    cactus_destroy(whisper)

    transcript = raw["response"]
    print(f"You said: {transcript}")

    # Step 3: Route through hybrid
    messages = [{"role": "user", "content": transcript}]
    result = generate_hybrid(messages, TOOLS)

    # Step 4: Show result
    print(f"Source: {result['source']}")
    print(f"Time: {result['total_time_ms']:.0f}ms")
    for call in result["function_calls"]:
        print(f"→ {call['name']}({call['arguments']})")

if __name__ == "__main__":
    while True:
        input("\nPress Enter to speak a command...")
        voice_to_action()
```

Install deps: `pip install sounddevice scipy`
Download whisper: `cactus download whisper-small`

---

## 14. Quick Reference Card (For Demo Day)

| Command                                      | What it does                    |
| -------------------------------------------- | ------------------------------- |
| `python benchmark.py`                        | Test your current score         |
| `python submit.py --team "X" --location "Y"` | Submit to leaderboard           |
| `python main.py`                             | Run the example (weather query) |
| `https://cactusevals.ngrok.app`              | Live leaderboard                |
| `https://www.reddit.com/r/cactuscompute/`    | Ask technical questions         |

| Parameter                             | Recommended value                        |
| ------------------------------------- | ---------------------------------------- |
| `tool_rag_top_k`                      | `2`                                      |
| `force_tools`                         | `True`                                   |
| `max_tokens`                          | `256`                                    |
| `confidence` threshold to trust local | `>= 0.65` for easy, `>= 0.75` for medium |
| Gemini model                          | `"gemini-2.0-flash-001"`                 |
