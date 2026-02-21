
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"
# gemma_path="cactus/weights/gemma-3-270m-it"
from pathlib import Path
gemma_path = str(Path(__file__).parent.parent / "cactus/weights/gemma-3-1b-it")

import json, os, time
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types

def __identify_subtasks(model, messages):
    """Use the local model to identify subtasks in a complex user request."""
    # raw_str = cactus_complete(
    #     model,
    #     [{"role": "system", "content": "You are a helpful assistant that breaks down complex requests into subtasks. Break the following request into subtasks, if it is a complex request with multiple parts. If there is no request or it is a simple request, return the original message. Return a JSON object with a 'subtasks' key that is a list of subtasks."}] + messages,
    #     max_tokens=256,
    #     stop_sequences=["<|im_end|>", "<end_of_turn>"],
    # )

    user_message = messages[0]["content"] # get the message from the user
    example1 = "Request 1: Set a 15 minute timer, play classical music, and remind me to stretch at 4:00 PM. \n Task 1: Set a 15 minute timer \n Task 2: Play classical music \n Task 3: remind me to stretch at 4:00 PM"
    example2 = "Request 2: Send a message to Jula saying good morning. \n Task 1: Send a message to Julia saying good morning"
    example3 = "Request 3: Find Tom in my contacts and send him a message saying happy birthday. \n Task 1: Find Tom in my contacts \n Task 2: Send Tom a message that says happy birthday "
    raw_str = cactus_complete(
        model,
        [{"role": "user", "content": f"The request message is made up of one or more subtasks. Change all pronouns (e.g. him, her, them, etc.) to the correct name. Break the request into sub tasks like the examples: \n {example1} \n {example2} \n {example3} \n Request message: {user_message} \n ONLY OUTPUT THE TASKS, DO NOT OUTPUT ANYTHING ELSE"}] ,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    try:
        raw = json.loads(raw_str)
        #output =  [[{'role': 'user', 'content': x}] for x in raw.get("response").split('\n') if len(x)>5 and 'DO NOT OUTPUT' not in x]
        output =  [[{'role': 'user', 'content': x}] for x in user_message.split('and')]

        return output
    except json.JSONDecodeError:
        return []
    
def __generate_cactus(model, messages, tools):
    """Helper to run function calling on-device via FunctionGemma + Cactus."""
    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]
    print("=====>SUBTASK TOOLS")
    print(cactus_tools)
    print("======>SUBTASK MESSAGES")
    print(messages)
    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools. Remove leading articles (e.g. 'the') from task parameters. Don't insert punctuation in task parameters."}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )
    print("R=====> SUBTASK RAW RESPONSE")
    print(raw_str)

    try:
        raw = json.loads(raw_str)
        return {
            "function_calls": raw.get("function_calls", []),
            "total_time_ms": raw.get("total_time_ms", 0),
            "confidence": raw.get("confidence", 0),
        }
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
        }

def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    helper_model = cactus_init(gemma_path)
    model = cactus_init(functiongemma_path)

    submessages = __identify_subtasks(helper_model, messages)
    result = {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 1,
        }
    print("SPLIT RESULTS")
    print(submessages)
    for i, sub in enumerate(submessages):
        print(f"SUBTASK: {i}")
        sub_result = __generate_cactus(model, sub, tools)
        result["function_calls"].extend(sub_result["function_calls"])
        result["total_time_ms"] += sub_result["total_time_ms"]
        result["confidence"] = min(result["confidence"], sub_result["confidence"])

    cactus_destroy(model)
    cactus_destroy(helper_model)

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


def generate_hybrid(messages, tools, confidence_threshold=0.1):
    """Baseline hybrid inference strategy; fall back to cloud if Cactus Confidence is below threshold."""
    #ask local model to first split the message into multiple messages  if the request has multiple parts
    

    local = generate_cactus(messages, tools)
    print(f"=== CONFIDENCE: {local["confidence"]}")
    if local["confidence"] >= confidence_threshold:
        print(f"===== MESSAGES \n {messages}")
        local["source"] = "on-device"
        return local

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
    model = cactus_init(functiongemma_path)
    messages = [{"role": "user", "content": "Text Emma saying good night, check the weather in Chicago, and set an alarm for 5 AM."}]
    # messages = [{"role": "user", "content": "Set a 15 minute timer, play classical music, and remind me to stretch at 4:00 PM."}]
    # example = "Example: 'Look up Jake in my contacts, send him a message saying let's meet, and check the weather in Seattle.'->'Look up Jake in my contacts; send Jake a message saying let's meet; check the weather in Seattle.'"
    # raw_str = cactus_complete(
    #     model,
    #     [{"role": "user", "content": "If the following messsage has subrequest, split it into its constituent parts on the basis of commas or 'and's, otherwise return the original message. Do not reply to the content"+example}] + messages,
    #     max_tokens=256,
    #     stop_sequences=["<|im_end|>", "<end_of_turn>"],
    # )
    user_message = messages[0]["content"] # get the message from the user
    example1 = "Request 1: Set a 15 minute timer, play classical music, and remind me to stretch at 4:00 PM. \n Task 1: Set a 15 minute timer \n Task 2: Play classical music \n Task 3: remind me to stretch at 4:00 PM"
    example2 = "Request 2: Send a message to Jula saying good morning. \n Task 1: Send a message to Julia saying good morning"
    example3 = "Request 3: Find Tom in my contacts and send him a message saying happy birthday. \n Task 1: Find Tom in my contacts \n Task 2: Send Tom a message that says happy birthday "
    raw_str = cactus_complete(
        model,
        [{"role": "user", "content": f"The user request is made up of one or more subtasks. Change all pronouns (e.g. him, her, them, etc.) to the correct name. Break the request into sub tasks like the examples: \n {example1} \n {example2} \n {example3} \n User message: {user_message}"}] ,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    raw = json.loads(raw_str)
    print(raw.get("response"))
    cactus_destroy(model)
    # tools = [{
    #     "name": "get_weather",
    #     "description": "Get current weather for a location",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "location": {
    #                 "type": "string",
    #                 "description": "City name",
    #             }
    #         },
    #         "required": ["location"],
    #     },
    # }]

    # messages = [
    #     {"role": "user", "content": "What is the weather in San Francisco?"}
    # ]
                
        # on_device = generate_cactus(messages, tools)
        # print_result("FunctionGemma (On-Device Cactus)", on_device)

        # cloud = generate_cloud(messages, tools)
        # print_result("Gemini (Cloud)", cloud)

        # hybrid = generate_hybrid(messages, tools)
        # print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)