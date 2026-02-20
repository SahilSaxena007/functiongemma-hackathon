
############## Don't change setup ##############

import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"


############## Using Cactus ##############

from cactus import cactus_init, cactus_complete, cactus_destroy
import json

model = cactus_init(functiongemma_path)

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["location"]
        }
    }
}]

messages = [
    {"role": "system", "content": "You are a helpful assistant that can use tools."},
    {"role": "user", "content": "What is the weather in San Francisco?"}
]

response = json.loads(cactus_complete(
    model,
    messages,
    tools=tools,
    force_tools=True,
    max_tokens=256,
    stop_sequences=["<|im_end|>", "<end_of_turn>"]
))

cactus_destroy(model)

############## Print resonse and function call ############## 

print("\n=== Full On-Device Response ===\n")
print(json.dumps(response, indent=2))

print("\n=== Function Calls FunctionGemma ===\n")
for call in response.get("function_calls", []):
    print(f"Function: {call['name']}")
    print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


############## Gemini via GCP #################

import os
from google import genai
from google.genai import types


def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    get_weather_tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="get_weather",
                description="Get current weather for a location",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "location": types.Schema(
                            type="STRING",
                            description="City name",
                        ),
                    },
                    required=["location"],
                ),
            ),
        ],
    )

    gemini_response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="What is the weather in San Francisco?",
        config=types.GenerateContentConfig(
            tools=[get_weather_tool],
        ),
    )

    print("\n=== Gemini Response ===\n")
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                print(f"Function: {part.function_call.name}")
                print(f"Arguments: {json.dumps(dict(part.function_call.args), indent=2)}")
            elif part.text:
                print(part.text)


generate()