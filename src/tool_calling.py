from dotenv import load_dotenv
from openai import OpenAI
import json
from tts import generate_speech

load_dotenv()

client = OpenAI()

tools = [
    {
        "type": "function",
        "name": "generate_speech",
        "description": "Generate an MP3 audio file from text. Use this when the user asks to hear, speak, narrate, or create audio from text.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The exact text that should be spoken."
                }
            },
            "required": ["text"],
            "additionalProperties": False
        },
        "strict": True
    }
]

def execute_tool(name, arguments):
    if name == "generate_speech":
        return generate_speech(
            text=arguments["text"],
            output_file="outputs/speech.mp3"
        )

    raise ValueError(f"Unknown tool: {name}")


user_input = input("What would you like the AI to do? ")

response = client.responses.create(
    model="gpt-5.5",
    input=user_input,
    tools=tools
)

while True:
    tool_calls = [
        item
        for item in response.output
        if item.type == "function_call"
    ]

    if not tool_calls:
        print(response.output_text)
        break

    tool_outputs = []

    for tool_call in tool_calls:
        arguments = json.loads(tool_call.arguments)

        print(f"Calling tool: {tool_call.name}")
        print(f"Arguments: {arguments}")

        result = execute_tool(
            tool_call.name,
            arguments
        )

        print(f"Tool result: {result}")

        tool_outputs.append(
            {
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": str(result)
            }
        )

    response = client.responses.create(
        model="gpt-5.5",
        previous_response_id=response.id,
        input=tool_outputs,
        tools=tools
    )

    
