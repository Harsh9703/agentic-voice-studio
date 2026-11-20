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

user_input = input("What would you like the AI to do? ")

response = client.responses.create(
    model="gpt-5.5",
    input=user_input,
    tools=tools
)

for item in response.output:
    if item.type == "function_call":
        print("The model requested a tool!")
        print(f"Tool: {item.name}")
        print(f"Arguments: {item.arguments}")
    
        arguments = json.loads(item.arguments)
    
        if item.name == "generate_speech":
            result = generate_speech(
                text = arguments["text"],
                output_file = "outputs/speech.mp3"
            )
        
            print(f"Tool result: {result}")
    
    final_response = client.responses.create(
        model="gpt-5.5",
        previous_response_id=response.id,
        input=[
            {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": str(result)
            }
        ],
        tools=tools
    )
    print(final_response.output_text)    

    
