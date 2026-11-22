from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from tts import generate_speech


load_dotenv()

llm = ChatOpenAI(model="gpt-5.5")

SYSTEM_PROMPT = """
You are an AI Voice Studio agent.

Your job is to help users create high-quality spoken content.

You can:
- help write and rewrite text for natural narration
- estimate how long text will take to speak
- generate spoken audio when the user requests it

Use tools when they are useful for completing the user's request.
Do not generate audio unless the user asks for spoken audio or an audio file.
"""



@tool
def generate_speech_tool(text: str) -> str:
    """Generate an MP3 audio file containing spoken audio from the provided text."""

    output_path = generate_speech(
        text=text,
        output_file="outputs/speech.mp3"
    )

    return str(output_path)

@tool
def estimate_duration(text: str) -> str:
    """Estimate how many minutes the provided text will take to speak aloud."""

    words = len(text.split())
    minutes = words / 150

    return f"Estimated speaking duration: {minutes:.2f} minutes"


tools = [generate_speech_tool, estimate_duration]

llm_with_tools = llm.bind_tools(tools)


def agent(state: MessagesState):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ] + state["messages"]

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }


builder = StateGraph(MessagesState)

builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")

builder.add_conditional_edges(
    "agent",
    tools_condition
)

builder.add_edge("tools", "agent")

graph = builder.compile()


user_input = input("What would you like the AI to do? ")

result = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": user_input
            }
        ]
    }
)

print(result["messages"][-1].content)

print("\n--- FULL MESSAGE HISTORY ---")

for message in result["messages"]:
    print(type(message).__name__)
    print(message)
    print()


