from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from tts import generate_speech

load_dotenv()

llm = ChatOpenAI(model="gpt-5.5")


class PodcastState(TypedDict):
    file_path: str
    document_text: str
    script: str
    duration: float
    revision_count: int
    audio_path: str

def read_document(state: PodcastState):
    with open(state["file_path"], "r", encoding="utf-8") as file:
        text = file.read()

    return {
        "document_text": text
    }


def create_script(state: PodcastState):
    prompt = f"""
    Turn the following source material into a natural,
    beginner-friendly spoken narration.

    Target approximately one minute of spoken audio.

    Preserve the important information from the source.
    Explain difficult concepts clearly.
    Do not invent facts that are not supported by the source.

    Write only the narration script.

    SOURCE MATERIAL:

    {state["document_text"]}
    """

    response = llm.invoke(prompt)

    return {
        "script": response.content
    }

def estimate_duration(state: PodcastState):
    word_count = len(state["script"].split())
    duration = word_count / 150

    print(
        f"Duration check: {word_count} words "
        f"≈ {duration:.2f} minutes"
    )

    return {
        "duration": duration
    }

MAX_REVISIONS = 3
def check_duration(state: PodcastState):
    if 0.9 <= state["duration"] <= 1.1:
        return "generate_audio"
    
    if state["revision_count"] >= MAX_REVISIONS:
        return "generate_audio"

    return "rewrite"

def rewrite_script(state: PodcastState):
    direction = (
        "shorter"
        if state["duration"] > 1.1
        else "longer"
    )

    prompt = f"""
    Rewrite the narration below so that it is {direction}
    and approximately one minute when spoken at 150 words per minute.

    Write only the revised narration.

    Narration:
    {state["script"]}
    """

    response = llm.invoke(prompt)

    return {
        "script": response.content,
        "revision_count": state["revision_count"] + 1
    }


def generate_audio(state: PodcastState):
    output_path = generate_speech(
        text=state["script"],
        output_file="outputs/podcast.mp3"
    )

    return {
        "audio_path": str(output_path)
    }

builder = StateGraph(PodcastState)
builder.add_node("read_document", read_document)
builder.add_node("create_script", create_script)
builder.add_node("estimate_duration", estimate_duration)
builder.add_node("rewrite", rewrite_script)
builder.add_node("generate_audio", generate_audio)

builder.add_edge(START, "read_document")
builder.add_edge("read_document", "create_script")
builder.add_edge("create_script", "estimate_duration")

builder.add_conditional_edges(
    "estimate_duration",
    check_duration,
    {
        "rewrite": "rewrite",
        "generate_audio": "generate_audio"
    }
)

builder.add_edge("rewrite", "estimate_duration")
builder.add_edge("generate_audio", END)

graph = builder.compile()

file_path = input("Enter the path to your document: ")

result = graph.invoke(
    {
        "file_path": file_path,
        "document_text": "",
        "script": "",
        "duration": 0.0,
        "revision_count": 0,
        "audio_path": ""
    }
)


print("\n--- FINAL RESULT ---")
print(f"Duration: {result['duration']:.2f} minutes")
print(f"Revisions: {result['revision_count']}")
print(f"Audio: {result['audio_path']}")
print("\nScript:")
print(result["script"])