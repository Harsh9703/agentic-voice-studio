from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


def generate_speech(text, output_file):
    output_path = Path(output_file)

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=text,
        instructions="Speak clearly at a steady, conversational pace.",
    ) as response:
        response.stream_to_file(output_path)

    return output_path


if __name__ == "__main__":
    text = input("What would you like me to say? ")

    saved_file = generate_speech(
        text,
        "outputs/speech.mp3"
    )

    print(f"Audio saved to {saved_file}")
