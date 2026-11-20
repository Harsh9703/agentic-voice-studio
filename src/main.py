from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

user_input = input("What would you like the AI to do? ")

response = client.responses.create(
    model="gpt-5.5",
    input=user_input
)

print(response.output_text)
