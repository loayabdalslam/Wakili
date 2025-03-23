from dotenv import load_dotenv
import os

load_dotenv()

print('GEMINI_KEY: ',os.getenv("GEMINI_API_KEY"))
print("GEMINI_MODEL_NAME: ",os.getenv("GEMINI_MODEL"))


import base64
import os
from google import genai
from google.genai import types


def llm_gemini(question):
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    model = os.getenv("GEMINI_MODEL")
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=question),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=.7,
        top_p=0.95,
        top_k=64,
        max_output_tokens=8192,
        response_mime_type="text/plain",
    )
        
    genereate_content = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    return genereate_content



from pydantic import BaseModel

class ResearchRespone(BaseModel):
    topic : str
    summary: str
    soruces : list[str]
    tools_used: list[str]


inputx = input("Enter your question: ")
print(llm_gemini(inputx).text)