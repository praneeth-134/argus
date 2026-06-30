import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def call_llm(system_prompt: str, user_prompt: str, model: str = "llama-3.1-8b-instant") -> str:
    """
    Calls the Groq API with a system + user prompt.
    Returns the raw text response.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,  # lower temperature = more focused, consistent diagnostic output
        max_tokens=500
    )
    return response.choices[0].message.content