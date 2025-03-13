import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

_ = load_dotenv()

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


async def generate_text(prompt: str, max_tokens: int = 100) -> str:
    """
    Calls the OpenAI API with the given prompt and returns the generated text.
    """
    try:
        response = await client.chat.completions.create(
            model="o1-mini",
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )

        content = response.choices[0].message.content
        assert isinstance(content, str)

        return content.strip()
    except Exception as e:
        print(f"Error generating text: {e}")
        return "Error generating text."
