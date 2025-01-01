from fastapi import APIRouter

from utils.generate_text import generate_text

router = APIRouter(prefix="/generate", tags=["Health Check"])


@router.get("/simple")
async def generate_simple():
    """
    Generate simple sentences for children.
    """
    prompt = "Generate 5 simple sentences for a child, difficulty: easy."
    result = await generate_text(prompt)

    return {"sentences": result.split("\n")}


@router.get("/story")
async def generate_story():
    """
    Generate a short story suitable for children.
    """
    prompt = "Generate a simple story for children with 5 sentences. Please format the story as a list of sentences, where each sentence is on a new line."
    result = await generate_text(prompt, max_tokens=200)

    return {"story": result}


@router.get("/conversation")
async def generate_conversation(ai_name: str = "AI assistant"):
    """
    Generate a conversation between a child and a (named) AI assistant.
    """
    prompt = f"Generate a 5-line conversation between a child and an AI named {ai_name}. The conversation should alternate between the child and the AI, without including speaker labels like 'Child' or 'AI'. The output should be formatted as a dictionary with two keys: 'child' and 'ai', each containing an array of strings representing their respective lines of dialogue."
    result = await generate_text(prompt)

    return {"conversation": result.split("\n")}
