import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import chainlit as cl

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/v1")
MODEL = os.getenv("RAG_MODEL", "rag-model")

client = AsyncOpenAI(
    api_key="not-required",
    base_url=BACKEND_URL,
)


@cl.on_chat_start
async def on_start():
    cl.user_session.set("history", [])
    await cl.Message(
        content="Hello! I'm your **RAG + MCP Assistant**.\n\n"
        "I can search your knowledge base and query your PostgreSQL database to answer questions. "
        "What would you like to know?"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    history: list = cl.user_session.get("history", [])
    history.append({"role": "user", "content": message.content})

    response_msg = cl.Message(content="")
    await response_msg.send()

    response_text = ""

    try:
        stream = await client.chat.completions.create(
            model=MODEL,
            messages=history,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                response_text += delta
                await response_msg.stream_token(delta)

    except Exception as e:
        error = f"\n\n**Error:** {e}\n\nMake sure the backend is running at `{BACKEND_URL}`."
        response_text += error
        await response_msg.stream_token(error)

    await response_msg.update()
    history.append({"role": "assistant", "content": response_text})
    cl.user_session.set("history", history)
