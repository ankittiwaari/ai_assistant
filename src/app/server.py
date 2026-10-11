from fastapi import FastAPI
from pydantic import BaseModel, Field

from .main import init_chat

app = FastAPI()


class RequestBody(BaseModel):
    question: str | None = Field(
        default=None, title="The question you want to ask", max_length=300
    )


@app.post("/ask")
async def root(body: RequestBody):
    if body.question is None:
        return ""
    return await init_chat(body.question, [])
