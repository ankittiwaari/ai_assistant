import asyncio

import psycopg
from agents import function_tool
from pgvector import Vector
from pgvector.psycopg import register_vector_async

from app.infra.config import (
    DB_NAME,
    DB_PASSWORD,
    DB_PORT,
    DB_USER,
    EMBEDDING_MODEL,
    MODEL_NAME,
)
from src.app.infra.client import get_client

client = get_client()


async def retrieve_context(query: str, client, top_k=1):
    response = await client.embeddings.create(input=query, model=EMBEDDING_MODEL)
    # wrap in Vector so it's sent as `vector`, not `double precision[]`
    query_vector = Vector(response.data[0].embedding)
    async with await psycopg.AsyncConnection.connect(
        f"dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD} port={DB_PORT}"
    ) as conn:
        await register_vector_async(conn)
        async with conn.cursor() as cur:
            await cur.execute(
                """
                    SELECT content, page_number
                    FROM document_chunks
                    ORDER BY embedding <=> %s
                    LIMIT %s
                """,
                (query_vector, top_k),
            )
            matches = await cur.fetchall()
            if not matches:
                return None
            if top_k == 1:
                return matches[0][0]
            return [{"content": row[0], "page": row[1]} for row in matches]


@function_tool
async def search_documents(query: str) -> str:
    """Search the ingested PDF documents (Indian Constitution, Five Year Plans).
    Use this for any question about their content. Returns passages with page numbers.

    Args:
        query: A standalone search query describing the information needed.
    """
    chunks = await retrieve_context(query=query, client=client, top_k=5)
    if not chunks:
        return "No relevant passages found."
    ctx = "\n\n".join(f"[Page {c['page']}]\n{c['content']}" for c in chunks)
    return ctx


async def get_answer(q: str):
    print("Calling tool")
    result = await retrieve_context(query=q, client=client, top_k=10)
    context = ""
    if result:
        context = [
            f"Page: {chunk['page']} Text: {chunk['content']}" for chunk in result
        ]
    relevant_context = (" ").join(context)
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question "
                    "using ONLY the provided context. If the answer is not in "
                    "the context, say 'I don't know'. When you get the result, "
                    "specify the page number as well in the response."
                ),
            },
            {
                "role": "user",
                "content": f"Context: {relevant_context}\n\nQuestion: {q}",
            },
        ],
    )
    print("\n\nResponse::\n")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    query = input("\nEnter your question:\n")
    # psycopg async doesn't support Windows' default ProactorEventLoop
    asyncio.run(get_answer(query), loop_factory=asyncio.SelectorEventLoop)
