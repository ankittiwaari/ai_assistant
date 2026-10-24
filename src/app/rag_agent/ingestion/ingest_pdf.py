import asyncio
from pathlib import Path

import psycopg
import pymupdf4llm
from pgvector.psycopg import register_vector_async

from app.infra.client import get_client
from app.infra.config import DB_NAME, DB_PASSWORD, DB_PORT, DB_USER, EMBEDDING_MODEL
from app.infra.logging_config import setup_logging

MAX_CONCURRENT_REQUESTS = 8
# resolve relative to this file, not the current working directory
PDF_PATH = Path(__file__).resolve().parent.parent / "documents" / "constitution.pdf"

client = get_client()


async def embed(text: str, semaphore: asyncio.Semaphore) -> list[float]:
    async with semaphore:
        response = await client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding


async def main():
    setup_logging()
    # pymupdf4llm is blocking (OCR), so run it off the event loop
    chunks = await asyncio.to_thread(
        pymupdf4llm.to_markdown,
        str(PDF_PATH),
        force_ocr=True,
        page_chunks=True,
    )
    assert isinstance(chunks, list)  # page_chunks=True returns a list of dicts

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    embeddings = await asyncio.gather(
        *(embed(chunk["text"], semaphore) for chunk in chunks)
    )

    rows = [
        (chunk["metadata"].get("page_number"), chunk["text"], embedding)
        for chunk, embedding in zip(chunks, embeddings)
    ]

    async with await psycopg.AsyncConnection.connect(
        f"dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD} port={DB_PORT}"
    ) as conn:
        await register_vector_async(conn)
        async with conn.cursor() as cur:
            await cur.executemany(
                """
                INSERT INTO document_chunks (page_number, content, embedding)
                VALUES (%s, %s, %s)
                """,
                rows,
            )
        # the connection context manager commits on clean exit

    print(f"Successfully extracted {len(chunks)} pages with citations.")


if __name__ == "__main__":
    # psycopg async doesn't support Windows' default ProactorEventLoop
    asyncio.run(main(), loop_factory=asyncio.SelectorEventLoop)
