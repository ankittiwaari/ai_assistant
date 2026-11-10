from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from pgvector.sqlalchemy import VECTOR
from qdrant_client import AsyncQdrantClient, models
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    Table,
    Text,
    delete,
    func,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.infra.config import COLLECTION, EMBEDDING_MODEL, QDRANT_URL
from app.infra.db import dispose_engine, get_engine
from app.rag_agent.ingestion.docling_chunker import (
    Chunk,
    build_chunker,
    build_converter,
    chunk_file,
    embed,
    embed_one,
)

BM25_OPTS = models.Bm25Config(k=1.2, b=0.75, avg_len=300)
BM25_MODEL = "Qdrant/bm25"
POINT_NS = uuid.UUID("6f1c2b0e-4d1a-4a8e-9a57-3c2f8a0d9b11")

metadata = MetaData()
documents = Table(
    "documents",
    metadata,
    Column("doc_id", Text, primary_key=True),
    Column("tenant_id", Text, nullable=False),
    Column("source", Text, nullable=False),
    Column("active_version", Text),
    Column(
        "status", Text, nullable=False, server_default="queued"
    ),  # queued|indexing|ready|failed
    Column("error", Text),
    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    ),
)

chunks_t = Table(
    "chunks",
    metadata,
    Column(
        "doc_id",
        Text,
        ForeignKey("documents.doc_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("doc_version", Text, nullable=False),
    Column("chunk_index", Integer, nullable=False),
    Column("chunk_hash", Text, nullable=False),
    Column("content", Text, nullable=False),  # raw text shown to users / the LLM
    Column(
        "embed_text", Text, nullable=False
    ),  # heading path + content: embedded and BM25-indexed
    Column("heading_path", ARRAY(Text), nullable=False, server_default="{}"),
    Column("pages", ARRAY(Integer), nullable=False, server_default="{}"),
    Column("item_types", ARRAY(Text), nullable=False, server_default="{}"),
    Column(
        "embed_model", Text, nullable=False
    ),  # never reuse a vector across embedding models
    Column(
        "embedding", VECTOR(), nullable=False
    ),  # cache only: no vector index, Postgres doesn't search
    PrimaryKeyConstraint("doc_id", "doc_version", "chunk_index"),
    Index("chunks_hash_idx", "chunk_hash", "embed_model"),
)


async def ensure_postgres() -> None:
    async with get_engine().begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(metadata.create_all)


async def ensure_qudrant(dim: int) -> AsyncQdrantClient:
    client = AsyncQdrantClient(url=QDRANT_URL)
    if not await client.collection_exists(COLLECTION):
        await client.create_collection(
            COLLECTION,
            vectors_config={
                "dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )

    await client.create_payload_index(
        COLLECTION,
        "tenant_id",
        field_schema=models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD, is_tenant=True
        ),
    )
    for field in ("doc_id", "doc_version"):
        await client.create_payload_index(
            COLLECTION, field, models.PayloadSchemaType.KEYWORD
        )
    return client


def point_id(doc_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(POINT_NS, f"{doc_id}:{chunk_index}"))


async def index_document(
    qdrant: AsyncQdrantClient,
    tenant_id: str,
    doc_id: str,
    source: str,
    chunks: list[Chunk],
    upsert_batch: int = 128,
) -> dict:
    engine = get_engine()
    version = chunks[0].doc_version
    async with engine.begin() as conn:
        active = await conn.scalar(
            select(documents.c.active_version).where(documents.c.doc_id == doc_id)
        )
        if active == version:
            return {"doc_id": doc_id, "skipped": True}
        await conn.execute(
            pg_insert(documents)
            .values(
                doc_id=doc_id, tenant_id=tenant_id, source=source, status="indexing"
            )
            .on_conflict_do_update(
                index_elements=[documents.c.doc_id],
                set_={
                    "status": "indexing",
                    "error": None,
                    "source": source,
                    "updated_at": func.now(),
                },
            )
        )

    try:
        async with engine.connect() as conn:
            rows = await conn.execute(
                select(chunks_t.c.chunk_hash, chunks_t.c.embedding).where(
                    chunks_t.c.doc_id == doc_id,
                    chunks_t.c.embed_model == EMBEDDING_MODEL,
                    chunks_t.c.chunk_hash.in_([c.chunk_hash for c in chunks]),
                )
            )
            cached: dict[str, list[float]] = {
                h: v for h, v in rows
            }  # VECTOR -> list[float]
        todo = [c for c in chunks if c.chunk_hash not in cached]
        fresh = await embed([c.embed_text for c in todo]) if todo else []
        vec_for = cached | {c.chunk_hash: v for c, v in zip(todo, fresh)}

        async with engine.begin() as conn:
            await conn.execute(
                pg_insert(chunks_t).on_conflict_do_nothing(
                    index_elements=["doc_id", "doc_version", "chunk_index"]
                ),
                [
                    {
                        "doc_id": doc_id,
                        "doc_version": version,
                        "chunk_index": c.chunk_index,
                        "chunk_hash": c.chunk_hash,
                        "content": c.text,
                        "embed_text": c.embed_text,
                        "heading_path": c.heading_path,
                        "pages": c.pages,
                        "item_types": c.item_types,
                        "embed_model": EMBEDDING_MODEL,
                        "embedding": vec_for[c.chunk_hash],
                    }
                    for c in chunks
                ],
            )

        points = [
            models.PointStruct(
                id=point_id(doc_id, c.chunk_index),
                vector={
                    "dense": vec_for[c.chunk_hash],
                    "bm25": models.Document(
                        text=c.embed_text, model=BM25_MODEL, options=BM25_OPTS
                    ),
                },
                payload={
                    "tenant_id": tenant_id,
                    "doc_id": doc_id,
                    "doc_version": version,
                    "chunk_index": c.chunk_index,
                    "source": source,
                    "text": c.text,
                    "heading_path": c.heading_path,
                    "pages": c.pages,
                    "item_types": c.item_types,
                },
            )
            for c in chunks
        ]

        for i in range(0, len(points), upsert_batch):
            await qdrant.upsert(
                COLLECTION, points=points[i : i + upsert_batch], wait=True
            )

        await qdrant.delete(
            COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id", match=models.MatchValue(value=doc_id)
                        )
                    ],
                    must_not=[
                        models.FieldCondition(
                            key="doc_version", match=models.MatchValue(value=version)
                        )
                    ],
                )
            ),
            wait=True,
        )

        async with engine.begin() as conn:
            await conn.execute(
                update(documents)
                .where(documents.c.doc_id == doc_id)
                .values(active_version=version, status="ready", updated_at=func.now())
            )
            await conn.execute(
                delete(chunks_t).where(
                    chunks_t.c.doc_id == doc_id, chunks_t.c.doc_version != version
                )
            )
    except Exception as exc:
        async with engine.begin() as conn:
            await conn.execute(
                update(documents)
                .where(documents.c.doc_id == doc_id)
                .values(status="failed", error=str(exc)[:500], updated_at=func.now())
            )
        raise

    return {
        "doc_id": doc_id,
        "version": version,
        "chunks": len(chunks),
        "embedded": len(todo),
        "reused": len(chunks) - len(todo),
    }


async def delete_document(qdrant: AsyncQdrantClient, doc_id: str) -> None:
    """Search index first, then the source of truth (chunks cascade)."""
    await qdrant.delete(
        COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id", match=models.MatchValue(value=doc_id)
                    )
                ]
            )
        ),
        wait=True,
    )
    async with get_engine().begin() as conn:
        await conn.execute(delete(documents).where(documents.c.doc_id == doc_id))


# --------------------------------------------------------------------------- read path (sanity check)


async def hybrid_search(
    qdrant: AsyncQdrantClient,
    query: str,
    tenant_id: str,
    candidates: int = 50,
    top: int = 10,
) -> list[models.ScoredPoint]:
    tenant = models.Filter(
        must=[
            models.FieldCondition(
                key="tenant_id", match=models.MatchValue(value=tenant_id)
            )
        ]
    )
    qvec = await embed_one(query)
    res = await qdrant.query_points(
        COLLECTION,
        prefetch=[
            models.Prefetch(query=qvec, using="dense", limit=candidates, filter=tenant),
            models.Prefetch(
                query=models.Document(text=query, model=BM25_MODEL, options=BM25_OPTS),
                using="bm25",
                limit=candidates,
                filter=tenant,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top,
        with_payload=True,
    )
    return res.points


# --------------------------------------------------------------------------- end to end


async def main() -> None:
    source = str(Path(__file__).parent.parent / "documents" / "constitution.pdf")
    converter, chunker = build_converter(), build_chunker()
    chunks = await asyncio.to_thread(chunk_file, source, "manual", converter, chunker)

    await ensure_postgres()
    dim = len(await embed_one("dimension probe"))  # size the collection from the model
    qdrant = await ensure_qudrant(dim)
    try:
        print(
            await index_document(
                qdrant,
                tenant_id="acme",
                doc_id="manual",
                source=source,
                chunks=chunks,
            )
        )
        for p in await hybrid_search(
            qdrant, "What are fundamental rights?", tenant_id="acme"
        ):
            print(
                round(p.score, 4),
                " > ".join(p.payload["heading_path"]),
                p.payload["pages"],
            )
    finally:
        await qdrant.close()
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
