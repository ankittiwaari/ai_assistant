"""Retrieval: question -> rewritten queries -> hybrid candidates -> rerank -> cited context.

Checked against qdrant-client 1.20.0 and openai 3.37.0, with OpenRouter for every model
call (embeddings, query rewriting, reranking) through the OpenAI SDK. Does not import
Docling, so it is safe to use in a FastAPI request path.

Install:
    uv add qdrant-client openai pydantic python-dotenv
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient, models

from app.infra.client import get_client  # cached AsyncOpenAI pointed at OpenRouter
from app.infra.config import (
    BM25_MODEL,
    BM25_OPTS,
    COLLECTION,
    EMBEDDING_MODEL,
    QDRANT_URL,
    RERANK_MODEL,
    REWRITE_MODEL,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- data shapes


@dataclass
class RetrievedChunk:
    label: str  # "c1", "c2", ... the citation label used in the prompt
    text: str
    score: float  # rerank score (or fused score if reranking was skipped)
    doc_id: str
    chunk_index: int
    source: str
    heading_path: list[str]
    pages: list[int]


@dataclass
class RetrievalResult:
    question: str
    queries: list[str]  # what was actually searched
    chunks: list[RetrievedChunk]
    context: str  # labelled text block to put in the LLM prompt
    found: bool  # False -> nothing relevant enough: answer "not found"
    timings_ms: dict[str, float] = field(default_factory=dict)


class RewrittenQuery(BaseModel):
    """Schema the rewrite model must fill (sent as a strict JSON schema)."""

    standalone_query: str = Field(
        description="The user's latest question rewritten to be understandable on its own, "
        "resolving pronouns and references using the conversation."
    )
    alternative_queries: list[str] = Field(
        description="Different phrasings of the same information need, using synonyms or "
        "more specific terms. No new facts, no answers."
    )


class RerankResult(BaseModel):
    index: int  # position in the documents list that was sent
    relevance_score: float


class RerankResponse(BaseModel):
    results: list[RerankResult]  # best first


REWRITE_PROMPT = """You rewrite questions for a document search engine.
Conversation so far:
{history}

Latest question: {question}

Return the standalone version of the latest question and {n} alternative search phrasings."""


# --------------------------------------------------------------------------- model calls (OpenAI SDK -> OpenRouter)


async def embed_queries(queries: list[str]) -> list[list[float]]:
    """All query variants in one embeddings request."""
    resp = await get_client().embeddings.create(
        model=EMBEDDING_MODEL, input=queries, encoding_format="float"
    )
    return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]


async def call_rewrite_model(prompt: str) -> RewrittenQuery:
    out = await get_client().chat.completions.parse(
        model=REWRITE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=RewrittenQuery,
        temperature=0,
        extra_body={
            "provider": {"require_parameters": True}
        },  # only schema-enforcing providers
    )
    msg = out.choices[0].message
    if msg.parsed is None:
        raise ValueError(f"no parsable output (refusal={msg.refusal!r})")
    return msg.parsed


async def call_rerank(
    query: str, documents: list[str], top_n: int
) -> list[RerankResult]:
    """OpenRouter's POST /rerank. The SDK has no rerank method, so use its generic post()."""
    data = await get_client().post(
        "/rerank",
        cast_to=object,
        body={
            "model": RERANK_MODEL,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
        },
    )
    return RerankResponse.model_validate(data).results


# --------------------------------------------------------------------------- 1. query understanding


async def rewrite_query(
    question: str, history: list[dict], n_variants: int = 2
) -> list[str]:
    """Return [standalone query, *variants]. Falls back to the raw question on any failure."""
    if n_variants == 0 and not history:
        return [question]  # nothing to resolve, skip the LLM call
    turns = "\n".join(f"{t['role']}: {t['content']}" for t in history[-6:]) or "(none)"
    try:
        out = await call_rewrite_model(
            REWRITE_PROMPT.format(history=turns, question=question, n=n_variants)
        )
    except Exception:
        log.warning("query rewrite failed; searching the raw question", exc_info=True)
        return [question]

    queries = [out.standalone_query.strip() or question]
    for q in out.alternative_queries[:n_variants]:
        if q.strip() and q.strip().lower() not in {x.lower() for x in queries}:
            queries.append(q.strip())
    if not history and question.lower() not in {x.lower() for x in queries}:
        queries.insert(0, question)  # keep the user's own words as a safety net
    return queries


# --------------------------------------------------------------------------- 2. candidate generation


def tenant_filter(tenant_id: str) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="tenant_id", match=models.MatchValue(value=tenant_id)
            )
        ]
    )


async def hybrid_candidates(
    qdrant: AsyncQdrantClient,
    queries: list[str],
    tenant_id: str,
    per_query: int = 40,
    limit: int = 50,
    rrf_k: int = 60,
) -> list[models.ScoredPoint]:
    """One Qdrant request: dense + BM25 sub-search per query, all fused with RRF."""
    flt = tenant_filter(tenant_id)
    vectors = await embed_queries(queries)

    prefetch: list[models.Prefetch] = []
    for q, vec in zip(queries, vectors):
        prefetch.append(
            models.Prefetch(query=vec, using="dense", limit=per_query, filter=flt)
        )
        prefetch.append(
            models.Prefetch(
                query=models.Document(text=q, model=BM25_MODEL, options=BM25_OPTS),
                using="bm25",
                limit=per_query,
                filter=flt,
            )
        )

    res = await qdrant.query_points(
        COLLECTION,
        prefetch=prefetch,
        query=models.RrfQuery(
            rrf=models.Rrf(k=rrf_k)
        ),  # fuse every ranked list by position
        query_filter=flt,  # belt and braces: tenant on the final step too
        limit=limit,
        with_payload=True,
    )
    return res.points


# --------------------------------------------------------------------------- 3. rerank


def rerank_text(payload: dict) -> str:
    """What the reranker reads: heading path for context, then the chunk text."""
    path = " > ".join(payload.get("heading_path") or [])
    return f"{path}\n{payload['text']}" if path else payload["text"]


async def rerank_points(
    query: str, points: list[models.ScoredPoint], top_n: int, min_score: float
) -> list[tuple[models.ScoredPoint, float]]:
    """Cross-encoder rerank via OpenRouter. On failure, degrade to the fused order."""
    if not points:
        return []
    try:
        results = await call_rerank(
            query, [rerank_text(p.payload) for p in points], top_n=top_n
        )
    except Exception:
        log.warning("rerank failed; falling back to fused order", exc_info=True)
        return [(p, p.score) for p in points[:top_n]]
    return [
        (points[r.index], r.relevance_score)
        for r in results
        if r.relevance_score >= min_score
    ]


# --------------------------------------------------------------------------- 4. context assembly


def approx_tokens(text: str) -> int:
    return len(text) // 4  # rough English estimate; fine for a budget


def build_context(
    ranked: list[tuple[models.ScoredPoint, float]], max_context_tokens: int
) -> tuple[list[RetrievedChunk], str]:
    chunks: list[RetrievedChunk] = []
    blocks: list[str] = []
    used = 0
    for point, score in ranked:
        p = point.payload
        label = f"c{len(chunks) + 1}"
        where = " > ".join(p.get("heading_path") or []) or "(no heading)"
        pages = ", ".join(str(n) for n in p.get("pages") or []) or "?"
        block = f"[{label}] source: {p['source']} | section: {where} | pages: {pages}\n{p['text']}"
        cost = approx_tokens(block)
        if chunks and used + cost > max_context_tokens:
            break  # budget reached; keep the best-ranked ones
        used += cost
        blocks.append(block)
        chunks.append(
            RetrievedChunk(
                label=label,
                text=p["text"],
                score=round(score, 4),
                doc_id=p["doc_id"],
                chunk_index=p["chunk_index"],
                source=p["source"],
                heading_path=p.get("heading_path") or [],
                pages=p.get("pages") or [],
            )
        )
    return chunks, "\n\n".join(blocks)


# --------------------------------------------------------------------------- the pipeline


async def retrieve(
    qdrant: AsyncQdrantClient,
    question: str,
    tenant_id: str,
    history: list[dict] | None = None,
    *,
    n_variants: int = 2,
    per_query: int = 40,
    candidates: int = 50,
    rrf_k: int = 60,
    top_n: int = 6,
    min_score: float = 0.2,
    max_context_tokens: int = 3000,
) -> RetrievalResult:
    timings: dict[str, float] = {}

    t = time.perf_counter()
    queries = await rewrite_query(question, history or [], n_variants)
    timings["rewrite"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    points = await hybrid_candidates(
        qdrant, queries, tenant_id, per_query=per_query, limit=candidates, rrf_k=rrf_k
    )
    timings["search"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    # Rerank against the standalone query: it carries the resolved meaning of the question
    ranked = await rerank_points(queries[0], points, top_n=top_n, min_score=min_score)
    timings["rerank"] = (time.perf_counter() - t) * 1000

    chunks, context = build_context(ranked, max_context_tokens)
    return RetrievalResult(
        question=question,
        queries=queries,
        chunks=chunks,
        context=context,
        found=bool(chunks),
        timings_ms={k: round(v, 1) for k, v in timings.items()},
    )


# --------------------------------------------------------------------------- demo


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    qdrant = AsyncQdrantClient(url=QDRANT_URL)
    try:
        # Matches the constitution.pdf ingested by store.py: the follow-up only makes
        # sense with the history, so this exercises the query rewrite.
        history = [
            {"role": "user", "content": "What are fundamental rights?"},
            {
                "role": "assistant",
                "content": "They are basic rights guaranteed to citizens by the constitution.",
            },
        ]
        result = await retrieve(
            qdrant, "can they be suspended?", tenant_id="acme", history=history
        )

        print("searched:", result.queries)
        print("timings :", result.timings_ms)
        if not result.found:
            print("nothing relevant enough: answer 'not found'")
        for c in result.chunks:
            print("\n#############################\n")
            print(
                f"[{c.label}] {c.score}  {' > '.join(c.heading_path)}  pages {c.pages}"
            )
        print("\n--- context for the LLM ---\n" + result.context)
    finally:
        await qdrant.close()


if __name__ == "__main__":
    asyncio.run(main())
