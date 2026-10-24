import asyncio
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    TableFormerMode,
    TableStructureOptions,
    ThreadedPdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

from app.infra.client import get_client
from app.infra.config import EMBED_TOKENIZER, EMBEDDING_MODEL, MAX_TOKENS

client = get_client()


@dataclass
class Chunk:
    doc_id: str
    chunk_index: int
    text: str  # raw chunk text: shown to the user and the LLM
    embed_text: str  # heading path + text: what you embed and BM25-index
    source: str
    heading_path: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    item_types: list[str] = field(default_factory=list)  # e.g. ["table", "text"]
    doc_version: str = ""
    chunk_hash: str = ""
    embedding: list[float] | None = None


def build_converter(ocr: bool = False) -> DocumentConverter:
    opts = ThreadedPdfPipelineOptions(
        do_ocr=ocr,
        do_table_structure=True,
        table_structure_options=TableStructureOptions(
            mode=TableFormerMode.ACCURATE, do_cell_matching=True
        ),
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def build_chunker() -> HybridChunker:
    tokenizer = HuggingFaceTokenizer.from_pretrained(
        EMBED_TOKENIZER, max_tokens=MAX_TOKENS
    )
    return HybridChunker(
        tokenizer=tokenizer, merge_peers=True, repeat_table_header=True
    )


def chunk_file(
    path: str, doc_id: str, convertor: DocumentConverter, chunker: HybridChunker
) -> list[Chunk]:
    doc = convertor.convert(path).document
    doc_version = hashlib.sha256(doc.export_to_markdown().encode()).hexdigest()[:16]
    out: list[Chunk] = []

    for i, ch in enumerate(chunker.chunk(dl_doc=doc)):
        embed_text = chunker.contextualize(chunk=ch)
        items = ch.meta.doc_items
        out.append(
            Chunk(
                doc_id=doc_id,
                chunk_index=i,
                text=ch.text,
                embed_text=embed_text,
                source=path,
                heading_path=list(ch.meta.headings or []),
                pages=sorted({p.page_no for item in items for p in item.prov}),
                item_types=sorted({item.label.value for item in items}),
                doc_version=doc_version,
                chunk_hash=hashlib.sha256(embed_text.encode()).hexdigest()[:16],
            )
        )
    return out


async def _embed_batch(batch: list[str], sem: asyncio.Semaphore) -> list[list[float]]:
    async with sem:  # caps requests in flight
        resp = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
            encoding_format="float",  # plain float lists; don't rely on base64 support
        )
    if len(resp.data) != len(batch):
        raise RuntimeError(f"expected {len(batch)} embeddings, got {len(resp.data)}")
    # Results carry an index; sort so vectors line up with the input order
    return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]


async def embed(
    texts: list[str], batch_size: int = 64, concurrency: int = 8
) -> list[list[float]]:
    """Embed many texts: batched requests, at most `concurrency` in flight at once.

    For a limit shared by every job in a worker, create one semaphore at startup
    and pass it in instead of creating one per call.
    """
    sem = asyncio.Semaphore(concurrency)
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    results = await asyncio.gather(*(_embed_batch(b, sem) for b in batches))
    return [v for batch in results for v in batch]  # gather preserves order


async def embed_one(text: str) -> list[float]:
    """Query time: embed a single user query."""
    return (await embed([text]))[0]


async def main() -> None:
    converter, chunker = (
        build_converter(),
        build_chunker(),
    )
    # Docling parsing is CPU-heavy and synchronous: keep it off the event loop
    chunks = await asyncio.to_thread(
        chunk_file,
        Path(__file__).parent.parent / "documents" / "constitution.pdf",
        "panchvarshiya-yojana-v1",
        converter,
        chunker,
    )

    vectors = await embed([c.embed_text for c in chunks])
    for c, v in zip(chunks, vectors):
        c.embedding = v

    print(f"{len(chunks)} chunks, embedding dim = {len(chunks[0].embedding)}")
    for c in chunks[:5]:
        print(f"#{c.chunk_index} pages={c.pages} types={c.item_types}")
        print("  path:", " > ".join(c.heading_path))
        print("  ", c.text[:120].replace("\n", " "), "...\n")


if __name__ == "__main__":
    asyncio.run(main())
