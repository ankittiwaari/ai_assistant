from dataclasses import dataclass, field

from docling.datamodel.pipeline_options import (
    ThreadedPdfPipelineOptions,
)
from docling.document_converter import DocumentConverter


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


def build_convertor(ocr: bool = False) -> DocumentConverter:
    opts = ThreadedPdfPipelineOptions
