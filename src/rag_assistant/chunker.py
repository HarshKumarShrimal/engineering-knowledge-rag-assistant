"""Sliding-window text chunker with document metadata and page tracking."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import List
from .document_parser import ParsedDocument


@dataclass
class DocumentChunk:
    chunk_id: str
    filename: str
    page_number: int
    text: str
    start_char: int
    end_char: int
    chunk_index: int


class DocumentChunker:
    """Chunks documents into overlapping passages preserving sentence context and metadata."""

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 80) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc: ParsedDocument) -> List[DocumentChunk]:
        all_chunks: List[DocumentChunk] = []
        global_idx = 0

        for page in doc.pages:
            text = page.text
            if not text:
                continue

            # Split text by character windows with overlap
            pos = 0
            text_len = len(text)

            while pos < text_len:
                end_pos = min(pos + self.chunk_size, text_len)
                # Try to break at sentence or word boundary if not at end
                if end_pos < text_len:
                    last_period = text.rfind(". ", pos, end_pos)
                    last_newline = text.rfind("\n", pos, end_pos)
                    break_point = max(last_period, last_newline)
                    if break_point > pos + self.chunk_size // 2:
                        end_pos = break_point + 1

                chunk_text = text[pos:end_pos].strip()
                if chunk_text:
                    deterministic_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc.filename}_{page.page_number}_{pos}"))
                    chunk = DocumentChunk(
                        chunk_id=deterministic_id,
                        filename=doc.filename,
                        page_number=page.page_number,
                        text=chunk_text,
                        start_char=pos,
                        end_char=end_pos,
                        chunk_index=global_idx,
                    )
                    all_chunks.append(chunk)
                    global_idx += 1

                if end_pos >= text_len:
                    break
                pos = max(pos + 1, end_pos - self.chunk_overlap)

        return all_chunks
