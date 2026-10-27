"""Document parsing module for text, markdown, and engineering PDF files."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Union
import pypdf


class ParsedPage:
    def __init__(self, page_number: int, text: str):
        self.page_number = page_number
        self.text = text.strip()


class ParsedDocument:
    def __init__(self, filename: str, pages: List[ParsedPage], metadata: Dict[str, str]):
        self.filename = filename
        self.pages = pages
        self.metadata = metadata

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text)


class DocumentParser:
    """Extracts clean text and page-level metadata from engineering manuals."""

    @staticmethod
    def parse(file_input: Union[str, Path, bytes], filename: str) -> ParsedDocument:
        suffix = Path(filename).suffix.lower()

        if suffix in (".txt", ".md", ".log"):
            return DocumentParser._parse_text(file_input, filename)
        elif suffix == ".pdf":
            return DocumentParser._parse_pdf(file_input, filename)
        else:
            raise ValueError(f"Unsupported document format '{suffix}'. Supported: .pdf, .txt, .md")

    @staticmethod
    def _parse_text(file_input: Union[str, Path, bytes], filename: str) -> ParsedDocument:
        raw_text: str
        if isinstance(file_input, (str, Path)):
            raw_text = Path(file_input).read_text(encoding="utf-8", errors="replace")
        elif isinstance(file_input, bytes):
            raw_text = file_input.decode("utf-8", errors="replace")
        else:
            raise TypeError("Unsupported file input type for text parsing.")

        # Split into synthetic pages if double form-feeds or long sections exist
        sections = raw_text.split("\n\nSECTION ")
        pages = []
        if len(sections) > 1:
            for idx, sec in enumerate(sections, start=1):
                prefix = "" if idx == 1 else "SECTION "
                pages.append(ParsedPage(page_number=idx, text=prefix + sec))
        else:
            pages.append(ParsedPage(page_number=1, text=raw_text))

        return ParsedDocument(
            filename=filename,
            pages=pages,
            metadata={"format": "text", "total_pages": str(len(pages))},
        )

    @staticmethod
    def _parse_pdf(file_input: Union[str, Path, bytes], filename: str) -> ParsedDocument:
        pdf_stream: io.BytesIO
        if isinstance(file_input, (str, Path)):
            pdf_stream = io.BytesIO(Path(file_input).read_bytes())
        elif isinstance(file_input, bytes):
            pdf_stream = io.BytesIO(file_input)
        else:
            raise TypeError("Unsupported file input type for PDF parsing.")

        reader = pypdf.PdfReader(pdf_stream)
        pages = []
        for page_idx, page in enumerate(reader.pages, start=1):
            extracted = page.extract_text() or ""
            pages.append(ParsedPage(page_number=page_idx, text=extracted))

        return ParsedDocument(
            filename=filename,
            pages=pages,
            metadata={
                "format": "pdf",
                "total_pages": str(len(pages)),
                "title": str(reader.metadata.title if reader.metadata else ""),
            },
        )
