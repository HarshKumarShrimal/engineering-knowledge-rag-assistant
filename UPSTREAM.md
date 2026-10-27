# Upstream Attribution & Engineering Contributions

## Upstream Project Details
- **Upstream Repository:** [roman-nalapko/ai-rag-platform](https://github.com/roman-nalapko/ai-rag-platform)
- **Original Author:** Roman Nalapko (@roman-nalapko)
- **Upstream License:** MIT License (retained in `LICENSE`)

---

## Retained Upstream Architectural Concepts
- Multi-channel retrieval orchestration and Reciprocal Rank Fusion (`RRF`) ranking formula:
  $$\text{RRF\_score}(d) = \sum_{m} \frac{1}{k + \text{rank}_m(d)}$$
- FastAPI REST backend routing principles.
- Containerized Docker deployment options for production environments.

---

## Substantive Engineering Contributions by Harsh Kumar Shrimal

1. **Local-First, Self-Contained Architecture:**
   - Redesigned the retrieval architecture so it executes seamlessly local-first without requiring external PostgreSQL, Qdrant clusters, or paid OpenAI API subscriptions.
   - Built an in-memory/persistent sparse BM25 engine (`rank-bm25`) paired with sub-linear TF-IDF semantic vector similarity.

2. **Domain Engineering Document Parsing & Chunking (`src/rag_assistant/document_parser.py`, `chunker.py`):**
   - Implemented multi-format parsing for technical PDFs (`pypdf`), engineering standards, and operations manuals.
   - Preserves page numbers, section headers, and character boundary metadata during sliding window chunking.

3. **Hybrid Search & Term-Boost Reranker (`src/rag_assistant/retrieval.py`):**
   - Engineered the `HybridSearchEngine` allowing consistent head-to-head benchmarking of `dense`, `lexical`, `hybrid` (RRF $k=60$), and `reranked` modes.
   - Built a cross-term precision reranker giving exact domain acronym boosts (e.g. TIT, NPSH, API 682).

4. **Evidence Grounding & Formal Abstention (`src/rag_assistant/generator.py`):**
   - Built strict evidence verification: if retrieved passages lack necessary technical keyword density or fall below relevance thresholds, the system formally abstains:
     `"I cannot find sufficient evidence in the uploaded documents to answer this question."`
   - Extracts verbatim supporting evidence sentences and produces granular citation metadata (`filename`, `page_number`, `chunk_id`, `snippet`).

5. **Security Hardening & Prompt Injection Defense:**
   - Implemented untrusted document text isolation and regex sanitization to neutralize adversarial jailbreak attempts embedded within uploaded document texts (e.g. "Ignore previous instructions").
   - Added path traversal protection on file uploads and payload size caps.

6. **Repeatable Retrieval Evaluation Suite (`src/rag_assistant/evaluation.py`, `scripts/evaluate_retrieval.py`):**
   - Built an automated evaluation harness measuring Recall@K, MRR@K, nDCG@K, citation validity rate, abstention accuracy, and latency across technical queries.
   - Generated and persisted benchmark outputs in `results/retrieval_benchmark.csv`.

7. **Interactive Dashboard UI & CI Automation:**
   - Developed an embedded web dashboard with real-time citation cards and mode switching.
   - Added full GitHub Actions CI workflow in `.github/workflows/ci.yml`.
