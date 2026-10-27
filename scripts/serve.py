#!/usr/bin/env python3
"""Run the Engineering Document Assistant RAG FastAPI service and Dashboard."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Start the Engineering Knowledge RAG service.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8002, help="Port to listen on.")
    parser.add_argument("--reload", action="store_true", help="Enable reload.")
    args = parser.parse_args()

    print(f"Starting Engineering Knowledge Assistant on http://{args.host}:{args.port}")
    uvicorn.run("rag_assistant.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
