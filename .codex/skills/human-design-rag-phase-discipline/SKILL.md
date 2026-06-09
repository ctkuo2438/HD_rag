---
name: human-design-rag-phase-discipline
description: |
  Use when planning or implementing any Human Design RAG feature. Phase 1 is local,
  text-only knowledge base construction from PDFs. Use this skill to prevent scope
  creep into AWS, web apps, SQL, vision, OCR, cloud vector stores, or later phases.
---

# Human Design RAG phase discipline

## Purpose

Use this skill to prevent scope creep.

The current project phase is Phase 1:
local-only, text-only knowledge base construction from Human Design PDF books.

## Phase 1 scope

Phase 1 is ONLY local text-based knowledge base construction.

In scope:
- local PDFs
- LlamaIndex PyMuPDFReader
- embedded PDF text extraction only
- SentenceSplitter(chunk_size=800, chunk_overlap=80)
- OpenAIEmbedding(text-embedding-3-small)
- Chroma persistent local vector store
- ingestion script
- retrieval smoke-test script
- tests with mocked or fake embeddings where possible

## Explicitly out of scope for Phase 1

Do NOT add:
- AWS
- S3
- Lambda
- API Gateway
- EC2
- SageMaker
- Pinecone
- OpenSearch
- LangChain
- SQL database
- user accounts
- Streamlit
- FastAPI
- web app
- Claude reading generation
- Vision extraction
- BodyGraph image interpretation
- OCR
- LlamaParse
- Tree Index
- SQL agent
- RAG response generation

## Decision protocol

When asked to add a feature:

1. Check whether it is inside Phase 1 scope.
2. If yes, implement or plan it.
3. If no, do not implement it.
4. If it belongs to a future phase, mention it as a future phase note only when useful.
5. Do not add implementation tasks for future-phase features.
6. If the request is ambiguous, ask before broadening scope.

## Common mistakes to reject

Reject these unless the user explicitly changes Phase 1 scope:

- "Let's also add Streamlit"
- "Let's upload PDFs to S3"
- "Let's use Pinecone now"
- "Let's parse BodyGraph images inside PDFs"
- "Let's add SQL user history"
- "Let's add LangChain just because"
- "Let's call Claude to answer questions"
- "Let's add OCR for image-heavy pages"
- "Let's add LlamaParse now"

## Current Phase 1 architecture

Local PDFs
   ↓
LlamaIndex PyMuPDFReader
   ↓
Extract embedded PDF text only
   ↓
SentenceSplitter(chunk_size=800, chunk_overlap=80)
   ↓
OpenAIEmbedding(text-embedding-3-small)
   ↓
Chroma persistent vector store
