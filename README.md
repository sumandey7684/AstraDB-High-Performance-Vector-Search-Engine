# AstraDB — High-Performance Vector Search Engine

AstraDB is a custom high-performance vector database and semantic retrieval engine built from scratch in C++. It implements HNSW-based Approximate Nearest Neighbor (ANN) indexing, KD-Tree search, and brute-force vector retrieval to power semantic search, AI retrieval systems, and Retrieval-Augmented Generation (RAG) pipelines.

The project demonstrates how modern AI retrieval systems internally perform high-dimensional semantic search, vector indexing, nearest-neighbor traversal, and contextual document retrieval for AI-powered applications.

---

# What is AstraDB?

AstraDB is a high-performance vector search engine designed for semantic retrieval and Retrieval-Augmented Generation (RAG) systems.

It supports:

- Graph-based ANN indexing
- Vector similarity search
- Real-time benchmarking
- PCA-based vector visualization
- Local LLM-powered querying using Ollama
- AI document retrieval pipelines

The system is built entirely from scratch in C++ to deeply understand the internal architecture behind modern vector databases.

---

# Why AstraDB?

Modern AI systems like ChatGPT, recommendation engines, semantic search systems, and RAG applications heavily depend on vector databases for efficient retrieval from high-dimensional embeddings.

Instead of relying on external vector database APIs, AstraDB was engineered to replicate and understand the internal infrastructure behind platforms like:

- Pinecone
- Weaviate
- Milvus
- ChromaDB

The project focuses on:

- Approximate Nearest Neighbor (ANN) search
- Graph-based indexing systems
- High-dimensional vector retrieval
- Semantic similarity search
- AI retrieval infrastructure
- Local Retrieval-Augmented Generation (RAG)

---

# How AstraDB Works

Text data is converted into vector embeddings using local embedding models via Ollama.

These embeddings are indexed using HNSW graphs for efficient Approximate Nearest Neighbor retrieval.

## Retrieval Pipeline

1. Query text is converted into an embedding vector
2. HNSW performs nearest-neighbor traversal
3. Relevant vectors or document chunks are retrieved
4. Retrieved context is passed into a local LLM
5. The LLM generates a contextual AI response

---

# Core Features

- HNSW implementation from scratch for fast ANN retrieval
- KD-Tree and Brute Force search comparison
- Cosine, Euclidean, and Manhattan similarity metrics
- Real-time vector benchmarking and latency analysis
- PCA-based semantic space visualization
- Local embedding generation using Ollama
- Retrieval-Augmented Generation (RAG) pipeline
- REST API for insertion, retrieval, benchmarking, and querying
- Interactive web UI for semantic search exploration

---

# Technical Highlights

- Built a high-performance vector database engine in C++ implementing HNSW, KD-Tree, and brute-force ANN search for semantic retrieval systems.

- Engineered a custom Approximate Nearest Neighbor indexing pipeline capable of fast high-dimensional vector search using graph-based traversal and similarity metrics.

- Designed an end-to-end RAG pipeline integrating embeddings, semantic retrieval, and local LLM-powered contextual querying using Ollama.

- Implemented benchmarking tools to compare ANN retrieval performance across multiple search strategies.

- Developed PCA-based visualization for understanding vector clustering and semantic relationships.

---

# Tech Stack

| Category | Technologies |
|---|---|
| Language | C++ |
| Vector Indexing | HNSW Graph |
| Search Algorithms | KD-Tree, Brute Force |
| API Layer | REST API |
| AI Infrastructure | Ollama |
| Embeddings | Semantic Embeddings |
| Visualization | PCA Visualization |
| AI Pipeline | RAG Architecture |

---

# Project Structure

```text
AstraDB/
├── main.cpp        # Vector engine, ANN indexing, REST API, RAG pipeline
├── httplib.h       # HTTP server library
├── index.html      # Interactive visualization dashboard
└── README.md
```

---

# Local Setup

## Prerequisites

Install the following dependencies:

- MSYS2 / g++
- Git
- Ollama

Install required Ollama models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

---

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/AstraDB.git
cd AstraDB
```

---

## Compile

```bash
g++ -std=c++17 -O2 main.cpp -o db -lws2_32
```

---

## Run AstraDB

Start Ollama:

```bash
ollama serve
```

Run AstraDB:

```bash
./db
```

Open in browser:

```text
http://localhost:8080
```

---

# Search Architecture

```text
Text Input
     ↓
Embedding Model
     ↓
Vector Representation
     ↓
HNSW ANN Index
     ↓
Nearest Neighbor Retrieval
     ↓
Retrieved Context
     ↓
LLM Response Generation
```

---

# Use Cases

- Semantic Search Engines
- AI Knowledge Retrieval Systems
- Retrieval-Augmented Generation (RAG)
- AI Chatbots
- Recommendation Systems
- Local LLM Applications
- Document Retrieval Systems
- Vector Similarity Analysis

---

# Future Improvements

- Persistent disk-based vector storage
- Distributed vector indexing
- GPU-accelerated ANN search
- Hybrid keyword + vector search
- Multi-modal embedding support
- Scalable sharding architecture
- Incremental HNSW updates
- Authentication and API security

---

# License

Copyright © 2026 Suman Dey. All rights reserved.

This repository is provided strictly for portfolio and educational viewing purposes only.

No permission is granted to use, copy, modify, distribute, sublicense, or commercialize any part of this codebase without explicit written permission from the author.
