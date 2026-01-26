# Legal Question Answering System
## Course Project: PDF/OCR Processing for Workflow Digitalization

**Project Type:** GraphRAG-based Legal QA System  
**Scope:** Course project on local laptops  
**Team Size:** 2-4 students  
**Timeline:** 6-8 weeks

---

## 1. Project Overview

### 1.1 What We're Building

A **Legal Question Answering System** that allows users to ask natural language questions about legal contracts and cases, with answers grounded in a custom knowledge base built from legal PDF documents.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LEGAL QA SYSTEM                              │
│                                                                      │
│   User: "What are the termination conditions in vendor contracts?"  │
│                                                                      │
│   System: "Based on 3 contracts in the knowledge base:              │
│   1. 30-day written notice required (ABC Corp, Section 4.2)         │
│   2. Immediate termination for material breach (XYZ Inc, Section 8) │
│   3. 60-day cure period for non-payment (Tech Services, Section 5)" │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Why This Approach (RAG + Knowledge Graph)

| Approach | Limitation |
|----------|------------|
| **LLM alone** | Doesn't know YOUR specific contracts; may hallucinate legal information |
| **Simple search** | Returns documents but doesn't answer questions |
| **LLM + Knowledge Base (RAG)** | Retrieves REAL data, then generates accurate, sourced answers |
| **GraphRAG (Our Approach)** | Additionally understands relationships between entities (parties, obligations, clauses) |

### 1.3 Core Capabilities

1. **Document Ingestion**: Parse legal PDFs and extract structured text
2. **Knowledge Extraction**: Extract entities and relationships as triplets
3. **Knowledge Storage**: Store in graph database + vector embeddings
4. **Intelligent Retrieval**: Find relevant information using hybrid search
5. **Answer Generation**: Generate accurate answers with source citations
6. **Chat Interface**: User-friendly Q&A interface

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION (One-time Setup)                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐    ┌─────────────┐    ┌──────────┐    ┌───────────┐  │
│  │  Legal   │    │ SmolDocling │    │  Text    │    │  Ollama   │  │
│  │  PDFs    │───▶│   (OCR +    │───▶│  Chunks  │───▶│ (Triplet  │  │
│  │  (CUAD)  │    │   Layout)   │    │          │    │ Extraction│  │
│  └──────────┘    └─────────────┘    └──────────┘    └─────┬─────┘  │
│                                                           │        │
│                         ┌─────────────────────────────────┘        │
│                         ▼                                          │
│           ┌─────────────────────────────────────────┐              │
│           │          KNOWLEDGE BASE                  │              │
│           │  ┌─────────────┐    ┌─────────────────┐ │              │
│           │  │   Neo4j     │    │   ChromaDB      │ │              │
│           │  │  (Graph:    │    │  (Vectors:      │ │              │
│           │  │  Triplets)  │    │   Embeddings)   │ │              │
│           │  └─────────────┘    └─────────────────┘ │              │
│           └─────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    QUERY TIME (User Interaction)                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐                                               │
│  │  User Question:  │                                               │
│  │  "What are the   │                                               │
│  │  liability caps?"│                                               │
│  └────────┬─────────┘                                               │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────┐    ┌──────────────────────────────────────┐   │
│  │ Embed Question  │───▶│  HYBRID RETRIEVAL                    │   │
│  │ (MiniLM)        │    │  • Vector search (similar chunks)    │   │
│  └─────────────────┘    │  • Graph search (related entities)   │   │
│                         └──────────────────┬───────────────────┘   │
│                                            │                        │
│                                            ▼                        │
│                         ┌──────────────────────────────────────┐   │
│                         │  RETRIEVED CONTEXT                    │   │
│                         │  • Relevant text chunks               │   │
│                         │  • Related triplets                   │   │
│                         │  • Source document info               │   │
│                         └──────────────────┬───────────────────┘   │
│                                            │                        │
│                                            ▼                        │
│                         ┌──────────────────────────────────────┐   │
│                         │  OLLAMA LLM (Llama 3.2)              │   │
│                         │  Generate answer using context        │   │
│                         └──────────────────┬───────────────────┘   │
│                                            │                        │
│                                            ▼                        │
│                         ┌──────────────────────────────────────┐   │
│                         │  ANSWER + SOURCES                     │   │
│                         │  "Liability is capped at $1M in       │   │
│                         │   ABC Contract (Section 5.2)..."      │   │
│                         └──────────────────────────────────────┘   │
│                                            │                        │
│                                            ▼                        │
│                         ┌──────────────────────────────────────┐   │
│                         │  STREAMLIT CHAT INTERFACE            │   │
│                         └──────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow Summary

| Stage | Input | Process | Output |
|-------|-------|---------|--------|
| **1. Parse** | Legal PDFs | SmolDocling OCR | Structured text + layout |
| **2. Chunk** | Full text | Split with overlap | 500-word chunks |
| **3. Extract** | Text chunks | Ollama triplet extraction | (Subject, Predicate, Object) |
| **4. Embed** | Text chunks | MiniLM embedding | 384-dim vectors |
| **5. Store** | Triplets + Vectors | Neo4j + ChromaDB | Knowledge base |
| **6. Query** | User question | Hybrid retrieval | Relevant context |
| **7. Generate** | Context + Question | Ollama LLM | Answer with sources |

---

## 3. Technology Stack

### 3.1 Components Overview

| Component | Tool | Purpose | Runs On |
|-----------|------|---------|---------|
| **PDF Parser** | Docling + SmolDocling | Extract text from legal PDFs | CPU (3-5 sec/page) |
| **Triplet Extractor** | Ollama (Llama 3.2 3B) | Extract relationships | CPU (local) |
| **Graph Database** | Neo4j Community | Store knowledge graph | Docker |
| **Vector Store** | ChromaDB | Store embeddings for search | Local (lightweight) |
| **Embedding Model** | all-MiniLM-L6-v2 | Convert text to vectors | CPU (fast) |
| **LLM for QA** | Ollama (Llama 3.2 3B) | Generate answers | CPU (local) |
| **Chat Interface** | Streamlit | User-friendly UI | Python |

### 3.2 Important: SmolDocling vs Ollama

**SmolDocling is NOT available on Ollama.** They serve different purposes:

| Tool | Type | Purpose |
|------|------|---------|
| **Ollama** | Text LLM runtime | Run Llama, Mistral, Phi (text-only models) |
| **SmolDocling** | Vision-Language Model | "See" documents as images, extract structure |

**SmolDocling requires the Docling library:**
```python
# Correct way to use SmolDocling
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("contract.pdf")
text = result.document.export_to_markdown()
```

### 3.3 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8 GB | 16 GB |
| **Storage** | 20 GB free | 50 GB free |
| **CPU** | 4 cores | 8 cores |
| **GPU** | Not required | Optional (speeds up inference) |
| **OS** | Windows/macOS/Linux | Linux or WSL2 recommended |

### 3.4 Software Dependencies

```txt
# requirements.txt

# Document Processing
docling>=2.0.0
docling-core>=2.0.0
transformers>=4.40.0
torch>=2.0.0
pdf2image>=1.16.0
Pillow>=10.0.0

# Knowledge Extraction & Storage
ollama>=0.1.0
neo4j>=5.0.0
chromadb>=0.4.0
sentence-transformers>=2.2.0

# Graph Processing
networkx>=3.0
pyvis>=0.3.0

# Utilities
pandas>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0

# Interface
streamlit>=1.30.0

# Development
jupyter>=1.0.0
python-dotenv>=1.0.0
```

### 3.5 Installation Commands

```bash
# 1. Create virtual environment
python -m venv legal_qa_env
source legal_qa_env/bin/activate  # Linux/Mac
# legal_qa_env\Scripts\activate   # Windows

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh  # Linux/Mac
# Windows: Download from https://ollama.com/download

# 4. Pull LLM model
ollama pull llama3.2:3b

# 5. Start Neo4j (Docker)
docker run -d \
  --name neo4j-legal \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  -e NEO4J_dbms_memory_heap_max__size=512m \
  neo4j:5-community

# 6. Verify installation
python -c "import docling; import chromadb; import ollama; print('All imports successful!')"
```

---

## 4. Dataset

### 4.1 Primary Dataset: CUAD (Contract Understanding Atticus Dataset)

| Attribute | Details |
|-----------|---------|
| **Size** | 510 contracts, 13,000+ annotations |
| **Format** | PDF + TXT + Expert Annotations |
| **Source** | SEC EDGAR filings |
| **License** | CC BY 4.0 (Free for any use) |
| **Labels** | 41 clause categories |

**Why CUAD is ideal:**
- Real commercial contracts (not synthetic)
- Expert annotations for clause types
- Includes both PDF and extracted text
- Perfect size for course project
- Well-documented with academic paper

### 4.2 What Are These Contracts?

Contracts are legally binding agreements between parties. CUAD includes:

| Contract Type | Description | Example Clauses |
|---------------|-------------|-----------------|
| License Agreement | Permission to use IP | Royalties, Territory, Exclusivity |
| Service Agreement | Services provided | Payment terms, SLAs, Deliverables |
| Employment Agreement | Executive employment | Compensation, Non-compete, Benefits |
| Joint Venture | Partnership between companies | Profit sharing, Governance, Exit terms |
| Supply Agreement | Goods provided | Pricing, Delivery, Quality standards |

### 4.3 Downloading the Dataset

```python
# Option A: HuggingFace (Recommended)
from datasets import load_dataset

# Load 50 contracts for course project
dataset = load_dataset(
    "dvgodoy/CUAD_v1_Contract_Understanding_PDF",
    split="train[:50]"
)

print(f"Loaded {len(dataset)} contracts")

# Save PDFs locally
import base64
import os

os.makedirs("data/pdfs", exist_ok=True)

for i, item in enumerate(dataset):
    pdf_bytes = base64.b64decode(item['pdf_bytes_base64'])
    filename = item['file_name'].replace('.txt', '.pdf')
    
    with open(f"data/pdfs/{filename}", "wb") as f:
        f.write(pdf_bytes)
    
    print(f"Saved: {filename}")
```

### 4.4 Supplementary Data Sources (Optional)

| Source | Content | Access | Best For |
|--------|---------|--------|----------|
| **SEC EDGAR** | Corporate filings | Free API (10 req/sec) | Additional contracts |
| **CourtListener** | Federal court docs | Free bulk download | Court cases |
| **LEDGAR** | 846K contract clauses | HuggingFace | Clause classification |

---

## 5. Implementation Guide

### 5.1 Project Structure

```
legal-qa-system/
│
├── data/
│   ├── pdfs/                    # Input legal PDFs
│   ├── processed/               # Extracted text
│   ├── triplets/                # Extracted triplets (JSON)
│   └── embeddings/              # Cached embeddings
│
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf_parser.py        # SmolDocling PDF processing
│   │   ├── chunker.py           # Text chunking
│   │   └── triplet_extractor.py # Ollama triplet extraction
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── graph_store.py       # Neo4j operations
│   │   └── vector_store.py      # ChromaDB operations
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── hybrid_retriever.py  # Combined retrieval
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   └── qa_generator.py      # Answer generation
│   │
│   └── interface/
│       ├── __init__.py
│       └── chat_app.py          # Streamlit interface
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_pdf_parsing.ipynb
│   ├── 03_triplet_extraction.ipynb
│   ├── 04_knowledge_graph.ipynb
│   └── 05_qa_testing.ipynb
│
├── config.yaml
├── requirements.txt
├── README.md
└── run_app.py
```

### 5.2 Core Implementation Code

#### 5.2.1 PDF Parser

```python
# src/ingestion/pdf_parser.py
from docling.document_converter import DocumentConverter
from pathlib import Path
import json

class PDFParser:
    def __init__(self):
        self.converter = DocumentConverter()
    
    def parse_pdf(self, pdf_path: str) -> dict:
        """Parse a single PDF and return structured content."""
        result = self.converter.convert(pdf_path)
        
        return {
            "file_name": Path(pdf_path).name,
            "text": result.document.export_to_markdown(),
            "num_pages": len(result.document.pages) if hasattr(result.document, 'pages') else None,
        }
    
    def parse_directory(self, pdf_dir: str, output_dir: str):
        """Parse all PDFs in a directory."""
        pdf_dir = Path(pdf_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for pdf_path in pdf_dir.glob("*.pdf"):
            print(f"Processing: {pdf_path.name}")
            try:
                parsed = self.parse_pdf(str(pdf_path))
                output_file = output_dir / f"{pdf_path.stem}.json"
                with open(output_file, "w") as f:
                    json.dump(parsed, f, indent=2)
                print(f"  ✓ Extracted {len(parsed['text'])} characters")
            except Exception as e:
                print(f"  ✗ Error: {e}")
```

#### 5.2.2 Text Chunker

```python
# src/ingestion/chunker.py
from typing import List

class TextChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, doc_id: str = None) -> List[dict]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_chunk_{len(chunks)}",
                "text": chunk_text,
                "start_word": i,
                "end_word": i + len(chunk_words)
            })
        
        return chunks
```

#### 5.2.3 Triplet Extractor

```python
# src/ingestion/triplet_extractor.py
import ollama
import json
from typing import List

class TripletExtractor:
    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model
        self.prompt_template = """Extract factual relationships from this legal contract text as Subject-Predicate-Object triplets.

Focus on:
- Parties involved (companies, individuals)
- Obligations and rights
- Payment terms and amounts
- Dates and durations
- Conditions and requirements

Text:
{text}

Return ONLY a valid JSON array of triplets:
[{{"subject": "ABC Corp", "predicate": "agrees_to_pay", "object": "XYZ Inc"}}]

JSON triplets:"""

    def extract_triplets(self, text: str) -> List[dict]:
        """Extract triplets from a single text chunk."""
        prompt = self.prompt_template.format(text=text[:2000])
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1}
            )
            
            content = response['message']['content']
            start = content.find('[')
            end = content.rfind(']') + 1
            
            if start != -1 and end > start:
                triplets = json.loads(content[start:end])
                return [t for t in triplets if self._validate_triplet(t)]
        except Exception as e:
            print(f"Extraction error: {e}")
        
        return []
    
    def _validate_triplet(self, triplet: dict) -> bool:
        """Validate triplet structure."""
        required = ['subject', 'predicate', 'object']
        return all(k in triplet and isinstance(triplet[k], str) and len(triplet[k]) > 0 for k in required)
```

#### 5.2.4 Graph Store

```python
# src/storage/graph_store.py
from neo4j import GraphDatabase
from typing import List

class GraphStore:
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 user: str = "neo4j", password: str = "password123"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def add_triplets_batch(self, triplets: List[dict]):
        """Add multiple triplets efficiently."""
        with self.driver.session() as session:
            session.run("""
                UNWIND $triplets AS t
                MERGE (s:Entity {name: t.subject})
                MERGE (o:Entity {name: t.object})
                MERGE (s)-[r:RELATES {type: t.predicate}]->(o)
                SET r.source_doc = t.source_doc
            """, triplets=triplets)
    
    def search_by_entity(self, entity_name: str, limit: int = 10) -> List[dict]:
        """Find all triplets involving an entity."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Entity)-[r:RELATES]->(o:Entity)
                WHERE toLower(s.name) CONTAINS toLower($entity)
                   OR toLower(o.name) CONTAINS toLower($entity)
                RETURN s.name AS subject, r.type AS predicate, 
                       o.name AS object, r.source_doc AS source
                LIMIT $limit
            """, entity=entity_name, limit=limit)
            return [dict(record) for record in result]
    
    def get_stats(self) -> dict:
        """Get graph statistics."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n:Entity) WITH count(n) AS nodes
                MATCH ()-[r:RELATES]->() WITH nodes, count(r) AS relationships
                RETURN nodes, relationships
            """)
            record = result.single()
            return {"nodes": record["nodes"], "relationships": record["relationships"]}
```

#### 5.2.5 Vector Store

```python
# src/storage/vector_store.py
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List

class VectorStore:
    def __init__(self, collection_name: str = "legal_docs", 
                 persist_dir: str = "./data/chroma"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def add_chunks(self, chunks: List[dict]):
        """Add text chunks with embeddings."""
        texts = [c['text'] for c in chunks]
        ids = [c['chunk_id'] for c in chunks]
        metadatas = [{"doc_id": c['doc_id']} for c in chunks]
        
        embeddings = self.embedder.encode(texts).tolist()
        
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
    
    def search(self, query: str, n_results: int = 5) -> List[dict]:
        """Search for similar chunks."""
        query_embedding = self.embedder.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        return [
            {
                "chunk_id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "doc_id": results['metadatas'][0][i]['doc_id'],
                "score": 1 - results['distances'][0][i]
            }
            for i in range(len(results['ids'][0]))
        ]
    
    def get_stats(self) -> dict:
        return {"total_chunks": self.collection.count()}
```

#### 5.2.6 QA Generator

```python
# src/generation/qa_generator.py
import ollama
from typing import List

class QAGenerator:
    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model
        self.system_prompt = """You are a legal assistant that answers questions based ONLY on the provided context.

Rules:
1. Only use information from the provided context
2. If the context doesn't contain the answer, say "I don't have information about that"
3. Always cite the source document when possible
4. Be precise and professional"""

    def generate_answer(self, question: str, context_chunks: List[dict], 
                        graph_context: List[dict] = None) -> str:
        """Generate an answer using retrieved context."""
        
        # Format text context
        text_context = "\n\n".join([
            f"[Source: {c['doc_id']}]\n{c['text']}" 
            for c in context_chunks
        ])
        
        # Format graph context
        graph_context_str = ""
        if graph_context:
            graph_context_str = "\n\nKnowledge Graph Facts:\n" + "\n".join([
                f"- {t['subject']} → {t['predicate']} → {t['object']}"
                for t in graph_context
            ])
        
        user_prompt = f"""Context from legal documents:
{text_context}
{graph_context_str}

Question: {question}

Answer with source citations:"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": 0.3}
        )
        
        return response['message']['content']
```

#### 5.2.7 Streamlit Chat Interface

```python
# src/interface/chat_app.py
import streamlit as st
from src.storage.vector_store import VectorStore
from src.storage.graph_store import GraphStore
from src.generation.qa_generator import QAGenerator

@st.cache_resource
def init_components():
    return {
        "vector_store": VectorStore(),
        "graph_store": GraphStore(),
        "qa_generator": QAGenerator()
    }

def main():
    st.set_page_config(page_title="Legal QA System", page_icon="⚖️")
    st.title("⚖️ Legal Question Answering System")
    st.caption("Ask questions about contracts in the knowledge base")
    
    components = init_components()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if question := st.chat_input("Ask a legal question..."):
        st.chat_message("user").markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                # Retrieve context
                vector_results = components["vector_store"].search(question, n_results=5)
                
                key_terms = question.split()[:3]
                graph_results = []
                for term in key_terms:
                    graph_results.extend(
                        components["graph_store"].search_by_entity(term, limit=3)
                    )
                
                # Generate answer
                answer = components["qa_generator"].generate_answer(
                    question, vector_results, graph_results
                )
                
                st.markdown(answer)
                
                with st.expander("📚 Sources"):
                    for r in vector_results[:3]:
                        st.markdown(f"**{r['doc_id']}** (score: {r['score']:.2f})")
                        st.text(r['text'][:200] + "...")
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
    
    with st.sidebar:
        st.header("📊 Knowledge Base Stats")
        vector_stats = components["vector_store"].get_stats()
        graph_stats = components["graph_store"].get_stats()
        st.metric("Document Chunks", vector_stats['total_chunks'])
        st.metric("Graph Nodes", graph_stats['nodes'])
        st.metric("Graph Relationships", graph_stats['relationships'])

if __name__ == "__main__":
    main()
```

---

## 6. Implementation Timeline

### 6-Week Schedule

| Week | Phase | Tasks | Deliverables |
|------|-------|-------|--------------|
| **1** | Setup & Data | Environment setup, download 50 CUAD contracts | Working environment |
| **2** | PDF Processing | Implement parser, chunk documents | Extracted text files |
| **3** | Knowledge Extraction | Triplet extraction with Ollama | JSON triplet files |
| **4** | Storage | Set up Neo4j + ChromaDB, load data | Populated knowledge base |
| **5** | QA System | Implement retrieval + generation | Working Q&A |
| **6** | Interface & Demo | Build Streamlit UI, prepare demo | Final application |

### Milestone Checkpoints

**Week 2:** Parse 5 PDFs successfully  
**Week 4:** 500+ triplets extracted, graph populated  
**Week 6:** End-to-end Q&A working, demo ready  

---

## 7. Example Use Cases

### Sample Questions

| Question | Expected Behavior |
|----------|-------------------|
| "What are the termination conditions?" | Search termination clauses, summarize |
| "Which contracts have liability caps?" | Query graph for liability amounts |
| "What obligations does the Provider have?" | Find triplets where Provider is subject |
| "Show me confidentiality clauses" | Retrieve relevant chunks, cite sources |

### Sample Interaction

```
User: What happens if a party breaches confidentiality?

System: Based on the contracts in the knowledge base:

1. **Injunctive relief** - The disclosing party may seek immediate 
   court action (ABC Corp Agreement, Section 7.2)

2. **Termination rights** - The non-breaching party may terminate 
   immediately (XYZ License, Section 12.1)

3. **Liability for damages** - The breaching party is liable for 
   actual damages (Tech Services Contract, Section 8.4)

Sources: ABC Corp Agreement, XYZ License, Tech Services Contract
```

---

## 8. Evaluation Metrics

| Metric | Target |
|--------|--------|
| **PDFs processed** | 20-50 contracts |
| **Triplets extracted** | 500+ triplets |
| **Graph nodes** | 200+ entities |
| **Answer relevance** | 7/10 rated helpful |
| **Response time** | <10 seconds |

---

## 9. Quick Start Commands

```bash
# 1. Setup
mkdir legal-qa-system && cd legal-qa-system
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Start Neo4j
docker run -d --name neo4j-legal -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 neo4j:5-community

# 3. Pull Ollama model
ollama pull llama3.2:3b

# 4. Download dataset (run Python)
# python download_cuad.py

# 5. Run pipeline
python -m src.ingestion.pdf_parser
python -m src.ingestion.triplet_extractor

# 6. Launch interface
streamlit run src/interface/chat_app.py
```

---

## 10. References

**Dataset:**
- CUAD Paper: https://arxiv.org/abs/2103.06268
- CUAD Data: https://huggingface.co/datasets/dvgodoy/CUAD_v1_Contract_Understanding_PDF
- Atticus Project: https://www.atticusprojectai.org/cuad

**Tools:**
- Docling: https://github.com/DS4SD/docling
- Ollama: https://ollama.com
- Neo4j: https://neo4j.com
- ChromaDB: https://www.trychroma.com
- Streamlit: https://streamlit.io

---

## Summary

This project builds a **Legal QA System** that:

1. **Ingests** legal PDFs using SmolDocling
2. **Extracts** knowledge as triplets using Ollama
3. **Stores** in Neo4j (graph) + ChromaDB (vectors)
4. **Retrieves** using hybrid search
5. **Generates** accurate, sourced answers
6. **Presents** through a Streamlit chat interface

Users can ask natural language questions about legal contracts and receive accurate, cited answers from the custom knowledge base.


---

## 11. Contributing

1. **Clone** the repository to your local system.
2. **Create a new branch** for your feature or bugfix. Use `git checkout -b feature-name`.
3. **Make your changes** and ensure all tests pass.
4. **Commit your changes** with clear messages. Use `git commit -m "Description of changes"`.
5. **Push your branch** to the remote repository. Use `git push origin feature-name`.
6. **Open a Pull Request** on GitHub for review.
