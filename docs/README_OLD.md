# PDF/OCR Processing for Workflow Digitalization
## Technical Requirements - Course Project Edition

**Scope:** Small-scale implementation for academic course  

---

## 1.  Model Recommendations

### 1.1 Document Understanding - Lightweight Options

#### Option A: SmolDocling (Recommended for Learning)
```python
# Can run on CPU, ~500MB VRAM if GPU available
# Model size: ~500MB download
pip install docling docling-core transformers torch
```

| Aspect | Specification |
|--------|---------------|
| **Model** | SmolDocling-256M-preview |
| **Size** | 256M parameters (~500MB) |
| **CPU Speed** | 3-5 sec/page |
| **GPU Speed** | 0.3-0.5 sec/page |
| **Capabilities** | OCR, layout, tables, formulas |

#### Option B: Docling Library with Traditional Pipeline
```python
# Uses ensemble of smaller models - more CPU friendly
pip install docling
```
- Uses EasyOCR/Tesseract for OCR
- TableTransformer for tables
- Lower VRAM requirements
- More modular/debuggable

#### Option C: Tesseract + LayoutParser (Most Lightweight)
```python
# Fallback for very constrained hardware
pip install pytesseract layoutparser pdf2image
```
- Pure CPU operation
- ~1-2 sec/page for OCR only
- Requires separate layout model

### 1.2 LLM for Triplet Extraction

#### Local LLM Options (Free, Private)

| Model | Size | RAM Required | Quality | Speed (CPU) |
|-------|------|--------------|---------|-------------|
| **Llama 3.2 3B** | 2GB | 6GB | Good | 5-10 tok/s |
| **Phi-3 Mini 3.8B** | 2.3GB | 8GB | Good | 5-8 tok/s |
| **Gemma 2 2B** | 1.5GB | 4GB | Moderate | 8-12 tok/s |
| **Qwen2.5 1.5B** | 1GB | 4GB | Moderate | 10-15 tok/s |
| **TinyLlama 1.1B** | 700MB | 3GB | Basic | 15-20 tok/s |

**Recommended Setup:**
```bash
# Install Ollama for easy local LLM management
curl -fsSL https://ollama.com/install.sh | sh

# Pull a lightweight model
ollama pull llama3.2:3b      # Best quality/size balance
ollama pull phi3:mini        # Good for constrained RAM
ollama pull gemma2:2b        # Google's efficient model
```

#### API Options (Faster, Costs Money)

| Provider | Model | Cost | Free Tier |
|----------|-------|------|-----------|
| **OpenAI** | GPT-4o-mini | $0.15/1M input | $5 credit |
| **Anthropic** | Claude 3 Haiku | $0.25/1M input | Limited |
| **Google** | Gemini 2.5 Flash | $0.075/1M input | Generous free tier |
| **Groq** | Llama 3.1 70B | Free (rate limited) | Yes! |
| **Together AI** | Various | $0.20/1M input | $25 credit |

**Best Free Option:** Groq API provides free access to Llama 3.1 70B with excellent speed.

### 1.3 Embedding Models (for Vector Search)

| Model | Dimensions | Size | RAM |
|-------|------------|------|-----|
| **all-MiniLM-L6-v2** | 384 | 80MB | 200MB |
| **BAAI/bge-small-en-v1.5** | 384 | 130MB | 300MB |
| **nomic-embed-text-v1** | 768 | 270MB | 500MB |

```python
# Recommended for laptops
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, small
```

---

## 2. Scaled-Down Dataset Recommendations

### 2.1 Document Layout Datasets

| Dataset | Full Size | Course Subset | Download |
|---------|-----------|---------------|----------|
| **DocLayNet** | 80K pages | 500-1000 pages | HuggingFace |
| **PubLayNet** | 360K pages | 1000 pages | Sample available |
| **DocBank** | 500K pages | 500 pages | Sample |

**Recommended Approach:**
```python
from datasets import load_dataset

# Load only a small subset
dataset = load_dataset("ds4sd/DocLayNet", split="train[:500]")

# Or filter by category for focused experiments
legal_docs = dataset.filter(lambda x: x['category'] == 'laws_and_regulations')
```

### 2.2 Relation Extraction Datasets

| Dataset | Full Size | Course Subset | Why This Size |
|---------|-----------|---------------|---------------|
| **WebNLG** | 25K samples | 1000 samples | Quick experiments |
| **NYT** | 70K sentences | 2000 sentences | Reasonable training |
| **SciERC** | 500 abstracts | 100 abstracts | Small but complete |
| **DocRED** | 5K documents | 200 documents | Document-level RE |

### 2.3 Legal Domain (Smaller Sets)

| Dataset | Subset Recommendation | Use Case |
|---------|----------------------|----------|
| **CUAD** | 50 contracts | Contract clause extraction |
| **LEDGAR** | 5000 provisions | Clause classification |
| **SEC EDGAR** | 20-50 filings | Real-world testing |

### 2.4 Creating Your Own Test Set

For better learning, create a small annotated dataset:
- **10-20 PDF documents** from public sources
- **Manual annotation** of 50-100 triplets
- Use for validation and error analysis

**Good Sources for Free PDFs:**
- arXiv papers (scientific)
- SEC EDGAR filings (legal/financial)
- Government reports (regulations)
- Wikipedia PDF exports (general)

---

## 3. Architecture

### 3.1 Course Project Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LOCAL DEVELOPMENT SETUP                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │  PDF Input   │────▶│  SmolDocling │────▶│   DocTags    │    │
│  │  (10-50 docs)│     │  (Local CPU) │     │   Output     │    │
│  └──────────────┘     └──────────────┘     └──────┬───────┘    │
│                                                    │            │
│                                                    ▼            │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │   Ollama     │◀────│ Text Chunks  │◀────│   Parser     │    │
│  │  (Local LLM) │     │              │     │              │    │
│  └──────┬───────┘     └──────────────┘     └──────────────┘    │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │    SPO       │────▶│   Neo4j      │────▶│   PyVis      │    │
│  │  Triplets    │     │  (Local/Docker)    │   Graph UI   │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Storage Options (Replacing Cloud Neo4j)

| Option | Setup Complexity | RAM Usage | Best For |
|--------|------------------|-----------|----------|
| **Neo4j Desktop** | Medium | 1-2GB | Full graph features |
| **Neo4j Docker** | Low | 512MB-1GB | Reproducible setup |
| **NetworkX + JSON** | Very Low | 100MB | Prototyping only |
| **SQLite + JSON** | Low | 50MB | Simple persistence |

**Recommended: Neo4j Docker**
```bash
# Minimal Neo4j for course project
docker run -d \
  --name neo4j-course \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  -e NEO4J_PLUGINS='["apoc"]' \
  -e NEO4J_dbms_memory_heap_max__size=512m \
  neo4j:5-community
```

**Fallback: In-Memory with NetworkX**
```python
import networkx as nx
import json

# Simple graph for prototyping
G = nx.DiGraph()
G.add_edge("Entity1", "Entity2", relation="works_for")

# Save/load
nx.write_gml(G, "knowledge_graph.gml")
```

---

## 4. Simplified Software Stack

### 4.1 Minimal Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Core dependencies only
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only PyTorch
pip install transformers docling docling-core
pip install sentence-transformers
pip install neo4j networkx pyvis
pip install pandas numpy
pip install fastapi uvicorn  # Optional: for API
pip install jupyter  # For experimentation
```

### 4.2 requirements.txt

```
# Document Processing
docling>=2.0.0
docling-core>=2.0.0
transformers>=4.40.0
torch>=2.0.0
pdf2image>=1.16.0
Pillow>=10.0.0

# Knowledge Extraction (choose one LLM approach)
# Option A: Local LLM via Ollama (install Ollama separately)
ollama>=0.1.0

# Option B: API-based
openai>=1.0.0
# anthropic>=0.20.0

# Embeddings
sentence-transformers>=2.2.0

# Graph Storage
neo4j>=5.0.0
networkx>=3.0
pyvis>=0.3.0

# Utilities
pandas>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0

# Development
jupyter>=1.0.0
pytest>=7.0.0
```

### 4.3 Folder Structure

```
pdf-kg-project/
├── data/
│   ├── raw/              # Input PDFs (10-50 documents)
│   ├── processed/        # DocTags output
│   ├── triplets/         # Extracted JSON triplets
│   └── datasets/         # Downloaded dataset subsets
├── models/               # Local model caches
├── notebooks/
│   ├── 01_document_processing.ipynb
│   ├── 02_triplet_extraction.ipynb
│   ├── 03_graph_construction.ipynb
│   └── 04_visualization.ipynb
├── src/
│   ├── document_parser.py
│   ├── triplet_extractor.py
│   ├── graph_builder.py
│   └── visualizer.py
├── tests/
├── config.yaml
├── requirements.txt
└── README.md
```

---

## 5. Implementation Timeline

### 8-Week Course Project Schedule (can probably do this quicker tbh)

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | Setup | Environment, Neo4j, sample data |
| 2 | Document Processing | SmolDocling pipeline working |
| 3 | Text Extraction | Chunking, preprocessing complete |
| 4 | Triplet Extraction | LLM prompts, basic extraction |
| 5 | Entity Processing | Standardization, deduplication |
| 6 | Graph Construction | Neo4j import, basic queries |
| 7 | Visualization | PyVis graphs, basic UI |
| 8 | Demo & Report | Final presentation, documentation |

---

## 6. Cost Considerations

### Free Options (Recommended for Course)

| Component | Free Option |
|-----------|-------------|
| **Document OCR** | SmolDocling (local) |
| **LLM** | Ollama + Llama 3.2 3B (local) |
| **LLM API** | Groq API (free tier) |
| **Graph DB** | Neo4j Community (Docker) |
| **Embeddings** | all-MiniLM-L6-v2 (local) |
| **Visualization** | PyVis (open source) |

### Optional Paid Options (Better Quality)

| Service | Cost | Benefit |
|---------|------|---------|
| OpenAI API | ~$5-10 total | Better triplet quality |
| Neo4j Aura Free | $0 | Managed database |
| Google Colab Pro | $10/month | GPU access |

**Estimated Total Cost: $0-20 for entire project**

---

## 7. Simplified Code Examples

### 7.1 Document Processing with SmolDocling

```python
# document_parser.py
from docling.document_converter import DocumentConverter
from pathlib import Path

def process_pdf(pdf_path: str) -> dict:
    """Process a single PDF and extract structured content."""
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    
    # Export to different formats
    markdown = result.document.export_to_markdown()
    
    return {
        "text": markdown,
        "pages": len(result.document.pages),
        "tables": len(result.document.tables),
        "path": pdf_path
    }

# Usage
doc = process_pdf("data/raw/sample_contract.pdf")
print(f"Extracted {len(doc['text'])} characters from {doc['pages']} pages")
```

### 7.2 Triplet Extraction with Ollama

```python
# triplet_extractor.py
import ollama
import json

EXTRACTION_PROMPT = """Extract factual relationships from this text as Subject-Predicate-Object triplets.

Rules:
1. Only extract relationships explicitly stated
2. Use simple, clear predicates (e.g., "works_for", "located_in", "founded")
3. Return valid JSON array

Text: {text}

Return ONLY a JSON array like:
[{{"subject": "Apple", "predicate": "founded_by", "object": "Steve Jobs"}}]
"""

def extract_triplets(text: str, model: str = "llama3.2:3b") -> list:
    """Extract SPO triplets using local Ollama model."""
    response = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": EXTRACTION_PROMPT.format(text=text[:2000])  # Limit context
        }]
    )
    
    try:
        # Parse JSON from response
        content = response['message']['content']
        # Find JSON array in response
        start = content.find('[')
        end = content.rfind(']') + 1
        if start != -1 and end > start:
            return json.loads(content[start:end])
    except json.JSONDecodeError:
        pass
    return []

# Usage
text = "Apple was founded by Steve Jobs in Cupertino, California."
triplets = extract_triplets(text)
# [{"subject": "Apple", "predicate": "founded_by", "object": "Steve Jobs"},
#  {"subject": "Apple", "predicate": "located_in", "object": "Cupertino, California"}]
```

### 7.3 Simple Graph Builder

```python
# graph_builder.py
from neo4j import GraphDatabase
import networkx as nx
from pyvis.network import Network

class KnowledgeGraph:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password123"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.nx_graph = nx.DiGraph()  # Backup in-memory graph
    
    def add_triplet(self, subject: str, predicate: str, obj: str, source: str = None):
        """Add a triplet to both Neo4j and NetworkX."""
        # Neo4j
        with self.driver.session() as session:
            session.run("""
                MERGE (s:Entity {name: $subject})
                MERGE (o:Entity {name: $object})
                MERGE (s)-[r:RELATES_TO {type: $predicate, source: $source}]->(o)
            """, subject=subject, object=obj, predicate=predicate, source=source)
        
        # NetworkX backup
        self.nx_graph.add_edge(subject, obj, relation=predicate)
    
    def add_triplets_batch(self, triplets: list, source: str = None):
        """Add multiple triplets efficiently."""
        for t in triplets:
            self.add_triplet(t['subject'], t['predicate'], t['object'], source)
    
    def visualize(self, output_path: str = "knowledge_graph.html"):
        """Generate interactive visualization."""
        net = Network(height="600px", width="100%", directed=True)
        net.from_nx(self.nx_graph)
        net.show_buttons(filter_=['physics'])
        net.save_graph(output_path)
        return output_path
    
    def get_stats(self) -> dict:
        """Get graph statistics."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n) WITH count(n) as nodes
                MATCH ()-[r]->() WITH nodes, count(r) as rels
                RETURN nodes, rels
            """)
            record = result.single()
            return {"nodes": record["nodes"], "relationships": record["rels"]}

# Usage
kg = KnowledgeGraph()
kg.add_triplets_batch(triplets, source="sample_contract.pdf")
kg.visualize("output/graph.html")
print(kg.get_stats())
```

### 7.4 Complete Pipeline Script

```python
# main.py - Simple end-to-end pipeline
from pathlib import Path
from document_parser import process_pdf
from triplet_extractor import extract_triplets
from graph_builder import KnowledgeGraph

def process_document_to_graph(pdf_path: str, kg: KnowledgeGraph):
    """Process a single document and add to knowledge graph."""
    print(f"Processing: {pdf_path}")
    
    # Step 1: Extract text
    doc = process_pdf(pdf_path)
    print(f"  Extracted {len(doc['text'])} chars")
    
    # Step 2: Chunk text (simple approach)
    chunk_size = 500
    chunks = [doc['text'][i:i+chunk_size] 
              for i in range(0, len(doc['text']), chunk_size)]
    print(f"  Created {len(chunks)} chunks")
    
    # Step 3: Extract triplets from each chunk
    all_triplets = []
    for i, chunk in enumerate(chunks[:10]):  # Limit for speed
        triplets = extract_triplets(chunk)
        all_triplets.extend(triplets)
        print(f"  Chunk {i+1}: {len(triplets)} triplets")
    
    # Step 4: Add to graph
    kg.add_triplets_batch(all_triplets, source=pdf_path)
    print(f"  Total triplets: {len(all_triplets)}")
    
    return all_triplets

def main():
    # Initialize graph
    kg = KnowledgeGraph()
    
    # Process all PDFs in data/raw
    pdf_dir = Path("data/raw")
    for pdf_path in pdf_dir.glob("*.pdf"):
        process_document_to_graph(str(pdf_path), kg)
    
    # Generate visualization
    kg.visualize("output/knowledge_graph.html")
    
    # Print stats
    stats = kg.get_stats()
    print(f"\nFinal Graph: {stats['nodes']} nodes, {stats['relationships']} relationships")

if __name__ == "__main__":
    main()
```

---

## 8. Testing with Small Datasets

### Quick Start Test Set

Create a minimal test with 5 documents:

```bash
# Download sample PDFs
mkdir -p data/raw

# Option 1: SEC EDGAR (financial/legal)
curl -o data/raw/apple_10k.pdf "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm"

# Option 2: arXiv papers (scientific)
curl -o data/raw/attention_paper.pdf "https://arxiv.org/pdf/1706.03762.pdf"

# Option 3: Use provided samples from DocLayNet
python -c "
from datasets import load_dataset
ds = load_dataset('ds4sd/DocLayNet', split='test[:5]')
for i, sample in enumerate(ds):
    # Save sample images/data
    print(f'Sample {i}: {sample}')
"
```

### Validation Script

```python
# validate_triplets.py
def validate_triplet(triplet: dict) -> bool:
    """Basic triplet validation."""
    required = ['subject', 'predicate', 'object']
    if not all(k in triplet for k in required):
        return False
    if not all(isinstance(triplet[k], str) and len(triplet[k]) > 0 for k in required):
        return False
    return True

def calculate_metrics(predicted: list, gold: list) -> dict:
    """Calculate precision, recall, F1 for triplet extraction."""
    pred_set = {(t['subject'], t['predicate'], t['object']) for t in predicted}
    gold_set = {(t['subject'], t['predicate'], t['object']) for t in gold}
    
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1": f1}
```

---

## 9. Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| **Out of memory** | Use smaller model (Gemma 2B), reduce chunk size |
| **Slow inference** | Use Groq API (free), or quantized models |
| **Neo4j won't start** | Reduce heap size, check Docker memory |
| **Poor triplet quality** | Improve prompts, use larger LLM for validation |
| **PDF parsing fails** | Try Tesseract fallback, check PDF quality |

---

## Summary of Key Changes from Enterprise Version

| Aspect | Enterprise | Course Project |
|--------|------------|----------------|
| **Hardware** | GPU servers | Laptops (CPU) |
| **Documents** | Millions | 20-50 |
| **Dataset size** | Full datasets | 500-2000 samples |
| **LLM** | GPT-4 / Claude | Llama 3.2 3B (local) |
| **Database** | Neo4j Aura | Neo4j Docker/Community |
| **Processing** | Distributed | Single machine |
| **Budget** | $5,000+/month | $0-20 total |
| **Timeline** | 20 weeks | 8 weeks |

This scaled-down version maintains all the core learning objectives while being practical for student laptops and course timelines.
