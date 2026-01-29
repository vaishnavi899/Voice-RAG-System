# Zero-Latency Voice RAG System

A high-performance Retrieval-Augmented Generation (RAG) system designed for voice-based technical support queries. Built to achieve sub-800ms Time to First Byte (TTFB) for real-time voice applications.

## Installation

### Prerequisites
- Python 3.11 or higher
- Google AI API key

### Setup

1. Clone the repository:
git clone <repository-url>
cd VoiceAgent

2. Install dependencies:
pip install -r requirements.txt

3. Create a `.env` file in the project root:
GOOGLE_API_KEY=your_api_key_here

4. Add your technical manual:
```bash
mkdir -p data
# Place your PDF file as data/Manual.pdf
```

## Usage

### Step 1: Index Your Document

Run the ingestion script to process and index your technical manual:

```bash
python ingest.py
```

This will:
- Load the PDF document
- Split it into chunks
- Generate embeddings
- Create a FAISS vector index
- Save everything to `cisco_7604_index/`

Expected output:
```
Ingestion complete! Created 3052 chunks in cisco_7604_index
```

### Step 2: Run the Voice RAG System

```bash
python main.py
```

Example interaction:
```
USER: And what about the second slot?
AI: Let me check that for you...
AI: Say: show module 
This shows all your modules
Find your Sup 720 in the list 
Look at its status

METRICS
Perceived TTFB: 150 ms (<800 ms)
Total latency:  7880 ms
```

## Project Structure

```
VoiceAgent/
├── main.py                 # Main RAG system
├── ingest.py              # Document processing and indexing
├── requirements.txt       # Python dependencies
├── .env                   # API keys (create this)
├── data/
│   └── Manual.pdf        # Your technical documentation
└── cisco_7604_index/     # Generated index (after running ingest.py)
    ├── index.faiss       # Vector embeddings
    └── index.pkl         # Document store
```

## Configuration

### Customizing the System

**Adjust chunk size** (in `ingest.py`):
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Increase for larger chunks
    chunk_overlap=200,     # Adjust overlap
)
```

**Change retrieval count** (in `main.py`):
```python
async def vector_search(self, query, top_k=5):  # Adjust top_k
```

**Add more phonetic mappings** (in `main.py`):
```python
PHONETIC = {
    "CLI": "C-L-I",
    "API": "A-P-I",
    # Add your terms here
}
```

## Technical Details

### Architecture

1. **Query Processing**:
   - User query → Rule-based rewriting → Standalone query

2. **Parallel Retrieval**:
   - Vector search (FAISS)
   - BM25 keyword search
   - Results merged and deduplicated

3. **Response Generation**:
   - Top result used for context
   - LLM generates response (streaming)
   - Voice optimizer processes output in real-time

### Key Components

**VoiceOptimizer**: Transforms text for speech synthesis
- Expands technical abbreviations
- Breaks sentences longer than 15 words
- Applies phonetic spelling rules

**BM25**: Keyword-based search implementation
- Calculates term frequency and inverse document frequency
- Ranks documents by relevance score

**ZeroLatencyVoiceRAG**: Main system orchestrator
- Manages conversation history
- Coordinates parallel searches
- Streams optimized responses

## API Rate Limits

The free tier of Google's Generative AI API has limits:
- 5 requests per minute for Gemini 2.5 Flash
- 1,500 requests per day

For production use, consider upgrading to a paid plan.

## Troubleshooting

### Common Issues

**Error: "Index folder not found"**
```bash
# Solution: Run ingestion first
python ingest.py
```

**Error: "429 Quota exceeded"**
```bash
# Solution: Wait 60 seconds between requests or upgrade API plan
```

**Error: "API key not found"**
```bash
# Solution: Create .env file with GOOGLE_API_KEY
echo "GOOGLE_API_KEY=your_key_here" > .env
```

## Performance Optimization Tips

1. **Reduce context size**: Limit `page_content[:800]` to smaller values
2. **Adjust top_k**: Retrieve fewer documents for faster processing
3. **Use smaller models**: Switch to Gemini Flash Lite for lower latency
4. **Batch queries**: Minimize API calls by caching embeddings

## Contributing

To extend the system:

1. Add more phonetic mappings in `VoiceOptimizer.PHONETIC`
2. Implement custom query rewriting rules in `fast_rewrite()`
3. Adjust BM25 parameters (k1, b) for your domain
4. Customize the generation prompt for your use case

## License

MIT License

## Acknowledgments

Built for CCaaS platforms requiring real-time voice AI capabilities for technical support.
