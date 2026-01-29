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

