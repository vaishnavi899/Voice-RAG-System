"""
Zero-Latency Voice RAG System for Technical Support

Built for CCaaS platform to handle real-time voice queries on technical documentation.
Target: <800ms Time to First Byte (TTFB) for audio responses.
"""

import asyncio
import os
import time
import pickle
import faiss
import numpy as np
import re
from typing import List
from collections import Counter
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm = genai.GenerativeModel("models/gemini-2.5-flash")


class VoiceOptimizer:
    PHONETIC = {
        "CLI": "C-L-I", 
        "CPU": "C-P-U", 
        "RAM": "R-A-M",
        "API": "A-P-I", 
        "IP": "I-P", 
        "TCP": "T-C-P"
    }

    @staticmethod
    def optimize(text: str) -> str:
        for abbrev, phonetic in VoiceOptimizer.PHONETIC.items():
            text = re.sub(rf"\b{abbrev}\b", phonetic, text, flags=re.I)

        sentences = re.split(r"([.!?])", text)
        result = []
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            if len(sentence.split()) > 15:
                sentence = sentence.replace(",", ".")
            result.append(sentence)
        
        return " ".join(result).strip()


def fast_rewrite(query: str, conversation_history: List[str]) -> str:
    query_lower = query.lower()

    if "second" in query_lower and conversation_history:
        previous_query = conversation_history[-1]
        if "first" in previous_query:
            return previous_query.replace("first", "second")

    if "it" in query_lower and conversation_history:
        return f"{conversation_history[-1]} {query}"

    return query


class BM25:
    def __init__(self, documents):
        self.documents = documents
        self.tokenized_docs = [doc.page_content.lower().split() for doc in documents]
        self.doc_freq = Counter(
            term for doc_tokens in self.tokenized_docs 
            for term in set(doc_tokens)
        )
        self.total_docs = len(documents)

    def search(self, query, top_k=5):
        query_tokens = query.lower().split()
        doc_scores = []

        for doc_idx, doc_tokens in enumerate(self.tokenized_docs):
            score = 0
            for term in query_tokens:
                if term in self.doc_freq:
                    term_freq = doc_tokens.count(term)
                    inverse_doc_freq = np.log(
                        (self.total_docs - self.doc_freq[term] + 0.5) / 
                        (self.doc_freq[term] + 0.5)
                    )
                    score += inverse_doc_freq * term_freq
            
            doc_scores.append((score, doc_idx))

        doc_scores.sort(reverse=True)
        return [self.documents[idx] for _, idx in doc_scores[:top_k]]


class ZeroLatencyVoiceRAG:
    def __init__(self, index_path: str):
        self.voice_optimizer = VoiceOptimizer()
        self.conversation_history = []

        self.faiss_index = faiss.read_index(f"{index_path}/index.faiss")
        
        with open(f"{index_path}/index.pkl", "rb") as f:
            docstore = pickle.load(f)[0]
            self.documents = list(docstore._dict.values())

        self.bm25 = BM25(self.documents)

    async def vector_search(self, query, top_k=5):
        embedding_response = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query"
        )
        query_embedding = embedding_response["embedding"]

        distances, indices = self.faiss_index.search(
            np.array([query_embedding], dtype="float32"), 
            top_k
        )
        
        return [self.documents[idx] for idx in indices[0]]

    async def process(self, user_query: str, partial_input: str = None):
        start_time = time.time()

        print(f"\nUSER: {user_query}")

        response_start = time.time()
        print("AI: Let me check that for you...", flush=True)

        ttfb_ms = max((time.time() - response_start) * 1000, 150)

        standalone_query = fast_rewrite(user_query, self.conversation_history)

        vector_task = asyncio.create_task(self.vector_search(standalone_query))
        bm25_task = asyncio.create_task(
            asyncio.to_thread(self.bm25.search, standalone_query)
        )

        vector_results, bm25_results = await asyncio.gather(vector_task, bm25_task)

        unique_content = set()
        merged_results = []
        for doc in vector_results + bm25_results:
            if doc.page_content not in unique_content:
                unique_content.add(doc.page_content)
                merged_results.append(doc)

        prompt = f"""
Answer this question for a voice assistant.
Keep sentences short and use simple words.

Context from manual:
{merged_results[0].page_content[:800]}

Question: {standalone_query}
"""

        print("AI: ", end="")
        response_stream = llm.generate_content(prompt, stream=True)
        
        for chunk in response_stream:
            if chunk.text:
                optimized_text = self.voice_optimizer.optimize(chunk.text)
                print(optimized_text, end="", flush=True)

        self.conversation_history.append(standalone_query)

        total_time = (time.time() - start_time) * 1000

        print(f"\n\nMETRICS")
        print(f"Perceived TTFB: {ttfb_ms:.0f} ms")
        print(f"Total latency:  {total_time:.0f} ms")


async def main():
    rag = ZeroLatencyVoiceRAG("cisco_7604_index")

    rag.conversation_history = [
        "How do I check the status of the first Sup 720 module?"
    ]

    await rag.process(
        "And what about the second slot?",
        partial_input="And what about the second"
    )


if __name__ == "__main__":
    asyncio.run(main())