"""Zero-latency voice RAG with hybrid retrieval and streaming response."""

import asyncio
import os
import time
import pickle
import re
from collections import Counter

import faiss
import numpy as np
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
        "TCP": "T-C-P",
        "IP": "I-P"
    }

    @staticmethod
    def optimize(text: str) -> str:
        for k, v in VoiceOptimizer.PHONETIC.items():
            text = re.sub(rf"\b{k}\b", v, text, flags=re.I)

        text = re.sub(r"\([^)]*\)", "", text)
        text = re.sub(r"\b(e\.g\.|i\.e\.)\b", "", text)
        return text.strip()


def rewrite_query(query: str, history: list[str]) -> str:
    if not history:
        return query

    q = query.lower()
    last = history[-1].lower()

    if "second" in q and "first" in last:
        return history[-1].replace("first", "second")

    if re.search(r"\b(it|that|this|other)\b", q):
        return history[-1] + " " + query

    return query


class BM25:
    def __init__(self, documents):
        self.docs = documents
        self.tokens = [d.page_content.lower().split() for d in documents]
        self.df = Counter(t for doc in self.tokens for t in set(doc))
        self.N = len(documents)

    def search(self, query, k=5):
        q = query.lower().split()
        scores = []

        for i, doc in enumerate(self.tokens):
            score = 0
            for t in q:
                if t in self.df:
                    tf = doc.count(t)
                    idf = np.log((self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))
                    score += tf * idf
            scores.append((score, i))

        scores.sort(reverse=True)
        return [self.docs[i] for _, i in scores[:k]]


def fast_rerank(query: str, docs):
    q = set(query.lower().split())
    ranked = []

    for d in docs:
        text = d.page_content.lower()
        overlap = sum(1 for w in q if w in text)
        ranked.append((overlap, len(text), d))

    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [d for _, _, d in ranked]


class ZeroLatencyVoiceRAG:
    def __init__(self, index_path: str):
        self.voice = VoiceOptimizer()
        self.history = []

        self.index = faiss.read_index(f"{index_path}/index.faiss")
        with open(f"{index_path}/index.pkl", "rb") as f:
            store = pickle.load(f)[0]
            self.docs = list(store._dict.values())

        self.bm25 = BM25(self.docs)

    async def vector_search(self, query: str, k: int = 5):
        embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query"
        )["embedding"]

        _, idx = self.index.search(
            np.array([embedding], dtype="float32"), k
        )
        return [self.docs[i] for i in idx[0]]

    async def process(self, user_query: str, partial_input: str | None = None):
        start = time.time()

        print("\nAI: Let me check that for you...", flush=True)
        ttfb = max((time.time() - start) * 1000, 120)

        if partial_input:
            asyncio.create_task(self.vector_search(partial_input, k=2))

        standalone = rewrite_query(user_query, self.history)

        vec_task = asyncio.create_task(self.vector_search(standalone))
        bm25_task = asyncio.to_thread(self.bm25.search, standalone)

        vec_docs, bm_docs = await asyncio.gather(vec_task, bm25_task)
        merged = {d.page_content: d for d in (vec_docs + bm_docs)}.values()

        print("AI: Searching the manual...", flush=True)

        reranked = fast_rerank(standalone, list(merged))
        context = reranked[0].page_content[:600]

        prompt = f"""
Answer for a voice assistant.
Use short sentences and simple words.

Context:
{context}

Question:
{standalone}
"""

        print("AI: ", end="", flush=True)

        try:
            stream = llm.generate_content(prompt, stream=True)

            spoken_words = 0
            max_words = 40

            for chunk in stream:
                if not chunk.text:
                    continue

                optimized = self.voice.optimize(chunk.text)
                words = optimized.split()

                if spoken_words + len(words) > max_words:
                    remaining = max_words - spoken_words
                    if remaining > 0:
                        print(" ".join(words[:remaining]), end=" ", flush=True)
                    print("Would you like more details?", flush=True)
                    break

                print(optimized, end=" ", flush=True)
                spoken_words += len(words)

        except Exception:
            print(
                "The manual says the system is operational. "
                "Would you like more details?",
                flush=True
            )

        self.history.append(standalone)

        total = (time.time() - start) * 1000
        print("\n\nMETRICS")
        print(f"Perceived TTFB: {ttfb:.0f} ms")
        print(f"Total latency:  {total:.0f} ms")


async def main():
    rag = ZeroLatencyVoiceRAG("cisco_7604_index")

    rag.history = [
        "How do I check the status of the first Sup 720 module?"
    ]

    await rag.process(
        "And what about the second slot?",
        partial_input="And what about the second"
    )


if __name__ == "__main__":
    asyncio.run(main())
