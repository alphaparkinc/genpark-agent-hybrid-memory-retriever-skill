import sys, json, time, math, re
from collections import Counter

class AgentHybridMemoryRetriever:
    """
    Hybrid Sparse-Dense Memory Retriever for Autonomous Agents.
    Combines BM25-style term weighting with character/word n-gram semantic similarity
    and exponential half-life recency decay.
    """
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.memories = {}  # id -> dict
        self.doc_count = 0
        self.avg_doc_len = 0.0

    def _tokenize(self, text):
        return re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text.lower())

    def _get_ngrams(self, text, n=3):
        cleaned = re.sub(r"\s+", " ", text.lower().strip())
        return [cleaned[i:i+n] for i in range(max(0, len(cleaned) - n + 1))]

    def store_memory(self, memory_id, content, tags=None, metadata=None, importance=1.0, timestamp=None):
        tokens = self._tokenize(content)
        ngrams = Counter(self._get_ngrams(content, 3))
        ts = timestamp if timestamp is not None else time.time()
        
        self.memories[memory_id] = {
            "id": memory_id,
            "content": content,
            "tags": tags or [],
            "metadata": metadata or {},
            "importance": max(0.1, min(5.0, float(importance))),
            "timestamp": ts,
            "tokens": Counter(tokens),
            "doc_len": len(tokens),
            "ngrams": ngrams
        }
        self.doc_count = len(self.memories)
        total_len = sum(m["doc_len"] for m in self.memories.values())
        self.avg_doc_len = total_len / max(1, self.doc_count)
        return {"status": "STORED", "memory_id": memory_id, "doc_length": len(tokens)}

    def retrieve_hybrid_memories(self, query, top_k=3, alpha=0.6, half_life_hours=48.0, filter_tags=None):
        """
        Retrieve memories with fused hybrid score:
        Score = (alpha * BM25 + (1 - alpha) * SemanticSimilarity) * RecencyDecay * Importance
        """
        if not self.memories:
            return {"results": [], "query": query, "total_memories": 0}

        q_tokens = self._tokenize(query)
        q_ngrams = Counter(self._get_ngrams(query, 3))
        now = time.time()
        results = []

        # Calculate IDF for query tokens
        idf = {}
        for token in set(q_tokens):
            docs_with_token = sum(1 for m in self.memories.values() if token in m["tokens"])
            idf[token] = math.log(1.0 + (self.doc_count - docs_with_token + 0.5) / (docs_with_token + 0.5))

        # Query n-gram norm for cosine similarity
        q_norm = math.sqrt(sum(v * v for v in q_ngrams.values())) or 1.0

        for mid, mem in self.memories.items():
            if filter_tags and not any(t in mem["tags"] for t in filter_tags):
                continue

            # 1. Sparse BM25 Score
            bm25 = 0.0
            for token in q_tokens:
                tf = mem["tokens"].get(token, 0)
                if tf > 0:
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (mem["doc_len"] / max(1.0, self.avg_doc_len)))
                    bm25 += idf.get(token, 0.5) * (numerator / denominator)
            norm_bm25 = math.tanh(bm25 / 3.0)

            # 2. Dense Semantic n-gram Cosine Score
            intersection = sum(q_ngrams[ng] * mem["ngrams"].get(ng, 0) for ng in q_ngrams)
            mem_norm = math.sqrt(sum(v * v for v in mem["ngrams"].values())) or 1.0
            cosine_sim = intersection / (q_norm * mem_norm)

            # 3. Recency Decay (exponential half-life)
            delta_hours = max(0.0, (now - mem["timestamp"]) / 3600.0)
            decay_factor = math.pow(0.5, delta_hours / max(1.0, half_life_hours))

            # 4. Fused Score
            fused_base = (alpha * norm_bm25) + ((1.0 - alpha) * cosine_sim)
            final_score = round(fused_base * decay_factor * mem["importance"], 4)

            results.append({
                "id": mid,
                "content": mem["content"],
                "final_score": final_score,
                "sparse_bm25": round(norm_bm25, 4),
                "dense_semantic": round(cosine_sim, 4),
                "recency_decay": round(decay_factor, 4),
                "tags": mem["tags"]
            })

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return {
            "query": query,
            "top_memories": results[:top_k],
            "total_candidates": len(results)
        }

    def prune_decayed_memories(self, min_score_threshold=0.05, max_age_hours=720.0):
        now = time.time()
        to_delete = []
        for mid, mem in self.memories.items():
            age_hours = (now - mem["timestamp"]) / 3600.0
            if age_hours > max_age_hours:
                to_delete.append(mid)
        for mid in to_delete:
            del self.memories[mid]
        self.doc_count = len(self.memories)
        return {"pruned_count": len(to_delete), "remaining": self.doc_count}

    def run_memory_benchmark(self):
        self.memories.clear()
        now = time.time()
        
        # Populate episodic memory facts
        self.store_memory("mem_1", "User prefers deploying microservices with Docker and Kubernetes on AWS EKS.", tags=["infra", "preference"], timestamp=now - 3600)
        self.store_memory("mem_2", "Primary programming language is Python with FastAPI for backend APIs.", tags=["coding", "python"], timestamp=now - 7200)
        self.store_memory("mem_3", "Production database is PostgreSQL with connection pooling via PgBouncer.", tags=["database", "infra"], timestamp=now - 86400)
        self.store_memory("mem_4", "User frequently asks for zero-pip dependency implementations in Python.", tags=["coding", "preference"], importance=1.5, timestamp=now - 1800)

        query = "What framework and language does user prefer for backend APIs?"
        search_res = self.retrieve_hybrid_memories(query, top_k=2)

        return {
            "suite": "Hybrid Memory Retriever Benchmark",
            "search_query": query,
            "top_retrievals": search_res["top_memories"],
            "memory_bank_size": len(self.memories),
            "status": "OPERATIONAL"
        }
