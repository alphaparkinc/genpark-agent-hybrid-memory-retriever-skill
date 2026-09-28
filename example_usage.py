from client import AgentHybridMemoryRetriever
import json

retriever = AgentHybridMemoryRetriever()
print("=== AGENT HYBRID MEMORY RETRIEVER BENCHMARK ===")
res = retriever.run_memory_benchmark()
print(json.dumps(res, indent=2))
