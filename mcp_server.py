import sys, json
from client import AgentHybridMemoryRetriever

def handle_mcp():
    retriever = AgentHybridMemoryRetriever()
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print(json.dumps(retriever.run_memory_benchmark(), indent=2))
        return

    for line in sys.stdin:
        if not line.strip(): continue
        try:
            req = json.loads(line)
            method = req.get("method")
            msg_id = req.get("id")
            
            if method == "initialize":
                resp = {"jsonrpc": "2.0", "id": msg_id, "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "genpark-agent-hybrid-memory-retriever-skill", "version": "1.0.0"},
                    "capabilities": {"tools": {}}
                }}
            elif method == "tools/list":
                resp = {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": [
                    {"name": "store_memory", "description": "Store episodic memory with tags and importance.", "inputSchema": {"type": "object", "properties": {"memory_id": {"type": "string"}, "content": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}, "importance": {"type": "number"}}}},
                    {"name": "retrieve_hybrid_memories", "description": "Query memories using hybrid search with decay.", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}}}},
                    {"name": "prune_decayed_memories", "description": "Prune stale episodic memories.", "inputSchema": {"type": "object"}},
                    {"name": "run_memory_benchmark", "description": "Run memory benchmark.", "inputSchema": {"type": "object"}}
                ]}}
            elif method == "tools/call":
                tname = req.get("params", {}).get("name")
                args = req.get("params", {}).get("arguments", {})
                if tname == "store_memory":
                    res = retriever.store_memory(args.get("memory_id", "m1"), args.get("content", ""), args.get("tags"), None, args.get("importance", 1.0))
                elif tname == "retrieve_hybrid_memories":
                    res = retriever.retrieve_hybrid_memories(args.get("query", ""), args.get("top_k", 3))
                elif tname == "prune_decayed_memories":
                    res = retriever.prune_decayed_memories()
                else:
                    res = retriever.run_memory_benchmark()
                resp = {"jsonrpc": "2.0", "id": msg_id, "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}}
            else:
                resp = {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": "Method not found"}}
            
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "error": {"code": -32000, "message": str(e)}}) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    handle_mcp()
