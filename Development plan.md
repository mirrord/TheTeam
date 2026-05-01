This file is in the ideation phase. Ignore it.


# Broad Schedule
1. Build out foundation
x    a. test: ensure chat, tools, and flowchart execution is solid.
x    b. CoT flowcharts are unnecessary, remove
x    c. abstract agent for other local libraries (vllm, litellm, llama.cpp)
x    d. break up large modules: agent.py, tools.py, flowchart.py, flownode.py, memory_tool.py
x    e. response streaming support
x    f. flowchart execution trace
x    g. tool init cache
x    h. sandbox python nodes
x    i. config pathing fix
x    j. flowchart execution log w/ rolling window
x    k. conversation persistence (via db)
x    l. searchable conversations
x    m. flowcharts as nodes (subflowcharts)
x    n. flowchart validation
    o. llm_condition node
2. Impl advanced memory & knowledge functions:
    a. from file upload; PDF, .doc, more?
    b. compactifying context
    c. short/long memory with autorecall (semantic search?)
    d. self-modifiable context via recall functions
    e. memory store TTL/invalidation strategy (via importance score metadata?)
3. Impl advanced tooling 
    a. skills via memory notes
    b. flowchart creation/modification/execution as tool
x    c. flowchart progress callback
    d. tool RAG
4. Team coordination
    a. planning
    b. conversation
    c. shared resources
5. Integration
    a. Knowledge sources: web search interface (!!)
    b. direction/orders from openclaw
    c. delegation to vscode/copilot
    d. STT & TTS, webcam


Memory features:

1. Configurable automatic memory compaction. When a configured context size is reached: tag old messages for storage in memory, summarize them and include a note listing important entities not directly named in the summary, and replace the stored messages with the summary generated.
2. Automatic recall. If enabled, memories are automatically retrieved from the available knowledge and stored conversations and prepended to the agent's history. The inserted memory should not be subject to automatic compaction. If a previous automatic retrieval is already present, it should be replaced. The retrieval is performed with RAG based on an ephemeral query to the agent in its state at the time of the prompt with a wrapper prompt asking the agent for semantic search criteria. Once the memory has been prepended to the conversation history, the wrapped prompt is removed and the inner prompt is presented to generate the agent's response.
