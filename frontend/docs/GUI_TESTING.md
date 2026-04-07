# GUI Testing — Coverage and Constraints

## Test suite

**Location:** `frontend/src/__tests__/`  
**Framework:** Vitest 4 + React Testing Library + jsdom  
**Run:** `npm test` (from `frontend/`)

| File | Tests | What is covered |
|---|---|---|
| `lib/utils.test.ts` | 21 | `cn`, `formatDate`, `truncate`, `debounce` (all branches) |
| `lib/fetchWithToast.test.ts` | 9 | All HTTP-status branches (2xx / 4xx / 5xx / network error), double-toast guard |
| `store/toastStore.test.ts` | 11 | `addToast` (types, auto-remove, duration=0), `removeToast` |
| `store/chatStore.test.ts` | 24 | Streaming pipeline (`startStreaming`, `appendStreamChunk`, `finalizeStreaming`), `addMessage`, `setCurrentConversation`, `setProcessing`, `fetchConversations`, `createConversation`, `deleteConversation`, `updateAgent` |
| `store/flowchartStore.test.ts` | 28 | `setNodes/Edges/updateNodeData/updateEdgeData` (pure state), `getFlowchart` YAML→ReactFlow conversion (node types, start/end detection, edge-type mapping, handle-ID selection, legacy `node.next` fallback, error handling) |
| `store/agentStore.test.ts` | 15 | `setCurrentAgent`, `fetchAgents`, `getAgent`, `createAgent`, `updateAgent`, `deleteAgent`, `testAgent` |
| `components/ConnectionStatus.test.tsx` | 8 | All 4 socket states (connected, connecting, reconnecting, error/dismiss, disconnected) |
| `components/ToastContainer.test.tsx` | 9 | Rendering all 4 types, dismiss button, per-type CSS classes |
| `components/NodeMenu.test.tsx` | 28 | All 11 node-type buttons render and fire `onAddNode` with the correct `type` and `defaultData` |
| `components/EdgeEditDialog.test.tsx` | 14 | Open/close, field initialisation from props, `CountCondition` limit field visibility, `edgeData` prop update, Save/Cancel/X close, correct `onSave` payload |

---

## Bug found and fixed during testing

**`lib/fetchWithToast.ts`** — the catch block intended to suppress double-toasting for 5xx errors used a fragile string-contains check (`error.message.includes('Server error')`). This silently added a second toast for any JSON error body whose message did not include that exact substring (e.g. `"boom"`). Fixed by marking the already-toasted error with `err._toasted = true` and checking that flag in the catch block.

---

## Functions not covered by automated tests

### `store/socketStore.ts` — entire store

| Function | Why not testable |
|---|---|
| `connect()` | Calls `io()` from `socket.io-client`, which opens a real TCP connection. Even with the client library mocked, the store wires up a dense set of Socket.IO event listeners whose behaviour depends on receiving server frames. Meaningful tests would require a live server or a Socket.IO mock-server library (e.g. `socket.io-mock`). |
| `disconnect()` | Requires a live socket to call `.disconnect()` on. |
| `emit(event, data)` | No logic beyond delegating to `socket.emit()`; only verifiable against a real server. |
| `on(event, handler)` / `off(event, handler)` | Thin delegation wrappers. |
| `resetError()` | Trivial one-liner (`set({ error: null })`). The state transition is not interesting enough to warrant a test. |
| Exponential-backoff reconnect logic | The `attemptReconnect` closure uses module-level mutable state (`reconnectDelay`, `reconnectTimer`) and relies on the socket's `connect` event firing. Testing this accurately requires a mock socket with controllable connect/disconnect events; it would be a significant test-infrastructure investment for a standard library pattern. |
| Ping health-check (`socket.on('pong', …)`) | Again requires real socket frames. |

**Recommended approach:** use `socket.io-mock` or spin up a `socket.io` server inside a `vitest` global setup if socket behaviour becomes a stability issue.

---

### `pages/FlowchartEditor.tsx` — canvas and drag-and-drop interactions

| Function / behaviour | Why not testable |
|---|---|
| ReactFlow canvas render (node layout, panning, zoom) | Requires WebGL/Canvas APIs absent in jsdom. ReactFlow renders SVG and canvas elements that need a real browser rendering context. |
| Drag-and-drop node positioning | Uses the HTML5 Drag-and-Drop API (`ondragstart`, `ondrop`). jsdom has only stub implementations; `@testing-library/user-event` does not simulate HTML5 drag events reliably. |
| `onConnect` (drawing an edge between two nodes) | Fired by ReactFlow internals when the user drags from one handle to another. No public API to trigger this synthetically without a real canvas. |
| `handleSave` serialisation | The save function reads `useReactFlow().getNodes()` / `getEdges()` from the ReactFlow context. This context is only properly populated when ReactFlow has rendered a canvas; it returns empty arrays in jsdom. The serialisation logic itself (building the `nodes`/`edges` config dict) is inline inside `FlowchartEditor` and not exported; testing it would require extracting it into a utility function. |
| Execution highlighting (socket `node_execution` event → node style mutation) | Requires both a live socket and a rendered ReactFlow canvas. |
| Import from YAML (file picker `<input type="file">`) | File access via `FileReader` / `<input type="file">` requires browser APIs not available in jsdom. Can be partially tested in an E2E framework (Playwright/Cypress). |
| Export to YAML (creates a `.yaml` download) | Uses `URL.createObjectURL` + a synthetic anchor click — browser-only. |

---

### `pages/ChatInterface.tsx` — live streaming

| Function / behaviour | Why not testable |
|---|---|
| Real-time streaming (`stream_start` → `stream_chunk` → `stream_end`) | Requires a live Socket.IO server emitting real `stream_*` events. The store's streaming mutations (`startStreaming`, `appendStreamChunk`, `finalizeStreaming`) are tested in isolation in `chatStore.test.ts`; what is not tested is the glue code inside `ChatInterface` that registers socket handlers. |
| Sending a message via WebSocket | The component uses `useSocketStore().emit(...)` which requires an active socket. |
| Agent/model selector visibility and save | Depends on `useAgentStore` fetch and intermediate UI state; testable with mocks in principle but the component renders a large form with many conditional branches. |

---

### `pages/AgentConfig.tsx` — complex form

| Function / behaviour | Why not testable without significant investment |
|---|---|
| Form save / field validation (full round-trip) | The component has ~400 lines of state interleaved with API calls. Component-level integration tests are possible (RTL + mocked fetch) but the effort/value ratio is low — the individual store actions are already fully tested. |
| `normalizeAgentConfig()` | This private helper inside `AgentConfig.tsx` converts the legacy YAML `tools` object format to the flat form shape. It is not exported. Extracting it to a utility file and testing it directly would be straightforward and is recommended if the normalisation logic grows. |

---

### `pages/EdgeDebugger.tsx`

This is a hardcoded debug tool with three fixed nodes and two fixed edges. It has no user-facing logic beyond a "Log State" button that writes to `console.log` and the ReactFlow canvas — which has the same jsdom constraints described above. No meaningful unit tests apply.

---

### `components/CustomEdge.tsx` — inline edit panel

The custom edge renderer includes an inline edit panel (toggled by clicking the edge label). Testing this requires a rendered ReactFlow canvas with actual edges visible in the viewport. The `EdgeEditDialog` component (which provides the same fields as a modal) **is** fully tested. The `data.onEdgeDataChange(id, data)` callback path in `CustomEdge` could be unit-tested if the component were extracted and rendered with mock ReactFlow context, but would require non-trivial mock wiring.

---

### Node components (`components/nodes/*.tsx`)

All 11 node components use `useReactFlow()` and `useEdges()` from the ReactFlow context. While they can be rendered inside a `<ReactFlowProvider>`, the hooks return stubs (empty arrays, no-op `setNodes`) — meaning variable-handle logic (`PromptNode`, `AgentPromptNode`), connected-handle detection (`isVariableConnected`), and the `handleStaticInputChange` callback (which calls `setNodes`) cannot be exercised meaningfully. The presentational properties (labels, badges, data display) **could** be tested with a mocked ReactFlow provider; this is a suitable follow-up if rendering regressions become a concern.
