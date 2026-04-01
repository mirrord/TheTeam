"""Demo: Automatic Context Compaction and Automatic Memory Recall.

This script walks through both new agent memory features introduced in pithos:

1. **Memory Compaction** — simulates a long conversation and shows the context
   being automatically summarised when the threshold is reached.

2. **Automatic Recall** — seeds the vector memory store with facts, then starts
   a fresh conversation and shows how past knowledge is injected before each
   response, without any manual retrieval call.

Run:
    python demos/memory_features_demo.py

Requirements:
    - Ollama running locally with a model available (default: glm-4.7-flash)
    - chromadb installed (pip install -e .)
"""

import sys
import tempfile
import textwrap
from pathlib import Path

# Ensure the src directory is on the path when running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pithos import OllamaAgent, ConfigManager
from pithos.agent.compaction import CompactionConfig
from pithos.agent.recall import RecallConfig

# ── helpers ──────────────────────────────────────────────────────────────────

DIVIDER = "─" * 70
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def header(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{DIVIDER}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{DIVIDER}{RESET}")


def step(label: str) -> None:
    print(f"\n{BOLD}{GREEN}▶ {label}{RESET}")


def info(text: str) -> None:
    for line in textwrap.wrap(text, width=68):
        print(f"  {DIM}{line}{RESET}")


def show_context_summary(agent: OllamaAgent, context_name: str = "default") -> None:
    ctx = agent.contexts[context_name]
    total = len(ctx.message_history)
    summaries = [
        m for m in ctx.message_history if "[CONTEXT SUMMARY]" in m.get("content", "")
    ]
    recalls = [m for m in ctx.message_history if m.get("_pithos_auto_recall")]
    normals = total - len(summaries) - len(recalls)
    print(
        f"  {YELLOW}Context snapshot:{RESET} "
        f"{total} messages total  "
        f"({normals} normal, {len(summaries)} summary, {len(recalls)} recall)"
    )
    for s in summaries:
        preview = s["content"][:200].replace("\n", " ")
        print(f"    {DIM}[SUMMARY] {preview}...{RESET}")
    for r in recalls:
        preview = r["content"][:200].replace("\n", " ")
        print(f"    {DIM}[RECALL]  {preview}...{RESET}")


def ask(model: str, prompt: str, default: str) -> str:
    answer = input(f"  {prompt} [{default}]: ").strip()
    return answer or default


# ── Demo 1: Compaction ────────────────────────────────────────────────────────


def demo_compaction(model: str, tmpdir: str) -> None:
    header("Demo 1 — Automatic Context Compaction")
    info(
        "We create an agent with a low compaction threshold (8 messages) and "
        "send it a rapid-fire series of astronomy questions. When the history "
        "reaches the threshold, the oldest messages are automatically summarised "
        "and replaced with a [CONTEXT SUMMARY] block."
    )

    config_manager = ConfigManager()
    agent = OllamaAgent(model, agent_name="compaction-demo")
    agent.enable_memory(config_manager, persist_directory=f"{tmpdir}/memory")

    # Very low threshold so we can see it trigger quickly
    agent.enable_compaction(
        CompactionConfig(
            threshold=8,
            keep_last=2,
            summary_model=model,
            memory_category="context_summaries",
        )
    )

    try:
        questions = [
            "What is the closest star to Earth?",
            "How many moons does Jupiter have?",
            "What is a neutron star?",
            "How large is the Milky Way galaxy?",
            "What causes a solar eclipse?",
            "What is the difference between a meteor and a meteorite?",
            "What is the Hubble constant?",
            "How hot is the surface of the Sun?",
            "What is dark matter?",
            "What is the cosmic microwave background?",
        ]

        step("Sending questions one by one…")
        for i, q in enumerate(questions, 1):
            print(f"\n  {DIM}[{i}/{len(questions)}] You: {q}{RESET}")
            try:
                response = agent.send(q)
            except Exception as exc:
                print(f"  [error] {exc}")
                continue
            preview = response[:120].replace("\n", " ")
            print(f"  Agent: {preview}{'…' if len(response) > 120 else ''}")
            show_context_summary(agent)

        step("Final context state:")
        show_context_summary(agent)

        summaries = [
            m
            for m in agent.contexts["default"].message_history
            if "[CONTEXT SUMMARY]" in m.get("content", "")
        ]
        if summaries:
            print(f"\n{GREEN}✓ Compaction fired! Summary content:{RESET}")
            print(textwrap.indent(summaries[0]["content"], "    "))
        else:
            info("Compaction did not fire yet — try lowering the threshold.")
    finally:
        agent.close()


# ── Demo 2: Automatic Recall ──────────────────────────────────────────────────


def demo_recall(model: str, tmpdir: str) -> None:
    header("Demo 2 — Automatic Memory Recall")
    info(
        "We seed the vector memory store with facts about a fictional project, "
        "then start a fresh agent (empty context) with recall enabled. The agent "
        "has never seen these facts in its conversation history — they live only in "
        "the memory store. Watch how the [RECALLED CONTEXT] block appears "
        "automatically before each response."
    )

    config_manager = ConfigManager()
    agent = OllamaAgent(model, agent_name="recall-demo")
    agent.enable_memory(config_manager, persist_directory=f"{tmpdir}/memory2")
    agent.enable_history(
        persist_directory=f"{tmpdir}/history",
        session_id="recall-demo-session",
    )
    try:
        # ── Seed the memory store with project facts ──────────────────────────────
        step("Seeding vector memory with project facts…")
        facts = [
            "The project is called Orion. It is a distributed task scheduler written in Go.",
            "Orion uses PostgreSQL 15 as its primary datastore with a schema called orion_db.",
            "The lead developer is Maya Chen. She can be reached at maya@orionproject.io.",
            "Orion's REST API runs on port 8443 and requires a JWT bearer token for all endpoints.",
            "The CI/CD pipeline is managed by GitHub Actions and deploys to a Kubernetes cluster on GKE.",
            "Known issue: the task retry mechanism has a race condition under high concurrency (issue #482).",
            "The project roadmap for Q2 includes: gRPC support, improved observability, and plugin API.",
        ]
        for fact in facts:
            agent.memory_store.store("project_notes", fact)
            print(f"  {DIM}stored: {fact[:80]}…{RESET}")

        # ── Enable recall and start a fresh conversation ──────────────────────────
        agent.enable_recall(
            RecallConfig(
                sources=["memory"],
                n_results=4,
                recall_model=model,
                min_relevance=0.4,
            )
        )

        step("Starting fresh conversation with recall enabled…")
        info(
            "The agent's context history is empty — it has not seen any of the project "
            "facts yet. But recall will retrieve them automatically."
        )

        queries = [
            "What database does the project use?",
            "Who should I contact about the project?",
            "What API port does the service run on?",
            "What known bugs are there?",
        ]

        for q in queries:
            print(f"\n  {BOLD}You:{RESET} {q}")
            try:
                response = agent.send(q)
            except Exception as exc:
                print(f"  [error] {exc}")
                continue
            print(
                f"  {BOLD}Agent:{RESET} {response[:300].replace(chr(10), ' ')}{'…' if len(response) > 300 else ''}"
            )
            show_context_summary(agent)

        step("Final context state:")
        show_context_summary(agent)

        recalls = [
            m
            for m in agent.contexts["default"].message_history
            if m.get("_pithos_auto_recall")
        ]
        if recalls:
            print(f"\n{GREEN}✓ Recall is active. Latest injected context:{RESET}")
            print(textwrap.indent(recalls[0]["content"][:500], "    "))
        else:
            info(
                "No recall injection found — ensure chromadb is installed and a model is available."
            )
    finally:
        agent.close()


# ── Demo 3: Combined ──────────────────────────────────────────────────────────


def demo_combined(model: str, tmpdir: str) -> None:
    header("Demo 3 — Compaction + Recall Together")
    info(
        "Both features enabled simultaneously. Compaction summaries are archived "
        "to the memory store, so past summaries can themselves be recalled in "
        "future turns."
    )

    config_manager = ConfigManager()
    agent = OllamaAgent(model, agent_name="combined-demo")
    agent.enable_memory(config_manager, persist_directory=f"{tmpdir}/memory3")
    agent.enable_history(
        persist_directory=f"{tmpdir}/history3",
        session_id="combined-demo",
    )

    agent.enable_compaction(
        CompactionConfig(
            threshold=10,
            keep_last=3,
            summary_model=model,
            memory_category="context_summaries",
        )
    )
    agent.enable_recall(
        RecallConfig(
            sources=["memory", "history"],
            n_results=3,
            recall_model=model,
            min_relevance=0.4,
        )
    )

    try:
        turns = [
            "What are the main features of the Rust programming language?",
            "How does Rust's ownership system prevent memory errors?",
            "What is the borrow checker in Rust?",
            "Can you give an example of a lifetime annotation in Rust?",
            "How does Rust handle concurrency?",
            "What is the difference between Arc and Rc in Rust?",
            "Now, can you recall what we discussed about ownership earlier?",
        ]

        step("Running conversation…")
        for i, q in enumerate(turns, 1):
            print(f"\n  {BOLD}[{i}] You:{RESET} {q}")
            try:
                response = agent.send(q)
            except Exception as exc:
                print(f"  [error] {exc}")
                continue
            print(f"  {BOLD}Agent:{RESET} {response[:200].replace(chr(10), ' ')}…")
            show_context_summary(agent)

        step("Done.")
        info(
            "The last question asked the agent to recall earlier content. "
            "If compaction fired, the recall system should have surfaced the archived "
            "summary from the memory store."
        )
    finally:
        agent.close()


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    print(f"\n{BOLD}pithos — Memory Features Demo{RESET}")
    print(f"{DIM}Automatic Context Compaction + Automatic Memory Recall{RESET}\n")

    model = ask(model="", prompt="Ollama model to use", default="glm-4.7-flash")

    print("\nAvailable demos:")
    print("  1. Context Compaction only")
    print("  2. Automatic Recall only")
    print("  3. Both features combined")
    print("  4. Run all three sequentially")
    choice = ask(model="", prompt="Select demo", default="4")

    with tempfile.TemporaryDirectory(prefix="pithos_demo_") as tmpdir:
        if choice in ("1", "4"):
            demo_compaction(model, tmpdir)
        if choice in ("2", "4"):
            demo_recall(model, tmpdir)
        if choice in ("3", "4"):
            demo_combined(model, tmpdir)

    print(f"\n{BOLD}{GREEN}Demo complete.{RESET}")


if __name__ == "__main__":
    main()
