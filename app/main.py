"""Entry point: CLI (chat), FastAPI, or Streamlit.

CLI:        python -m app.main                      (interactive)
            python -m app.main --query "..."
API:        uvicorn app.main:app --reload
Streamlit:  streamlit run app/main.py
"""
import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.agent_loop import AgentLoop  # noqa: E402


def ask(question: str, history: list[dict] | None = None) -> dict:
    return asdict(AgentLoop().run(question, history or []))


# ------------------------------------------------------------------ FastAPI
try:
    from fastapi import FastAPI
    from pydantic import BaseModel

    class AskRequest(BaseModel):
        question: str
        history: list[dict] = []

    app = FastAPI(title="Agentic RAG")

    @app.post("/ask")
    def ask_endpoint(req: AskRequest):
        return ask(req.question, req.history)
except ImportError:
    app = None


# ---------------------------------------------------------------------- CLI
def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help="Ask a single question and exit")
    args = parser.parse_args()

    agent = AgentLoop()
    history: list[dict] = []

    def turn(q: str) -> None:
        result = agent.run(q, history)
        print(f"\n{result.answer}\n")
        if result.sources:
            print("Sources:", ", ".join(f"[{s['label']}] {s['source']}" for s in result.sources))
        print(f"Steps: {result.steps} | Tokens: {result.cost['total_tokens']} | "
              f"Trace: {result.trace_path}\n")
        history.extend([{"role": "user", "content": q}, {"role": "assistant", "content": result.answer}])

    if args.query:
        turn(args.query)
        return
    print("Agentic RAG. Type 'exit' to quit.")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            turn(q)


# ---------------------------------------------------------------- Streamlit
def _in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def run_streamlit() -> None:
    import streamlit as st

    st.title("Agentic RAG")
    history = st.session_state.setdefault("history", [])
    for m in history:
        st.chat_message(m["role"]).write(m["content"])

    q = st.chat_input("Ask a question")
    if q:
        st.chat_message("user").write(q)
        result = AgentLoop().run(q, history)
        st.chat_message("assistant").write(result.answer)
        with st.expander("Sources"):
            st.json(result.sources)
        with st.expander("Cost"):
            st.json(result.cost)
        with st.expander("Trace"):
            st.code(Path(result.trace_path).read_text(encoding="utf-8"), language="json")
        history.extend([{"role": "user", "content": q}, {"role": "assistant", "content": result.answer}])


if _in_streamlit():
    run_streamlit()
elif __name__ == "__main__":
    cli()