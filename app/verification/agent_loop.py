"""Main controller: route -> plan -> retrieve -> grade -> answer -> verify."""
from dataclasses import asdict

from app.agent.llm_client import get_llm
from app.agent.planner import plan
from app.agent.router import route_question
from app.agent.state import AgentResult, AgentState, Attempt, format_history
from app.agent.stopping_criteria import check_stop
from app.config import settings
from app.generation.abstention import abstain_message
from app.generation.answer_generator import generate_answer
from app.grading.relevance_grader import grade_relevance
from app.grading.sufficiency_grader import grade_sufficiency
from app.observability.cost_tracker import tracker
from app.observability.trace_logger import TraceLogger
from app.query.hyde import hypothetical_passage
from app.query.query_expander import expand
from app.query.rewriter import rewrite_for_retry
from app.retrieval.reranker import rerank
from app.tools.registry import run_tool, tool_specs
from app.verification.answer_reviser import revise
from app.verification.citation_validator import CITE, validate_citations
from app.verification.claim_extractor import extract_claims
from app.verification.groundedness_checker import check_groundedness

TOOL_SYSTEM = """You are the retrieval controller. Choose the best search tool(s) for the sub-question.
- keyword_search: exact terms, IDs, names, error codes
- vector_search: conceptual or paraphrased questions
- hybrid_search: when unsure
- filtered_search: the question mentions a specific file, doc type, or date range
- fetch_document: read more context around a promising source
- sql_query: structured or numeric lookups
- web_search: only when the corpus lacks the answer
If earlier attempts with a tool found little, switch strategy. Call at most two tools.
Use the proposed query unless a better one is obvious."""

TEXT_TOOLS = {"vector_search", "keyword_search", "hybrid_search"}


class AgentLoop:
    def __init__(self):
        self.llm = get_llm()

    # ------------------------------------------------------------------ public
    def run(self, question: str, history: list[dict] | None = None) -> AgentResult:
        history = history or []
        tracker.start()
        trace = TraceLogger(question)
        state = AgentState(question=question, history=history)

        route = route_question(question, history)
        trace.log("route", **asdict(route))
        if not route.needs_retrieval:
            answer = self.llm.ask(
                "You are a helpful assistant. Answer concisely.",
                f"Conversation:\n{format_history(history)}\n\nUser: {question}",
                model=settings.planner_model,
            )
            trace.log("direct_answer", answer=answer)
            return self._finish(state, trace, answer, abstained=False)
        state.source = route.source

        state.standalone, state.sub_questions = plan(question, history)
        trace.log("plan", standalone=state.standalone, sub_questions=state.sub_questions)

        for sub_q in state.sub_questions:
            if state.total_steps >= settings.max_total_steps:
                trace.log("stop", reason="total_step_limit")
                break
            self._research(state, sub_q, trace)

        answer, abstained = self._draft_and_verify(state, trace)
        return self._finish(state, trace, answer, abstained)

    # ---------------------------------------------------------------- research
    def _research(self, state: AgentState, sub_q: str, trace: TraceLogger) -> bool:
        previous_queries: list[str] = []
        missing = ""
        sub_steps = streak = 0

        while True:
            sub_steps += 1
            state.total_steps += 1
            query = sub_q if not previous_queries else rewrite_for_retry(sub_q, previous_queries, missing)
            previous_queries.append(query)

            tools_used, chunks, errors = self._search(state, sub_q, query, sub_steps)
            pooled = state.pool.add(chunks)
            pooled = rerank(sub_q, pooled)

            relevant_ids = grade_relevance(sub_q, pooled)
            new_relevant = state.pool.mark_relevant(sub_q, relevant_ids)
            streak = 0 if new_relevant else streak + 1

            verdict = grade_sufficiency(sub_q, state.pool.relevant_for(sub_q))
            missing = verdict.missing

            state.attempts.append(
                Attempt(sub_q, query, tools_used, len(pooled), len(relevant_ids),
                        verdict.sufficient, "; ".join(errors) or None)
            )
            trace.log(
                "attempt", sub_question=sub_q, query=query, tools=tools_used,
                retrieved=len(pooled), relevant=len(relevant_ids), new_relevant=new_relevant,
                sufficient=verdict.sufficient, missing=missing, errors=errors,
            )

            decision = check_stop(
                sufficient=verdict.sufficient, sub_steps=sub_steps, total_steps=state.total_steps,
                tokens_used=tracker.total_tokens, no_progress_streak=streak,
            )
            if decision.stop:
                trace.log("stop", sub_question=sub_q, reason=decision.reason)
                return verdict.sufficient

    def _sources_for(self, state: AgentState, sub_steps: int) -> set[str]:
        sources = {"docs"}
        if state.source in settings.enabled_sources:
            sources.add(state.source)
        if sub_steps >= 2 and "web" in settings.enabled_sources:  # fallback after a failed attempt
            sources.add("web")
        return sources

    def _search(self, state, sub_q, query, sub_steps):
        past = [a for a in state.attempts if a.sub_question == sub_q]
        past_text = "\n".join(
            f"- tools={a.tools} query={a.query!r} -> {a.n_relevant}/{a.n_retrieved} relevant" for a in past
        ) or "none"
        resp = self.llm.complete(
            system=TOOL_SYSTEM,
            messages=[{"role": "user", "content": (
                f"Sub-question: {sub_q}\nProposed query: {query}\nPrevious attempts:\n{past_text}"
            )}],
            model=settings.grader_model,
            max_tokens=400,
            tools=tool_specs(self._sources_for(state, sub_steps)),
            tool_choice={"type": "any"},
        )
        calls = [(b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"][:2]
        if not calls:
            calls = [("vector_search", {"query": query})]

        tools_used, chunks, errors = [], [], []
        for name, args in calls:
            tools_used.append(name)
            if settings.use_hyde and name == "vector_search":
                args["query"] = hypothetical_passage(args.get("query", query))
            found, err = run_tool(name, args)
            chunks.extend(found)
            if err:
                errors.append(f"{name}: {err}")
            if settings.use_query_expansion and name in TEXT_TOOLS and "query" in args:
                for alt in expand(args["query"]):
                    more, _ = run_tool(name, {**args, "query": alt})
                    chunks.extend(more)
        return tools_used, chunks, errors

    # ------------------------------------------------------------ draft/verify
    def _draft_and_verify(self, state: AgentState, trace: TraceLogger) -> tuple[str, bool]:
        question = state.standalone or state.question
        queries = [a.query for a in state.attempts]

        evidence = state.pool.all_relevant(settings.max_evidence_chunks)
        if not evidence:
            trace.log("abstain", reason="no relevant evidence")
            return abstain_message(question, queries), True

        answer = generate_answer(question, state.history, evidence)
        trace.log("draft", answer=answer, n_evidence=len(evidence))
        if not settings.enable_verification:
            return answer, False

        for round_no in range(settings.max_revisions + 1):
            claims = extract_claims(answer)
            report = check_groundedness(claims, state.pool)
            cites = validate_citations(answer, state.pool)
            trace.log(
                "verify", round=round_no, n_claims=len(claims),
                problems=[asdict(v) for v in report.unsupported],
                invalid_citations=cites.invalid_labels, uncited_sentences=len(cites.uncited_sentences),
            )
            if report.all_supported and not cites.invalid_labels:
                return answer, False
            if round_no == settings.max_revisions:
                break

            rev = revise(question, answer, report, state.pool.all_relevant(settings.max_evidence_chunks))
            trace.log("revise", action=rev.action, query=rev.query)
            if rev.action == "abstain":
                return abstain_message(question, queries), True
            if rev.action == "retrieve_more" and rev.query and state.total_steps < settings.max_total_steps:
                self._research(state, rev.query, trace)
                answer = generate_answer(question, state.history, state.pool.all_relevant(settings.max_evidence_chunks))
            elif rev.answer:
                answer = rev.answer

        trace.log("verify_incomplete")
        return answer + "\n\n_Note: some statements could not be fully verified against the sources._", False

    # ------------------------------------------------------------------ finish
    def _finish(self, state: AgentState, trace: TraceLogger, answer: str, abstained: bool) -> AgentResult:
        cost = tracker.summary()
        cited = sorted(set(CITE.findall(answer)))
        sources = []
        for label in cited:
            chunk = state.pool.get(label)
            if chunk:
                sources.append({"label": label, "source": chunk.source, "preview": chunk.text[:200]})
        trace.log("final", abstained=abstained, steps=state.total_steps)
        path = trace.save({"answer": answer, "abstained": abstained, "cost": cost})
        return AgentResult(answer, abstained, sources, state.total_steps, cost, trace.run_id, str(path))