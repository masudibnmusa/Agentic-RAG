from app.config import settings
from app.grading.relevance_grader import grade_relevance


def test_empty_chunks_returns_empty_set_without_llm_call(fake_llm):
    assert grade_relevance("question?", []) == set()
    assert fake_llm.calls == []


def test_returns_only_chunks_the_llm_marks_relevant(fake_llm, make_chunks):
    _, chunks = make_chunks(("Refunds take 30 days",), ("Shipping is free",), ("Refund window is 30 days",))
    fake_llm.on_json("grade search results", {"relevant": ["S1", "S3"]})
    assert grade_relevance("What is the refund window?", chunks) == {chunks[0].id, chunks[2].id}


def test_nothing_relevant(fake_llm, make_chunks):
    _, chunks = make_chunks(("Shipping is free",))
    fake_llm.on_json("grade search results", {"relevant": []})
    assert grade_relevance("What is the refund window?", chunks) == set()


def test_ignores_unknown_labels_and_non_strings(fake_llm, make_chunks):
    _, chunks = make_chunks(("a",), ("b",))
    fake_llm.on_json("grade search results", {"relevant": ["S2", "S99", 5]})
    assert grade_relevance("q?", chunks) == {chunks[1].id}


def test_malformed_llm_output_is_treated_as_no_relevant_chunks(fake_llm, make_chunks):
    _, chunks = make_chunks(("a",))
    fake_llm.on_json("grade search results", {})
    assert grade_relevance("q?", chunks) == set()


def test_question_and_chunk_text_are_sent_to_the_grader(fake_llm, make_chunks):
    _, chunks = make_chunks(("Refunds take 30 days",))
    fake_llm.on_json("grade search results", {"relevant": []})
    grade_relevance("What is the refund window?", chunks)
    prompt = fake_llm.calls[-1][2]
    assert "What is the refund window?" in prompt
    assert "Refunds take 30 days" in prompt


def test_grading_disabled_keeps_every_chunk(fake_llm, make_chunks, monkeypatch):
    monkeypatch.setattr(settings, "enable_grading", False)
    _, chunks = make_chunks(("a",), ("b",))
    assert grade_relevance("q?", chunks) == {c.id for c in chunks}
    assert fake_llm.calls == []


def test_reranker_mode_uses_score_threshold_without_llm(fake_llm, make_chunks, monkeypatch):
    monkeypatch.setattr(settings, "grader_mode", "reranker")
    monkeypatch.setattr(settings, "relevance_threshold", 0.5)
    _, chunks = make_chunks(("a", 0.9), ("b", 0.4), ("c", 0.5))
    assert grade_relevance("q?", chunks) == {chunks[0].id, chunks[2].id}
    assert fake_llm.calls == []