from app.verification.claim_extractor import Claim
from app.verification.groundedness_checker import check_groundedness

KEY = "verify claims against evidence"


def test_no_claims_means_nothing_to_flag_and_no_llm_call(fake_llm, make_chunks):
    pool, _ = make_chunks(("Alpha fact",))
    report = check_groundedness([], pool)
    assert report.all_supported is True
    assert report.verdicts == []
    assert fake_llm.calls == []


def test_all_claims_supported(fake_llm, make_chunks):
    pool, _ = make_chunks(("Refunds are available within 30 days",))
    fake_llm.on_json(KEY, {"verdicts": [{"id": 1, "verdict": "supported", "reason": "stated"}]})
    report = check_groundedness([Claim("Refunds last 30 days", ["S1"])], pool)
    assert report.all_supported is True
    assert report.unsupported == []


def test_flags_unsupported_and_contradicted_claims(fake_llm, make_chunks):
    pool, _ = make_chunks(("Refunds are available within 30 days",))
    fake_llm.on_json(KEY, {"verdicts": [
        {"id": 1, "verdict": "supported", "reason": ""},
        {"id": 2, "verdict": "unsupported", "reason": "not in evidence"},
        {"id": 3, "verdict": "contradicted", "reason": "says 30 not 60"},
    ]})
    claims = [Claim("Refunds last 30 days", ["S1"]), Claim("Refunds are instant", ["S1"]), Claim("Refunds last 60 days", ["S1"])]
    report = check_groundedness(claims, pool)
    assert report.all_supported is False
    assert [v.verdict for v in report.unsupported] == ["unsupported", "contradicted"]
    assert report.unsupported[0].claim == "Refunds are instant"


def test_claim_missing_from_llm_output_defaults_to_unsupported(fake_llm, make_chunks):
    pool, _ = make_chunks(("Alpha fact",))
    fake_llm.on_json(KEY, {"verdicts": [{"id": 1, "verdict": "supported"}]})
    report = check_groundedness([Claim("one", ["S1"]), Claim("two", ["S1"])], pool)
    assert report.verdicts[1].verdict == "unsupported"
    assert report.all_supported is False


def test_garbage_llm_output_is_never_treated_as_supported(fake_llm, make_chunks):
    pool, _ = make_chunks(("Alpha fact",))
    fake_llm.on_json(KEY, {})
    report = check_groundedness([Claim("one", ["S1"])], pool)
    assert report.all_supported is False


def test_only_cited_chunks_are_sent_to_the_checker(fake_llm, make_chunks):
    pool, _ = make_chunks(("Alpha fact about refunds",), ("Beta fact about shipping",))
    fake_llm.on_json(KEY, {"verdicts": [{"id": 1, "verdict": "supported"}]})
    check_groundedness([Claim("Refunds exist", ["S1"])], pool)
    prompt = fake_llm.calls[-1][2]
    assert "Alpha fact" in prompt
    assert "Beta fact" not in prompt


def test_uncited_claim_falls_back_to_all_relevant_evidence(fake_llm, make_chunks):
    pool, chunks = make_chunks(("Alpha fact about refunds",), ("Beta fact about shipping",))
    pool.mark_relevant("q", {c.id for c in chunks})
    fake_llm.on_json(KEY, {"verdicts": [{"id": 1, "verdict": "supported"}]})
    check_groundedness([Claim("Something with no citation", [])], pool)
    prompt = fake_llm.calls[-1][2]
    assert "Alpha fact" in prompt
    assert "Beta fact" in prompt