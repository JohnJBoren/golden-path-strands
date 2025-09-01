import asyncio

from golden_path_strands.evaluation import LLMJudgeEvaluator


def test_consensus_score() -> None:
    ev = LLMJudgeEvaluator()
    evaluations = asyncio.run(ev.evaluate_response("q", "resp", []))
    score = ev.get_consensus_score(evaluations)
    assert 0.0 <= score <= 1.0
