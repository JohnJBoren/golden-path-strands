"""Core orchestrator for the Golden Path Strands framework."""
from __future__ import annotations

from typing import Dict, List, Any

from .logging import ExplorationLogger
from .evaluation import LLMJudgeEvaluator
from .dataset import DatasetCreator
from .config import load_config


class GoldenPathStrands:
    """High level orchestrator coordinating exploration, evaluation and dataset creation."""

    def __init__(self, config: Dict | None = None) -> None:
        """Create a new orchestrator instance.

        Parameters
        ----------
        config:
            Optional configuration dictionary. If ``None`` a configuration file and
            environment variables will be consulted.
        """
        self.config = config or load_config()
        self.logger = ExplorationLogger()
        self.evaluator = LLMJudgeEvaluator()
        self.dataset_creator = DatasetCreator()

    async def discover_golden_paths(self, tasks: List[str], iterations: int = 3) -> List[Dict[str, Any]]:
        """Stub exploration routine.

        The current implementation simply logs the provided tasks and returns a list of
        mock path dictionaries. In a full system this would coordinate multiple agents
        exploring the search space.
        """
        paths: List[Dict[str, Any]] = []
        for task in tasks:
            for i in range(iterations):
                decision = {"task": task, "iteration": i, "result": f"result-{i}"}
                self.logger.log_decision_point(decision, {"task": task})
                # pretend each iteration produces a path
                evaluation = await self.evaluator.evaluate_response(task, decision["result"], [])
                score = self.evaluator.get_consensus_score(evaluation)
                if score >= 0:
                    self.logger.mark_path_successful(score, decision)
                    paths.append({"task": task, "score": score, "decision": decision})
        return paths

    async def create_training_dataset(self, paths: List[Dict[str, Any]]) -> str:
        """Persist successful paths as a JSONL dataset.

        Returns the path to the created dataset file.
        """
        return self.dataset_creator.write_paths(paths)

    async def run_complete_pipeline(self, tasks: List[str]) -> str:
        """Execute exploration, evaluation and dataset creation end-to-end."""
        paths = await self.discover_golden_paths(tasks)
        dataset_path = await self.create_training_dataset(paths)
        return dataset_path
