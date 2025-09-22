"""Core orchestrator for the Golden Path Strands framework."""
from __future__ import annotations

from typing import Dict, List, Any
from datetime import datetime
import json

from .logging import ExplorationLogger
from .evaluation import LLMJudgeEvaluator
from .dataset import DatasetCreator
from .config import load_config
from .ollama_provider import OllamaProvider
from .agent_orchestrator import AgentOrchestrator


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
        if config is None:
            settings = load_config()
            self.config = settings.to_dict()
        else:
            self.config = config

        self.logger = ExplorationLogger()
        self.evaluator = LLMJudgeEvaluator()
        self.dataset_creator = DatasetCreator()

        # Initialize Ollama provider
        ollama_host = self.config.get("ollama_host", "http://localhost:11434")
        ollama_model = self.config.get("ollama_model", "gpt-oss:20b")
        self.ollama_provider = OllamaProvider(host=ollama_host, model=ollama_model)

        # Initialize agent orchestrator
        self.agent_orchestrator = AgentOrchestrator(
            ollama_host=ollama_host,
            model=ollama_model
        )

    async def discover_golden_paths(self, tasks: List[str], iterations: int = 3) -> List[Dict[str, Any]]:
        """Explore tasks using Ollama agents to discover optimal paths.

        This implementation uses the Ollama provider with GPT-OSS model to explore
        the task space and find successful execution paths.
        """
        paths: List[Dict[str, Any]] = []
        min_success_score = self.config.get("min_success_score", 0.8)
        
        for task in tasks:
            for i in range(iterations):
                path_id = f"{task[:20]}_{i}_{datetime.now().timestamp()}"
                
                # Use agent orchestrator to explore the task
                try:
                    result = await self.agent_orchestrator.execute_single(
                        "researcher",
                        task,
                        context={"iteration": i, "depth": "comprehensive"}
                    )
                    
                    decision = {
                        "task": task,
                        "iteration": i,
                        "result": result.get("response", ""),
                        "path_id": path_id,
                        "agent": result.get("agent", "researcher"),
                        "tokens": result.get("tokens", 0)
                    }
                    
                    self.logger.log_decision_point(decision, {"task": task})
                    
                    # Evaluate the result
                    evaluation = await self.evaluator.evaluate_response(
                        task, 
                        decision["result"], 
                        []
                    )
                    score = self.evaluator.get_consensus_score(evaluation)
                    
                    if score >= min_success_score:
                        self.logger.mark_path_successful(score, decision)
                        paths.append({
                            "task": task,
                            "score": score,
                            "decision": decision,
                            "evaluation": evaluation
                        })
                except Exception as e:
                    print(f"Error exploring task '{task}': {e}")
                    continue
                    
        return paths

    async def create_training_dataset(self, paths: List[Dict[str, Any]]) -> str:
        """Persist successful paths as a JSONL dataset.

        Returns the path to the created dataset file.
        """
        return self.dataset_creator.write_paths(paths)

    async def run_complete_pipeline(self, tasks: List[str]) -> Dict[str, Any]:
        """Execute exploration, evaluation and dataset creation end-to-end."""
        paths = await self.discover_golden_paths(tasks)
        dataset_path = await self.create_training_dataset(paths)

        # Calculate statistics for the return value
        total_score = sum(p.get("score", 0) for p in paths)
        avg_score = total_score / len(paths) if paths else 0

        return {
            "tasks_completed": len(tasks),
            "discovered_paths": len(paths),
            "average_score": avg_score,
            "dataset_path": dataset_path,
            "paths": paths
        }
