"""Core Golden Path Strands Framework"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import structlog
from .exploration import ExplorationLogger
from .evaluation import LLMJudgeEvaluator
from .ollama_provider import OllamaProvider

logger = structlog.get_logger()


class GoldenPathStrands:
    """Main orchestrator for the Golden Path framework"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.exploration_logger = ExplorationLogger()
        self.evaluator = LLMJudgeEvaluator(config=self.config)
        self.ollama_provider = OllamaProvider(
            host=self.config.get("ollama_host", "http://localhost:11434"),
            model=self.config.get("ollama_model", "gpt-oss:20b")
        )
        self.discovered_paths = []
        
    def _default_config(self) -> Dict:
        return {
            "ollama_host": "http://localhost:11434",
            "ollama_model": "gpt-oss:20b",
            "min_success_score": 0.8,
            "evaluation_threshold": 0.85,
            "sample_size": 100,
            "iterations": 3,
        }
    
    async def discover_golden_paths(
        self, 
        tasks: List[str], 
        iterations: int = None
    ) -> Dict[str, Any]:
        """
        Discover optimal paths through task exploration
        """
        iterations = iterations or self.config["iterations"]
        results = {
            "discovered_paths": 0,
            "successful_paths": [],
            "average_score": 0.0,
            "tasks_completed": 0,
        }
        
        for task in tasks:
            logger.info("exploring_task", task=task)
            
            for i in range(iterations):
                path_id = f"{task[:20]}_{i}_{datetime.now().timestamp()}"
                
                # Explore using the agent
                exploration_result = await self._explore_task(task, path_id)
                
                # Log the exploration
                self.exploration_logger.log_decision_point(
                    exploration_result, 
                    {"task": task, "iteration": i}
                )
                
                # Evaluate the result
                evaluation = await self.evaluator.evaluate_response(
                    query=task,
                    response=exploration_result.get("response", ""),
                    tools=exploration_result.get("tools_used", [])
                )
                
                # Check if path is successful
                if evaluation["score"] >= self.config["min_success_score"]:
                    self.exploration_logger.mark_path_successful(
                        score=evaluation["score"],
                        metadata={"task": task, "evaluation": evaluation}
                    )
                    results["successful_paths"].append({
                        "task": task,
                        "path_id": path_id,
                        "score": evaluation["score"],
                        "exploration": exploration_result,
                    })
                    results["discovered_paths"] += 1
                    
            results["tasks_completed"] += 1
        
        # Calculate average score
        if results["successful_paths"]:
            results["average_score"] = (
                sum(p["score"] for p in results["successful_paths"]) / len(results["successful_paths"])
                if results["successful_paths"] else 0
            )
        
        self.discovered_paths = results["successful_paths"]
        return results
    
    async def _explore_task(self, task: str, path_id: str) -> Dict:
        """
        Explore a single task using the configured model
        """
        try:
            response = await self.ollama_provider.generate(
                prompt=self._create_exploration_prompt(task),
                system_prompt="You are an expert problem-solving agent. Break down the task into steps and execute them systematically."
            )
            
            return {
                "path_id": path_id,
                "task": task,
                "response": response.get("response", ""),
                "reasoning": response.get("reasoning", ""),
                "tools_used": response.get("tools", []),
                "tokens_used": response.get("total_tokens", 0),
                "model": self.config["ollama_model"],
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error("exploration_failed", task=task, error=str(e))
            return {
                "path_id": path_id,
                "task": task,
                "response": "",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
    
    def _create_exploration_prompt(self, task: str) -> str:
        """Create a structured prompt for task exploration"""
        return f"""
Task: {task}

Please complete this task step by step. For each step:
1. Explain your reasoning
2. Identify any tools or resources needed
3. Execute the step
4. Verify the result

Provide a comprehensive solution with clear reasoning at each decision point.
"""
    
    async def create_training_dataset(self, paths: Optional[List[Dict]] = None) -> str:
        """
        Create JSONL training dataset from successful paths
        """
        paths = paths or self.discovered_paths
        
        if not paths:
            logger.warning("no_paths_available")
            return ""
        
        # Ensure datasets directory exists
        import os
        os.makedirs("datasets", exist_ok=True)

        dataset_path = f"datasets/golden_paths_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        with open(dataset_path, 'w') as f:
            for path in paths:
                training_example = {
                    "instruction": path["task"],
                    "input": "",
                    "output": path["exploration"]["response"],
                    "metadata": {
                        "score": path["score"],
                        "model": path["exploration"]["model"],
                        "tools": path["exploration"].get("tools_used", []),
                    }
                }
                f.write(json.dumps(training_example) + '\n')
        
        logger.info("dataset_created", path=dataset_path, examples=len(paths))
        return dataset_path
    
    async def run_complete_pipeline(self, tasks: List[str]) -> Dict[str, Any]:
        """
        Run the complete golden path discovery and refinement pipeline
        """
        logger.info("pipeline_started", tasks_count=len(tasks))
        
        # Phase 1: Discover golden paths
        discovery_results = await self.discover_golden_paths(tasks)
        
        # Phase 2: Create training dataset
        if discovery_results["discovered_paths"] > 0:
            dataset_path = await self.create_training_dataset()
            discovery_results["dataset_path"] = dataset_path
        
        # Phase 3: Get successful paths for review
        successful_paths = self.exploration_logger.get_successful_paths()
        discovery_results["reviewed_paths"] = len(successful_paths)
        
        logger.info("pipeline_completed", results=discovery_results)
        return discovery_results