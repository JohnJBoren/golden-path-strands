"""LLM Judge Evaluator for ensemble evaluation"""

import asyncio
from typing import Dict, List, Any, Optional
from statistics import mean, median
import structlog
from .ollama_provider import OllamaProvider

logger = structlog.get_logger()


class LLMJudgeEvaluator:
    """Ensemble evaluation system using multiple LLM judges"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.judges = self._initialize_judges()
        self.voting_method = self.config.get("voting_method", "weighted_average")
        self.consensus_threshold = self.config.get("consensus_threshold", 0.8)
        
    def _initialize_judges(self) -> List[Dict]:
        """Initialize judge models based on configuration"""
        judge_configs = self.config.get("judge_models", [
            {"type": "ollama", "model_id": "gpt-oss:20b", "weight": 1.0}
        ])
        
        judges = []
        for judge_config in judge_configs:
            if judge_config["type"] == "ollama":
                provider = OllamaProvider(
                    host=judge_config.get("host", "http://localhost:11434"),
                    model=judge_config["model_id"]
                )
                judges.append({
                    "provider": provider,
                    "weight": judge_config.get("weight", 1.0),
                    "model_id": judge_config["model_id"],
                })
        
        return judges
    
    async def evaluate_response(
        self,
        query: str,
        response: str,
        tools: List[str] = None,
        criteria: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Evaluate a response using ensemble of judges"""
        
        criteria = criteria or self._default_criteria()
        evaluation_prompt = self._create_evaluation_prompt(query, response, tools, criteria)
        
        # Collect evaluations from all judges
        evaluations = await self._collect_evaluations(evaluation_prompt)
        
        # Calculate consensus score
        consensus_score = self.get_consensus_score(evaluations)
        
        # Determine if response meets quality threshold
        is_successful = consensus_score >= self.consensus_threshold
        
        return {
            "score": consensus_score,
            "is_successful": is_successful,
            "evaluations": evaluations,
            "criteria": criteria,
            "consensus_method": self.voting_method,
        }
    
    def _default_criteria(self) -> Dict:
        """Default evaluation criteria"""
        return {
            "correctness": {
                "weight": 0.3,
                "description": "Is the response factually correct and accurate?"
            },
            "completeness": {
                "weight": 0.25,
                "description": "Does the response fully address the query?"
            },
            "clarity": {
                "weight": 0.2,
                "description": "Is the response clear and well-organized?"
            },
            "efficiency": {
                "weight": 0.15,
                "description": "Is the approach efficient and well-reasoned?"
            },
            "tool_usage": {
                "weight": 0.1,
                "description": "Are tools used appropriately and effectively?"
            }
        }
    
    def _create_evaluation_prompt(
        self,
        query: str,
        response: str,
        tools: List[str],
        criteria: Dict
    ) -> str:
        """Create evaluation prompt for judges"""
        
        criteria_text = "\n".join([
            f"- {name} (weight: {info['weight']}): {info['description']}"
            for name, info in criteria.items()
        ])
        
        tools_text = "\n".join([f"- {tool}" for tool in (tools or [])])
        
        return f"""
Evaluate the following response to a query based on the given criteria.

QUERY: {query}

RESPONSE: {response}

TOOLS USED:
{tools_text if tools_text else "None"}

EVALUATION CRITERIA:
{criteria_text}

Please evaluate the response on a scale of 0.0 to 1.0 for each criterion.
Provide your evaluation in the following JSON format:

{{
    "scores": {{
        "correctness": 0.0-1.0,
        "completeness": 0.0-1.0,
        "clarity": 0.0-1.0,
        "efficiency": 0.0-1.0,
        "tool_usage": 0.0-1.0
    }},
    "overall_score": 0.0-1.0,
    "reasoning": "Brief explanation of your evaluation"
}}
"""
    
    async def _collect_evaluations(self, prompt: str) -> List[Dict]:
        """Collect evaluations from all judge models"""
        evaluations = []
        
        for judge in self.judges:
            try:
                response = await judge["provider"].generate(
                    prompt=prompt,
                    system_prompt="You are an expert evaluator. Provide objective, detailed evaluations.",
                    temperature=0.3  # Lower temperature for more consistent evaluations
                )
                
                # Parse the evaluation response
                evaluation = self._parse_evaluation(response.get("response", ""))
                evaluation["judge_model"] = judge["model_id"]
                evaluation["weight"] = judge["weight"]
                evaluations.append(evaluation)
                
            except Exception as e:
                logger.error("judge_evaluation_failed", judge=judge["model_id"], error=str(e))
                
        return evaluations
    
    def _parse_evaluation(self, response_text: str) -> Dict:
        """Parse evaluation response from judge"""
        import json
        import re
        
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fallback: extract scores using patterns
        scores = {}
        score_pattern = r'(\w+):\s*([\d.]+)'
        matches = re.findall(score_pattern, response_text)
        
        for criterion, score in matches:
            try:
                scores[criterion.lower()] = float(score)
            except ValueError:
                continue
        
        # Calculate overall if not present
        if scores and "overall_score" not in scores:
            scores["overall_score"] = mean(scores.values())
        
        return {
            "scores": scores,
            "overall_score": scores.get("overall_score", 0.5),
            "reasoning": response_text[:200]
        }
    
    def get_consensus_score(self, evaluations: List[Dict]) -> float:
        """Calculate consensus score based on voting method"""
        if not evaluations:
            return 0.0
        
        scores = [eval.get("overall_score", 0.0) for eval in evaluations]
        weights = [eval.get("weight", 1.0) for eval in evaluations]
        
        if self.voting_method == "weighted_average":
            total_weight = sum(weights)
            if total_weight == 0:
                return 0.0
            weighted_sum = sum(s * w for s, w in zip(scores, weights))
            return weighted_sum / total_weight
            
        elif self.voting_method == "majority":
            threshold_scores = [1.0 if s >= 0.5 else 0.0 for s in scores]
            return mean(threshold_scores)
            
        elif self.voting_method == "median":
            return median(scores)
            
        else:  # Simple average
            return mean(scores)
    
    async def batch_evaluate(
        self,
        items: List[Dict[str, Any]],
        batch_size: int = 5
    ) -> List[Dict[str, Any]]:
        """Evaluate multiple items in batches"""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_tasks = [
                self.evaluate_response(
                    query=item["query"],
                    response=item["response"],
                    tools=item.get("tools", [])
                )
                for item in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
            
            logger.info(
                "batch_evaluated",
                batch_num=i // batch_size + 1,
                total_batches=(len(items) + batch_size - 1) // batch_size
            )
        
        return results