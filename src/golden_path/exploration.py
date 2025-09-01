"""Exploration Logger for tracking decision paths"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import structlog

logger = structlog.get_logger()


class ExplorationLogger:
    """Comprehensive logging system for path discovery"""
    
    def __init__(self, log_dir: str = "logs/explorations"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = self.log_dir / f"session_{self.current_session}.jsonl"
        self.decision_points = []
        self.successful_paths = []
        
    def log_decision_point(self, agent_result: Dict, context: Dict) -> None:
        """Log a decision point in the exploration"""
        decision = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.current_session,
            "event_type": "decision_point",
            "agent_result": agent_result,
            "context": context,
            "path_id": agent_result.get("path_id"),
        }
        
        self.decision_points.append(decision)
        self._write_to_log(decision)
        
        logger.info(
            "decision_logged",
            path_id=agent_result.get("path_id"),
            task=context.get("task")
        )
    
    def mark_path_successful(self, score: float, metadata: Dict) -> None:
        """Mark a path as successful based on evaluation"""
        successful_path = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.current_session,
            "event_type": "successful_path",
            "score": score,
            "metadata": metadata,
            "decision_count": len([
                d for d in self.decision_points 
                if d.get("context", {}).get("task") == metadata.get("task")
            ])
        }
        
        self.successful_paths.append(successful_path)
        self._write_to_log(successful_path)
        
        logger.info(
            "path_marked_successful",
            score=score,
            task=metadata.get("task")
        )
    
    def get_successful_paths(self, min_score: Optional[float] = None) -> List[Dict]:
        """Retrieve successful paths, optionally filtered by minimum score"""
        if min_score is None:
            return self.successful_paths
        
        return [
            path for path in self.successful_paths
            if path["score"] >= min_score
        ]
    
    def get_decision_history(self, path_id: Optional[str] = None) -> List[Dict]:
        """Get decision history for a specific path or all paths"""
        if path_id is None:
            return self.decision_points
        
        return [
            decision for decision in self.decision_points
            if decision.get("path_id") == path_id
        ]
    
    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in successful vs unsuccessful paths"""
        analysis = {
            "total_decisions": len(self.decision_points),
            "successful_paths": len(self.successful_paths),
            "average_score": 0.0,
            "common_patterns": [],
            "failure_points": [],
        }
        
        if self.successful_paths:
            scores = [path["score"] for path in self.successful_paths]
            analysis["average_score"] = sum(scores) / len(scores)
            
            # Analyze common patterns in successful paths
            successful_tasks = [
                path["metadata"].get("task") 
                for path in self.successful_paths
            ]
            
            # Count task frequencies
            task_freq = {}
            for task in successful_tasks:
                task_key = task[:50] if task else "unknown"
                task_freq[task_key] = task_freq.get(task_key, 0) + 1
            
            analysis["common_patterns"] = [
                {"task": task, "frequency": freq}
                for task, freq in sorted(
                    task_freq.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
            ]
        
        return analysis
    
    def export_golden_paths(self, output_file: str = None) -> str:
        """Export successful paths for training"""
        output_file = output_file or f"golden_paths_{self.current_session}.json"
        
        export_data = {
            "session_id": self.current_session,
            "timestamp": datetime.now().isoformat(),
            "successful_paths": self.successful_paths,
            "statistics": self.analyze_patterns(),
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info("golden_paths_exported", file=output_file, count=len(self.successful_paths))
        return output_file
    
    def _write_to_log(self, entry: Dict) -> None:
        """Write entry to session log file"""
        with open(self.session_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')