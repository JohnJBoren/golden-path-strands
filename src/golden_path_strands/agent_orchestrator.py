"""Agent Orchestrator for managing research and coding agents"""

import asyncio
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import structlog
from .ollama_provider import OllamaProvider

logger = structlog.get_logger()


class AgentType(Enum):
    RESEARCH = "research"
    CODING = "coding"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    TESTING = "testing"


class Agent:
    """Base agent class"""
    
    def __init__(
        self,
        name: str,
        agent_type: AgentType,
        provider: OllamaProvider,
        system_prompt: str = None
    ):
        self.name = name
        self.agent_type = agent_type
        self.provider = provider
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.history = []
        
    def _default_system_prompt(self) -> str:
        prompts = {
            AgentType.RESEARCH: "You are a research agent. Your role is to gather information, analyze sources, and provide comprehensive research summaries.",
            AgentType.CODING: "You are a coding agent. Your role is to write, review, and optimize code. Follow best practices and write clean, maintainable code.",
            AgentType.ANALYSIS: "You are an analysis agent. Your role is to analyze data, identify patterns, and provide insights.",
            AgentType.PLANNING: "You are a planning agent. Your role is to break down complex tasks, create project plans, and organize workflows.",
            AgentType.TESTING: "You are a testing agent. Your role is to create test cases, identify edge cases, and ensure code quality.",
        }
        return prompts.get(self.agent_type, "You are a helpful assistant.")
    
    async def execute(self, task: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute a task"""
        prompt = self._build_prompt(task, context)
        
        response = await self.provider.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.7 if self.agent_type == AgentType.RESEARCH else 0.5
        )
        
        result = {
            "agent": self.name,
            "type": self.agent_type.value,
            "task": task,
            "response": response.get("response", ""),
            "tokens": response.get("total_tokens", 0),
            "context": context,
        }
        
        self.history.append(result)
        return result
    
    def _build_prompt(self, task: str, context: Optional[Dict]) -> str:
        """Build prompt with context"""
        prompt_parts = [f"Task: {task}"]
        
        if context:
            if "previous_results" in context:
                prompt_parts.append("\nPrevious Results:")
                for prev in context["previous_results"]:
                    prompt_parts.append(f"- {prev.get('agent', 'Unknown')}: {prev.get('response', '')[:200]}...")
            
            if "requirements" in context:
                prompt_parts.append(f"\nRequirements: {context['requirements']}")
            
            if "constraints" in context:
                prompt_parts.append(f"\nConstraints: {context['constraints']}")
        
        return "\n".join(prompt_parts)


class AgentOrchestrator:
    """Orchestrator for managing multiple agents"""
    
    def __init__(self, ollama_host: str = "http://localhost:11434", model: str = "gpt-oss:20b"):
        self.provider = OllamaProvider(host=ollama_host, model=model)
        self.agents = {}
        self.workflows = {}
        self._initialize_default_agents()
        
    def _initialize_default_agents(self):
        """Initialize default agents"""
        default_agents = [
            ("researcher", AgentType.RESEARCH),
            ("coder", AgentType.CODING),
            ("analyst", AgentType.ANALYSIS),
            ("planner", AgentType.PLANNING),
            ("tester", AgentType.TESTING),
        ]
        
        for name, agent_type in default_agents:
            self.register_agent(name, agent_type)
    
    def register_agent(
        self,
        name: str,
        agent_type: AgentType,
        system_prompt: Optional[str] = None
    ) -> Agent:
        """Register a new agent"""
        agent = Agent(name, agent_type, self.provider, system_prompt)
        self.agents[name] = agent
        logger.info("agent_registered", name=name, type=agent_type.value)
        return agent
    
    async def execute_single(
        self,
        agent_name: str,
        task: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Execute a task with a single agent"""
        if agent_name not in self.agents:
            raise ValueError(f"Agent {agent_name} not found")
        
        agent = self.agents[agent_name]
        return await agent.execute(task, context)
    
    async def execute_parallel(
        self,
        tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute multiple tasks in parallel"""
        async_tasks = []
        
        for task_spec in tasks:
            agent_name = task_spec["agent"]
            task = task_spec["task"]
            context = task_spec.get("context")
            
            if agent_name in self.agents:
                async_tasks.append(
                    self.execute_single(agent_name, task, context)
                )
        
        results = await asyncio.gather(*async_tasks)
        return results
    
    async def execute_sequential(
        self,
        workflow: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute tasks sequentially, passing results forward"""
        results = []
        context = {"previous_results": []}
        
        for step in workflow:
            agent_name = step["agent"]
            task = step["task"]
            
            # Add previous results to context
            step_context = {**context, **step.get("context", {})}
            
            result = await self.execute_single(agent_name, task, step_context)
            results.append(result)
            
            # Update context with latest result
            context["previous_results"].append({
                "agent": agent_name,
                "response": result["response"]
            })
        
        return results
    
    async def execute_research_workflow(
        self,
        topic: str,
        depth: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Execute a research workflow"""
        workflow = [
            {
                "agent": "planner",
                "task": f"Create a research plan for: {topic}. Depth level: {depth}",
                "context": {"requirements": f"Research depth: {depth}"}
            },
            {
                "agent": "researcher",
                "task": f"Conduct research on: {topic}",
                "context": {"depth": depth}
            },
            {
                "agent": "analyst",
                "task": "Analyze the research findings and identify key insights",
            },
            {
                "agent": "researcher",
                "task": "Create a comprehensive summary with actionable recommendations",
            }
        ]
        
        results = await self.execute_sequential(workflow)
        
        return {
            "topic": topic,
            "workflow": "research",
            "results": results,
            "summary": results[-1]["response"] if results else "",
        }
    
    async def execute_coding_workflow(
        self,
        requirements: str,
        language: str = "python",
        include_tests: bool = True
    ) -> Dict[str, Any]:
        """Execute a coding workflow"""
        workflow = [
            {
                "agent": "planner",
                "task": f"Create implementation plan for: {requirements}",
                "context": {"language": language, "requirements": requirements}
            },
            {
                "agent": "coder",
                "task": f"Implement the solution in {language}",
                "context": {"requirements": requirements, "language": language}
            }
        ]
        
        if include_tests:
            workflow.append({
                "agent": "tester",
                "task": "Create comprehensive test cases for the implementation",
                "context": {"language": language}
            })
            workflow.append({
                "agent": "coder",
                "task": "Review and optimize the code based on test requirements",
            })
        
        results = await self.execute_sequential(workflow)
        
        # Extract code from results
        code = ""
        tests = ""
        for result in results:
            if result["agent"] == "coder" and "```" in result["response"]:
                code = result["response"]
            elif result["agent"] == "tester" and "```" in result["response"]:
                tests = result["response"]
        
        return {
            "requirements": requirements,
            "language": language,
            "workflow": "coding",
            "results": results,
            "code": code,
            "tests": tests if include_tests else None,
        }
    
    async def execute_analysis_workflow(
        self,
        data_description: str,
        analysis_type: str = "exploratory"
    ) -> Dict[str, Any]:
        """Execute a data analysis workflow"""
        workflow = [
            {
                "agent": "analyst",
                "task": f"Plan {analysis_type} analysis for: {data_description}",
                "context": {"analysis_type": analysis_type}
            },
            {
                "agent": "coder",
                "task": "Write code for data processing and analysis",
                "context": {"language": "python", "libraries": "pandas, numpy, matplotlib"}
            },
            {
                "agent": "analyst",
                "task": "Interpret results and provide insights",
            },
            {
                "agent": "researcher",
                "task": "Create final report with visualizations and recommendations",
            }
        ]
        
        results = await self.execute_sequential(workflow)
        
        return {
            "data_description": data_description,
            "analysis_type": analysis_type,
            "workflow": "analysis",
            "results": results,
            "insights": results[-1]["response"] if results else "",
        }
    
    def register_custom_workflow(
        self,
        name: str,
        workflow_func: Callable
    ):
        """Register a custom workflow"""
        self.workflows[name] = workflow_func
        logger.info("workflow_registered", name=name)
    
    async def execute_custom_workflow(
        self,
        workflow_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute a custom workflow"""
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow {workflow_name} not found")
        
        workflow_func = self.workflows[workflow_name]
        return await workflow_func(self, **kwargs)