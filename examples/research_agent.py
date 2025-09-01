#!/usr/bin/env python3
"""Research Agent Example - Demonstrates research capabilities"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from golden_path_strands import AgentOrchestrator
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


async def research_topic(topic: str, depth: str = "comprehensive"):
    """Research a topic using the agent orchestrator"""
    
    console.print(Panel(f"[bold blue]Research Agent[/bold blue]\nTopic: {topic}\nDepth: {depth}", expand=False))
    
    # Initialize orchestrator
    orchestrator = AgentOrchestrator(
        ollama_host="http://localhost:11434",
        model="gpt-oss:20b"
    )
    
    # Execute research workflow
    console.print("[yellow]Starting research workflow...[/yellow]")
    
    try:
        result = await orchestrator.execute_research_workflow(
            topic=topic,
            depth=depth
        )
        
        # Display results
        console.print("\n[bold green]Research Results:[/bold green]")
        
        for i, step_result in enumerate(result["results"], 1):
            agent = step_result["agent"]
            response = step_result["response"]
            
            console.print(f"\n[cyan]Step {i} - Agent: {agent}[/cyan]")
            console.print(Panel(response[:500] + "..." if len(response) > 500 else response))
        
        # Display final summary
        console.print("\n[bold green]Final Summary:[/bold green]")
        console.print(Panel(Markdown(result["summary"])))
        
        return result
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return None


async def custom_research_pipeline():
    """Run a custom research pipeline with multiple topics"""
    
    topics = [
        "Latest advances in quantum computing and their practical applications",
        "Impact of large language models on software development workflows",
        "Emerging trends in distributed systems and microservices architecture",
    ]
    
    orchestrator = AgentOrchestrator()
    
    console.print("[bold]Custom Research Pipeline[/bold]")
    console.print("Researching multiple topics in parallel...\n")
    
    # Research topics in parallel
    tasks = []
    for topic in topics:
        task_spec = {
            "agent": "researcher",
            "task": f"Research and summarize: {topic}",
            "context": {"depth": "moderate", "max_length": 500}
        }
        tasks.append(task_spec)
    
    results = await orchestrator.execute_parallel(tasks)
    
    # Analyze all results together
    combined_context = {
        "previous_results": [
            {"agent": "researcher", "response": r["response"]} 
            for r in results
        ]
    }
    
    synthesis = await orchestrator.execute_single(
        "analyst",
        "Synthesize the research findings and identify common themes and connections",
        combined_context
    )
    
    # Display results
    for i, (topic, result) in enumerate(zip(topics, results), 1):
        console.print(f"\n[cyan]Topic {i}: {topic[:50]}...[/cyan]")
        console.print(Panel(result["response"][:300] + "..."))
    
    console.print("\n[bold green]Synthesis:[/bold green]")
    console.print(Panel(Markdown(synthesis["response"])))


async def interactive_research():
    """Interactive research mode"""
    console.print("[bold]Interactive Research Agent[/bold]")
    console.print("Enter topics to research (type 'quit' to exit)\n")
    
    orchestrator = AgentOrchestrator()
    
    while True:
        topic = console.input("[yellow]Enter research topic: [/yellow]")
        
        if topic.lower() in ['quit', 'exit', 'q']:
            break
        
        depth = console.input("[yellow]Depth (quick/moderate/comprehensive): [/yellow]") or "moderate"
        
        result = await research_topic(topic, depth)
        
        if result:
            # Ask for follow-up
            follow_up = console.input("\n[yellow]Any follow-up questions? (press Enter to skip): [/yellow]")
            
            if follow_up:
                context = {
                    "previous_results": [{
                        "agent": "researcher",
                        "response": result["summary"]
                    }]
                }
                
                follow_up_result = await orchestrator.execute_single(
                    "researcher",
                    follow_up,
                    context
                )
                
                console.print("\n[bold green]Follow-up Answer:[/bold green]")
                console.print(Panel(Markdown(follow_up_result["response"])))


async def main():
    """Main entry point"""
    console.print("[bold magenta]Golden Path Research Agent Examples[/bold magenta]\n")
    
    import argparse
    parser = argparse.ArgumentParser(description="Research Agent Examples")
    parser.add_argument("--topic", type=str, help="Research topic")
    parser.add_argument("--depth", type=str, default="moderate", 
                       choices=["quick", "moderate", "comprehensive"],
                       help="Research depth")
    parser.add_argument("--mode", type=str, default="single",
                       choices=["single", "pipeline", "interactive"],
                       help="Research mode")
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        await interactive_research()
    elif args.mode == "pipeline":
        await custom_research_pipeline()
    else:
        topic = args.topic or "The future of artificial intelligence in healthcare"
        await research_topic(topic, args.depth)


if __name__ == "__main__":
    asyncio.run(main())