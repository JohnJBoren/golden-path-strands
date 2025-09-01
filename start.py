#!/usr/bin/env python3
"""Main startup script for Golden Path Strands Framework"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from golden_path_strands import GoldenPathStrands, AgentOrchestrator
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
import click

console = Console()


def display_banner():
    """Display startup banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║           🌟 Golden Path Strands Framework 🌟               ║
    ║                                                              ║
    ║     AI Agent Orchestration with Ollama & GPT-OSS:20b        ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner, style="bold cyan"))


async def check_ollama_status():
    """Check if Ollama is running and model is available"""
    from golden_path_strands.ollama_provider import OllamaProvider
    
    console.print("[yellow]Checking Ollama status...[/yellow]")
    
    provider = OllamaProvider()
    
    try:
        # Check if Ollama is accessible
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{provider.host}/api/tags") as response:
                if response.status != 200:
                    return False, "Ollama server not accessible"
        
        # Check if model is available
        if await provider.check_model_availability():
            console.print("[green]✓ Ollama is running and gpt-oss:20b model is available[/green]")
            return True, "Ready"
        else:
            console.print("[yellow]⚠ Model gpt-oss:20b not found. Would you like to pull it?[/yellow]")
            if Confirm.ask("Pull gpt-oss:20b model? (This may take several minutes)"):
                console.print("[yellow]Pulling model... This will take a while (20GB model)[/yellow]")
                if await provider.pull_model():
                    console.print("[green]✓ Model pulled successfully[/green]")
                    return True, "Ready"
                else:
                    return False, "Failed to pull model"
            return False, "Model not available"
            
    except Exception as e:
        return False, f"Ollama not running: {str(e)}"


async def run_interactive_menu():
    """Run interactive menu"""
    while True:
        console.print("\n[bold cyan]Main Menu:[/bold cyan]")
        
        table = Table(show_header=False, box=None)
        table.add_column("Option", style="yellow")
        table.add_column("Description")
        
        table.add_row("1", "Research Agent - Conduct research on topics")
        table.add_row("2", "Coding Agent - Generate and review code")
        table.add_row("3", "Analysis Agent - Analyze data and patterns")
        table.add_row("4", "Custom Workflow - Create custom agent pipeline")
        table.add_row("5", "Golden Path Discovery - Find optimal patterns")
        table.add_row("6", "Settings - Configure agents and models")
        table.add_row("0", "Exit")
        
        console.print(table)
        
        choice = Prompt.ask("\n[cyan]Select option[/cyan]", choices=["0", "1", "2", "3", "4", "5", "6"])
        
        if choice == "0":
            console.print("[yellow]Goodbye! 👋[/yellow]")
            break
        elif choice == "1":
            await run_research_agent()
        elif choice == "2":
            await run_coding_agent()
        elif choice == "3":
            await run_analysis_agent()
        elif choice == "4":
            await run_custom_workflow()
        elif choice == "5":
            await run_golden_path_discovery()
        elif choice == "6":
            await show_settings()


async def run_research_agent():
    """Run research agent"""
    console.print("\n[bold]Research Agent[/bold]")
    
    topic = Prompt.ask("Enter research topic")
    depth = Prompt.ask("Research depth", choices=["quick", "moderate", "comprehensive"], default="moderate")
    
    orchestrator = AgentOrchestrator()
    
    with console.status("[yellow]Researching...[/yellow]"):
        result = await orchestrator.execute_research_workflow(topic, depth)
    
    console.print("\n[bold green]Research Complete![/bold green]")
    console.print(Panel(result["summary"][:1000] + "..." if len(result["summary"]) > 1000 else result["summary"]))
    
    if Confirm.ask("Save full results to file?"):
        filename = f"research_{topic[:30].replace(' ', '_')}.txt"
        with open(filename, 'w') as f:
            f.write(f"Topic: {topic}\n")
            f.write(f"Depth: {depth}\n\n")
            f.write("=" * 50 + "\n\n")
            f.write(result["summary"])
        console.print(f"[green]Results saved to {filename}[/green]")


async def run_coding_agent():
    """Run coding agent"""
    console.print("\n[bold]Coding Agent[/bold]")
    
    requirements = Prompt.ask("What do you want to build?")
    language = Prompt.ask("Programming language", default="python")
    include_tests = Confirm.ask("Include test cases?", default=True)
    
    orchestrator = AgentOrchestrator()
    
    with console.status("[yellow]Generating code...[/yellow]"):
        result = await orchestrator.execute_coding_workflow(requirements, language, include_tests)
    
    console.print("\n[bold green]Code Generated![/bold green]")
    
    # Display code
    if result["code"]:
        from rich.syntax import Syntax
        code = result["code"]
        # Extract actual code from markdown if present
        if "```" in code:
            import re
            matches = re.findall(rf'```{language}?\n(.*?)```', code, re.DOTALL)
            if matches:
                code = matches[0]
        
        syntax = Syntax(code[:1000] + "..." if len(code) > 1000 else code, 
                       language, theme="monokai", line_numbers=True)
        console.print(Panel(syntax))
    
    if Confirm.ask("Save code to file?"):
        ext_map = {"python": "py", "javascript": "js", "java": "java", "cpp": "cpp", "c": "c"}
        ext = ext_map.get(language.lower(), "txt")
        filename = f"generated_code.{ext}"
        
        with open(filename, 'w') as f:
            f.write(result["code"])
        
        if include_tests and result["tests"]:
            test_filename = f"test_{filename}"
            with open(test_filename, 'w') as f:
                f.write(result["tests"])
            console.print(f"[green]Code saved to {filename} and {test_filename}[/green]")
        else:
            console.print(f"[green]Code saved to {filename}[/green]")


async def run_analysis_agent():
    """Run analysis agent"""
    console.print("\n[bold]Analysis Agent[/bold]")
    
    data_desc = Prompt.ask("Describe the data or problem to analyze")
    analysis_type = Prompt.ask("Analysis type", 
                               choices=["exploratory", "statistical", "predictive"], 
                               default="exploratory")
    
    orchestrator = AgentOrchestrator()
    
    with console.status("[yellow]Analyzing...[/yellow]"):
        result = await orchestrator.execute_analysis_workflow(data_desc, analysis_type)
    
    console.print("\n[bold green]Analysis Complete![/bold green]")
    console.print(Panel(result["insights"][:1000] + "..." if len(result["insights"]) > 1000 else result["insights"]))


async def run_custom_workflow():
    """Run custom workflow"""
    console.print("\n[bold]Custom Workflow Builder[/bold]")
    
    orchestrator = AgentOrchestrator()
    workflow = []
    
    console.print("Build your workflow step by step (enter 'done' when finished)")
    
    step = 1
    while True:
        console.print(f"\n[cyan]Step {step}:[/cyan]")
        
        agent_type = Prompt.ask(
            "Select agent", 
            choices=["researcher", "coder", "analyst", "planner", "tester", "done"],
            default="done"
        )
        
        if agent_type == "done":
            break
        
        task = Prompt.ask(f"Task for {agent_type}")
        
        workflow.append({
            "agent": agent_type,
            "task": task,
            "context": {}
        })
        
        step += 1
    
    if workflow:
        console.print("\n[yellow]Executing custom workflow...[/yellow]")
        
        with console.status("[yellow]Processing...[/yellow]"):
            results = await orchestrator.execute_sequential(workflow)
        
        console.print("\n[bold green]Workflow Complete![/bold green]")
        
        for i, result in enumerate(results, 1):
            console.print(f"\n[cyan]Step {i} - {result['agent']}:[/cyan]")
            response = result["response"]
            console.print(Panel(response[:500] + "..." if len(response) > 500 else response))


async def run_golden_path_discovery():
    """Run golden path discovery"""
    console.print("\n[bold]Golden Path Discovery[/bold]")
    
    console.print("Enter tasks to discover optimal paths (one per line, empty line to finish):")
    
    tasks = []
    while True:
        task = Prompt.ask("Task", default="")
        if not task:
            break
        tasks.append(task)
    
    if not tasks:
        console.print("[yellow]No tasks provided[/yellow]")
        return
    
    iterations = int(Prompt.ask("Iterations per task", default="3"))
    
    gps = GoldenPathStrands()
    
    with console.status("[yellow]Discovering golden paths...[/yellow]"):
        results = await gps.run_complete_pipeline(tasks)
    
    console.print("\n[bold green]Discovery Complete![/bold green]")
    
    table = Table(title="Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Tasks Completed", str(results["tasks_completed"]))
    table.add_row("Paths Discovered", str(results["discovered_paths"]))
    table.add_row("Average Score", f"{results['average_score']:.2f}")
    
    if "dataset_path" in results:
        table.add_row("Dataset Created", results["dataset_path"])
    
    console.print(table)


async def show_settings():
    """Show and modify settings"""
    console.print("\n[bold]Settings[/bold]")
    
    table = Table(title="Current Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="yellow")
    
    table.add_row("Ollama Host", "http://localhost:11434")
    table.add_row("Model", "gpt-oss:20b")
    table.add_row("Default Temperature", "0.7")
    table.add_row("Max Tokens", "4096")
    
    console.print(table)
    
    if Confirm.ask("\nCheck Ollama status?"):
        status, message = await check_ollama_status()
        if status:
            console.print(f"[green]✓ {message}[/green]")
        else:
            console.print(f"[red]✗ {message}[/red]")


@click.command()
@click.option('--mode', type=click.Choice(['interactive', 'research', 'coding', 'discovery']), 
              default='interactive', help='Startup mode')
@click.option('--topic', help='Research topic (for research mode)')
@click.option('--requirements', help='Code requirements (for coding mode)')
@click.option('--check-only', is_flag=True, help='Only check Ollama status and exit')
def main(mode, topic, requirements, check_only):
    """Golden Path Strands Framework - AI Agent Orchestration"""
    
    display_banner()
    
    async def run():
        # Check Ollama status
        status, message = await check_ollama_status()
        
        if check_only:
            if status:
                console.print("[green]✓ System ready[/green]")
                sys.exit(0)
            else:
                console.print(f"[red]✗ {message}[/red]")
                sys.exit(1)
        
        if not status:
            console.print(f"[red]Error: {message}[/red]")
            console.print("\n[yellow]Please ensure Ollama is running:[/yellow]")
            console.print("1. Install Ollama: https://ollama.com/download")
            console.print("2. Start Ollama service")
            console.print("3. Pull the model: ollama pull gpt-oss:20b")
            sys.exit(1)
        
        # Run selected mode
        if mode == 'interactive':
            await run_interactive_menu()
        elif mode == 'research':
            if not topic:
                topic = Prompt.ask("Enter research topic")
            orchestrator = AgentOrchestrator()
            result = await orchestrator.execute_research_workflow(topic)
            console.print(Panel(result["summary"]))
        elif mode == 'coding':
            if not requirements:
                requirements = Prompt.ask("Enter code requirements")
            orchestrator = AgentOrchestrator()
            result = await orchestrator.execute_coding_workflow(requirements)
            console.print(Panel(result["code"]))
        elif mode == 'discovery':
            console.print("Running golden path discovery demo...")
            gps = GoldenPathStrands()
            demo_tasks = [
                "Create a Python function to sort a list",
                "Explain machine learning concepts",
                "Design a REST API"
            ]
            results = await gps.discover_golden_paths(demo_tasks, iterations=2)
            console.print(f"Discovered {results['discovered_paths']} golden paths")
    
    asyncio.run(run())


if __name__ == "__main__":
    main()