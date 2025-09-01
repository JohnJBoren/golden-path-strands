#!/usr/bin/env python3
"""Coding Agent Example - Demonstrates code generation capabilities"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from golden_path_strands import AgentOrchestrator
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


async def generate_code(requirements: str, language: str = "python", include_tests: bool = True):
    """Generate code based on requirements"""
    
    console.print(Panel(
        f"[bold blue]Coding Agent[/bold blue]\n"
        f"Requirements: {requirements}\n"
        f"Language: {language}\n"
        f"Include Tests: {include_tests}",
        expand=False
    ))
    
    # Initialize orchestrator
    orchestrator = AgentOrchestrator(
        ollama_host="http://localhost:11434",
        model="gpt-oss:20b"
    )
    
    console.print("[yellow]Starting code generation workflow...[/yellow]")
    
    try:
        result = await orchestrator.execute_coding_workflow(
            requirements=requirements,
            language=language,
            include_tests=include_tests
        )
        
        # Display planning phase
        console.print("\n[bold green]Implementation Plan:[/bold green]")
        if result["results"]:
            plan = result["results"][0]["response"]
            console.print(Panel(plan[:500] + "..." if len(plan) > 500 else plan))
        
        # Display generated code
        console.print("\n[bold green]Generated Code:[/bold green]")
        if result["code"]:
            # Extract code blocks
            code = extract_code_blocks(result["code"], language)
            if code:
                syntax = Syntax(code, language, theme="monokai", line_numbers=True)
                console.print(Panel(syntax))
        
        # Display tests if included
        if include_tests and result["tests"]:
            console.print("\n[bold green]Test Cases:[/bold green]")
            test_code = extract_code_blocks(result["tests"], language)
            if test_code:
                syntax = Syntax(test_code, language, theme="monokai", line_numbers=True)
                console.print(Panel(syntax))
        
        return result
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return None


def extract_code_blocks(text: str, language: str) -> str:
    """Extract code blocks from markdown-formatted text"""
    import re
    
    # Look for code blocks
    pattern = rf'```{language}?\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        return '\n\n'.join(matches)
    
    # If no code blocks found, look for indented code
    lines = text.split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        if line.startswith('    ') or line.startswith('\t'):
            code_lines.append(line[4:] if line.startswith('    ') else line[1:])
            in_code = True
        elif in_code and line.strip() == '':
            code_lines.append('')
        else:
            in_code = False
    
    return '\n'.join(code_lines) if code_lines else text


async def code_review_workflow():
    """Demonstrate code review workflow"""
    
    console.print("[bold]Code Review Workflow[/bold]\n")
    
    # Sample code for review
    sample_code = """
def calculate_fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

def find_prime_numbers(limit):
    primes = []
    for num in range(2, limit + 1):
        is_prime = True
        for i in range(2, num):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    return primes
"""
    
    orchestrator = AgentOrchestrator()
    
    # Review workflow
    workflow = [
        {
            "agent": "coder",
            "task": f"Review this Python code for correctness, efficiency, and best practices:\n```python\n{sample_code}\n```",
            "context": {"review_type": "comprehensive"}
        },
        {
            "agent": "tester",
            "task": "Identify edge cases and potential issues in the reviewed code",
        },
        {
            "agent": "coder",
            "task": "Provide optimized version of the code addressing the identified issues",
        }
    ]
    
    results = await orchestrator.execute_sequential(workflow)
    
    # Display results
    for i, result in enumerate(results, 1):
        agent = result["agent"]
        response = result["response"]
        
        console.print(f"\n[cyan]Step {i} - {agent}:[/cyan]")
        
        if "```" in response:
            # Extract and display code
            code = extract_code_blocks(response, "python")
            if code and i == 3:  # Optimized version
                syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
                console.print(Panel(syntax))
            else:
                console.print(Panel(response[:800] + "..." if len(response) > 800 else response))
        else:
            console.print(Panel(response[:600] + "..." if len(response) > 600 else response))


async def build_project():
    """Build a complete project structure"""
    
    console.print("[bold]Project Builder[/bold]\n")
    
    project_requirements = """
    Create a REST API service for a todo list application with the following features:
    - CRUD operations for todos
    - User authentication
    - Data persistence
    - Input validation
    - Error handling
    - API documentation
    """
    
    orchestrator = AgentOrchestrator()
    
    # Project building workflow
    workflow = [
        {
            "agent": "planner",
            "task": f"Design the architecture and file structure for: {project_requirements}",
            "context": {"project_type": "REST API", "framework": "FastAPI or Flask"}
        },
        {
            "agent": "coder",
            "task": "Create the main application file with API endpoints",
            "context": {"language": "python"}
        },
        {
            "agent": "coder",
            "task": "Create the data models and database schema",
            "context": {"database": "SQLite or PostgreSQL"}
        },
        {
            "agent": "coder",
            "task": "Implement authentication and authorization",
        },
        {
            "agent": "tester",
            "task": "Create API tests and usage examples",
        },
        {
            "agent": "coder",
            "task": "Create Docker configuration and deployment scripts",
        }
    ]
    
    console.print("[yellow]Building project components...[/yellow]\n")
    
    results = await orchestrator.execute_sequential(workflow)
    
    # Display project structure
    files = {}
    for i, result in enumerate(results, 1):
        agent = result["agent"]
        response = result["response"]
        
        console.print(f"\n[cyan]Component {i} - {agent}:[/cyan]")
        
        # Extract code and save to files dict
        if "```" in response:
            code = extract_code_blocks(response, "python")
            if code:
                # Determine file name based on step
                if i == 2:
                    files["main.py"] = code
                elif i == 3:
                    files["models.py"] = code
                elif i == 4:
                    files["auth.py"] = code
                elif i == 5:
                    files["test_api.py"] = code
                elif i == 6:
                    # Extract Dockerfile
                    docker_code = extract_code_blocks(response, "dockerfile")
                    if docker_code:
                        files["Dockerfile"] = docker_code
                
                # Display code snippet
                syntax = Syntax(code[:500] + "..." if len(code) > 500 else code, 
                              "python", theme="monokai", line_numbers=True)
                console.print(Panel(syntax))
        else:
            console.print(Panel(response[:400] + "..." if len(response) > 400 else response))
    
    # Summary
    console.print("\n[bold green]Project Structure Created:[/bold green]")
    for filename in files.keys():
        console.print(f"  📄 {filename}")


async def interactive_coding():
    """Interactive coding assistant"""
    
    console.print("[bold]Interactive Coding Assistant[/bold]")
    console.print("Describe what you want to build (type 'quit' to exit)\n")
    
    orchestrator = AgentOrchestrator()
    
    while True:
        requirements = console.input("[yellow]What do you want to build? [/yellow]")
        
        if requirements.lower() in ['quit', 'exit', 'q']:
            break
        
        language = console.input("[yellow]Programming language (default: python): [/yellow]") or "python"
        include_tests = console.input("[yellow]Include tests? (y/n, default: y): [/yellow]").lower() != 'n'
        
        result = await generate_code(requirements, language, include_tests)
        
        if result:
            # Ask for modifications
            modify = console.input("\n[yellow]Any modifications needed? (press Enter to skip): [/yellow]")
            
            if modify:
                context = {
                    "previous_results": [{
                        "agent": "coder",
                        "response": result["code"]
                    }]
                }
                
                modified = await orchestrator.execute_single(
                    "coder",
                    f"Modify the code with these requirements: {modify}",
                    context
                )
                
                console.print("\n[bold green]Modified Code:[/bold green]")
                code = extract_code_blocks(modified["response"], language)
                if code:
                    syntax = Syntax(code, language, theme="monokai", line_numbers=True)
                    console.print(Panel(syntax))


async def main():
    """Main entry point"""
    console.print("[bold magenta]Golden Path Coding Agent Examples[/bold magenta]\n")
    
    import argparse
    parser = argparse.ArgumentParser(description="Coding Agent Examples")
    parser.add_argument("--requirements", type=str, help="Code requirements")
    parser.add_argument("--language", type=str, default="python", help="Programming language")
    parser.add_argument("--no-tests", action="store_true", help="Skip test generation")
    parser.add_argument("--mode", type=str, default="generate",
                       choices=["generate", "review", "project", "interactive"],
                       help="Coding mode")
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        await interactive_coding()
    elif args.mode == "review":
        await code_review_workflow()
    elif args.mode == "project":
        await build_project()
    else:
        requirements = args.requirements or "Create a function to merge two sorted arrays"
        await generate_code(requirements, args.language, not args.no_tests)


if __name__ == "__main__":
    asyncio.run(main())