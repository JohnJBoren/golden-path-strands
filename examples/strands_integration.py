"""
Golden Path + Strands Agents SDK Integration Example

This example demonstrates how to:
1. Create Strands agents with Golden Path telemetry capture
2. Automatically log agent traces to JSONL for datamining
3. Discover golden paths from agent executions
4. Generate DPO training datasets from traces
"""

import asyncio
from pathlib import Path
from typing import Dict, Any

# Golden Path imports
from golden_path_strands.strands_telemetry import StrandsTelemetryIntegration
from golden_path_strands.jsonl_stream_writer import JSONLStreamWriter, StreamConfig
from golden_path_strands.process_miner import ProcessMiner
from golden_path_strands.evaluation import LLMJudgeEvaluator

# Strands SDK imports (will be available after pip install strands-agents)
try:
    from strands import Agent
    from strands.tools import tool
    from strands.telemetry import StrandsTelemetry
    STRANDS_AVAILABLE = True
except ImportError:
    STRANDS_AVAILABLE = False
    print("⚠️  Strands SDK not installed. Install with: pip install strands-agents")


# Example tools for the agent
@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for information.

    Args:
        query: Search query

    Returns:
        Search results
    """
    # Simulated search
    return f"Found 3 articles about '{query}'"


@tool
def calculate(expression: str) -> str:
    """Perform mathematical calculations.

    Args:
        expression: Math expression to evaluate

    Returns:
        Calculation result
    """
    import ast
    import operator

    # Safe evaluation using ast
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def eval_expr(node):
        if isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.Constant):  # Python >= 3.8
            return node.value
        elif isinstance(node, ast.BinOp):
            return ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
        elif isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](eval_expr(node.operand))
        else:
            raise ValueError(f"Unsupported expression type: {type(node)}")

    try:
        # Limit expression length to prevent DoS
        if len(expression) > 1000:
            return "Expression too long (max 1000 characters)"

        tree = ast.parse(expression, mode='eval')

        # Check AST depth to prevent deep recursion
        def check_depth(node, depth=0, max_depth=20):
            if depth > max_depth:
                raise ValueError("Expression too complex (max depth 20)")
            for child in ast.iter_child_nodes(node):
                check_depth(child, depth + 1, max_depth)

        check_depth(tree)
        result = eval_expr(tree.body)

        # Validate result is reasonable
        if isinstance(result, (int, float)):
            if abs(result) > 1e100:
                return "Result too large"

        return f"Result: {result}"
    except (ValueError, SyntaxError, KeyError, ZeroDivisionError, TypeError,
            AttributeError, RecursionError, MemoryError, OverflowError) as e:
        return f"Invalid expression: {str(e)}"


@tool
def generate_code(language: str, description: str) -> str:
    """Generate code in specified language.

    Args:
        language: Programming language
        description: What the code should do

    Returns:
        Generated code
    """
    # Simulated code generation
    return f"```{language}\n# {description}\nprint('Hello from {language}')\n```"


class GoldenPathStrandsAgent:
    """Strands agent with Golden Path telemetry integration."""

    def __init__(self, model: str = "gpt-4", output_dir: str = "datasets/strands_traces"):
        """Initialize the agent with telemetry.

        Args:
            model: Model to use (e.g., from Bedrock, OpenAI, etc.)
            output_dir: Directory for trace output
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Golden Path telemetry
        self.telemetry_integration = StrandsTelemetryIntegration(output_dir)

        # Initialize JSONL writer for real-time streaming
        self.stream_writer = JSONLStreamWriter(
            StreamConfig(
                output_dir=output_dir,
                max_file_size_mb=50,
                compression=True,
                buffer_size=10
            )
        )

        # Initialize evaluator
        self.evaluator = LLMJudgeEvaluator()

        if STRANDS_AVAILABLE:
            # Setup Strands telemetry with Golden Path processor
            self.telemetry_integration.setup_with_strands()

            # Create Strands agent
            self.agent = Agent(
                model=model,
                tools=[search_knowledge_base, calculate, generate_code],
                system_prompt="""You are a helpful AI assistant with access to tools.
                Use them to help users with their requests."""
            )
        else:
            self.agent = None

    async def run_with_telemetry(self, user_message: str) -> Dict[str, Any]:
        """Run the agent and capture telemetry.

        Args:
            user_message: User's request

        Returns:
            Agent result with telemetry data
        """
        if not self.agent:
            return {"error": "Strands SDK not available"}

        # Run the agent
        result = await self.agent.run(user_message)

        # Evaluate the response
        evaluation = await self.evaluator.evaluate_response(
            query=user_message,
            response=result.response,
            tools=["search_knowledge_base", "calculate", "generate_code"],
            metadata={
                "tokens": result.usage.total_tokens if hasattr(result, 'usage') else 0,
                "execution_time_ms": result.latency * 1000 if hasattr(result, 'latency') else 0
            }
        )

        # Create trace data for Golden Path
        trace_data = {
            "trace_id": result.trace_id if hasattr(result, 'trace_id') else "unknown",
            "session_id": self.telemetry_integration.processor.session_id,
            "agent_name": "golden_path_strands_agent",
            "user_message": user_message,
            "final_response": result.response,
            "decision_sequence": self._extract_decisions_from_result(result),
            "total_tokens": {
                "prompt": result.usage.prompt_tokens if hasattr(result, 'usage') else 0,
                "completion": result.usage.completion_tokens if hasattr(result, 'usage') else 0
            },
            "total_time_ms": result.latency * 1000 if hasattr(result, 'latency') else 0,
            "outcome": "successful" if result.response else "failed",
            "evaluation_score": self.evaluator.get_consensus_score(evaluation)
        }

        # Mark as golden path if high score
        if trace_data["evaluation_score"] >= 0.8:
            trace_data["golden_path"] = True

        # Write to JSONL stream
        self.stream_writer.write_agent_trace(trace_data)

        return {
            "response": result.response,
            "trace_data": trace_data,
            "evaluation": evaluation
        }

    def _extract_decisions_from_result(self, result) -> list:
        """Extract decision sequence from Strands result.

        Args:
            result: Strands agent result

        Returns:
            List of decision points
        """
        decisions = []

        # Extract from result.cycles if available
        if hasattr(result, 'cycles'):
            for i, cycle in enumerate(result.cycles):
                decision = {
                    "cycle_number": i,
                    "prompt": cycle.prompt if hasattr(cycle, 'prompt') else "",
                    "model_response": cycle.response if hasattr(cycle, 'response') else "",
                    "tool_calls": cycle.tool_calls if hasattr(cycle, 'tool_calls') else [],
                    "tokens_used": {
                        "prompt": cycle.prompt_tokens if hasattr(cycle, 'prompt_tokens') else 0,
                        "completion": cycle.completion_tokens if hasattr(cycle, 'completion_tokens') else 0
                    }
                }
                decisions.append(decision)

        return decisions


async def run_exploration_pipeline():
    """Run a complete exploration and datamining pipeline."""

    print("🚀 Starting Golden Path + Strands Integration Pipeline")
    print("=" * 60)

    # Initialize agent
    agent = GoldenPathStrandsAgent(
        model="gpt-4",  # Or use Bedrock: "us.anthropic.claude-3-haiku-20240307-v1:0"
        output_dir="datasets/strands_traces"
    )

    # Test queries for exploration
    test_queries = [
        "Search for information about quantum computing and calculate 2^10",
        "Generate Python code for a fibonacci function",
        "What is the square root of 144?",
        "Search for machine learning best practices and generate a simple neural network in Python",
        "Calculate the factorial of 5 and explain the result",
    ]

    print("\n📊 Running test queries...")
    results = []

    for query in test_queries:
        print(f"\n Query: {query[:50]}...")
        result = await agent.run_with_telemetry(query)
        results.append(result)

        if "error" not in result:
            print(f"  ✓ Response generated")
            print(f"  ✓ Evaluation score: {result['trace_data']['evaluation_score']:.2f}")
            if result['trace_data'].get('golden_path'):
                print(f"  🌟 Golden path identified!")

    # Close the stream writer to flush data
    agent.stream_writer.close()

    print("\n" + "=" * 60)
    print("📈 Analyzing traces with ProcessMiner...")

    # Initialize process miner
    miner = ProcessMiner(min_pattern_frequency=2)

    # Load traces
    miner.load_traces_from_directory("datasets/strands_traces")

    # Discover patterns
    patterns = miner.discover_patterns()
    print(f"\n✓ Discovered {len(patterns)} patterns")

    for i, pattern in enumerate(patterns[:3]):  # Show top 3
        print(f"  Pattern {i+1}: {' -> '.join(pattern.sequence)}")
        print(f"    Frequency: {pattern.frequency}, Avg Score: {pattern.avg_score:.2f}")

    # Identify bottlenecks
    bottlenecks = miner.identify_bottlenecks()
    if bottlenecks:
        print(f"\n✓ Identified {len(bottlenecks)} bottlenecks")
        for bottleneck in bottlenecks[:2]:  # Show top 2
            print(f"  {bottleneck.location}: {bottleneck.avg_delay_ms:.0f}ms delay")
            print(f"    Suggestion: {bottleneck.suggested_optimization}")

    # Generate DPO training data
    preference_pairs = miner.generate_preference_pairs()
    print(f"\n✓ Generated {len(preference_pairs)} preference pairs for DPO training")

    # Export results
    print("\n💾 Exporting results...")
    miner.export_golden_paths("datasets/golden_paths_analysis.json")
    miner.export_dpo_dataset("datasets/dpo_training_data.jsonl")

    # Print statistics
    stats = miner.get_statistics()
    print("\n📊 Pipeline Statistics:")
    print(f"  Total traces: {stats['total_traces']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")
    print(f"  Avg execution time: {stats['avg_execution_time_ms']:.0f}ms")
    print(f"  Avg evaluation score: {stats['avg_evaluation_score']:.2f}")

    print("\n✅ Pipeline complete! Check datasets/ directory for output files.")


def main():
    """Main entry point."""
    if not STRANDS_AVAILABLE:
        print("\n❌ Strands SDK is required for this example.")
        print("Install it with: pip install strands-agents strands-agents-tools")
        print("\nAlternatively, you can use the existing Ollama integration:")
        print("  python examples/research_agent.py")
        return

    # Run the async pipeline
    asyncio.run(run_exploration_pipeline())


if __name__ == "__main__":
    main()