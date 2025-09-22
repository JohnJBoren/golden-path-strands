# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Golden Path Strands is a framework for optimizing AI agent workflows through systematic exploration, evaluation, and distillation. It transforms expensive exploratory AI behaviors into efficient production systems by discovering "golden paths" - optimal solution patterns that can be distilled into specialized models.

**Key Goal**: Achieve 98% cost reduction while maintaining accuracy by systematically discovering optimal solution patterns and distilling them into specialized models.

## Development Commands

### Setup and Installation
```bash
# Automatic setup (recommended) - creates venv, installs deps, checks Ollama
./setup.sh

# Manual installation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"  # For development with test tools

# Create required directories
mkdir -p logs datasets config
```

### Running the Framework
```bash
# Interactive menu mode (recommended for exploration)
python start.py

# Direct execution modes
python start.py --mode research --topic "Your topic"
python start.py --mode coding --requirements "Your requirements"
python start.py --mode discovery  # Runs golden path discovery demo
python start.py --check-only  # Check Ollama status only
python start.py --debug  # Run with debug output for errors

# Run example agents
python examples/research_agent.py --topic "AI trends" --depth comprehensive
python examples/coding_agent.py --requirements "Binary search" --language python
python examples/strands_integration.py  # AWS Strands SDK integration
python examples/quick_start.py  # Quick start example
```

### Testing
```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=golden_path_strands --cov-report=html

# Run specific test
pytest tests/test_core.py::TestGoldenPathStrands::test_initialization

# Run tests with verbose output
pytest tests/ -v

# Run tests quietly (default in pyproject.toml)
pytest tests/ -q
```

### Code Quality
```bash
# Format code (modifies files)
black src/ tests/

# Check formatting without modifying
black src/ tests/ --check

# Linting
flake8 src/ tests/

# Type checking
mypy src/

# Sort imports
isort src/ tests/

# Run pre-commit hooks (runs black, isort, flake8, and tests)
pre-commit run --all-files
```

### Syntax Validation
```bash
# Quick syntax check for modified files
python3 -m py_compile src/golden_path_strands/*.py src/golden_path/*.py

# Run single test file
pytest tests/test_core.py -v

# Run tests matching pattern
pytest tests/ -k "test_initialization" -v
```

## High-Level Architecture

### Two Module System
The codebase has two parallel implementations:
1. **`src/golden_path/`** - Legacy module with complete exploration features, includes local file-based logging
2. **`src/golden_path_strands/`** - Main production module with AWS Strands SDK integration and OpenTelemetry

Both modules share similar structure but `golden_path_strands` is the primary focus for production use.

### Core Data Flow
```
User Request → AgentOrchestrator → OllamaProvider → LLM Response
                    ↓
            Decision Logging → JSONL Files → ProcessMiner
                    ↓
            Evaluation (LLMJudgeEvaluator) → Golden Paths
                    ↓
            DatasetCreator → DPO Training Pairs
```

### Key Architectural Decisions

1. **Async-First Design**: All agent operations use asyncio for concurrent execution
2. **Provider Abstraction**: `OllamaProvider` wraps Ollama API, easily extensible for other providers
3. **Streaming JSONL**: Real-time trace capture with automatic rotation and compression
4. **Hierarchical Config**: Settings loaded from environment > config.yaml > defaults
5. **Agent Composition**: Agents can be composed into sequential or parallel workflows
6. **Defensive Coding**: All dictionary access uses `.get()` with defaults, error handling continues workflows

### Critical Integration Points

#### Ollama Connection
- Default endpoint: `http://localhost:11434`
- Model: `gpt-oss:20b` (20GB download required)
- The framework auto-pulls the model if not available
- Connection pooling via aiohttp ClientSession
- Automatic retry on model not found

#### Configuration Loading
```python
# Settings must be converted to dict for core.py compatibility
from golden_path_strands.config import load_config
settings = load_config()
config_dict = settings.to_dict()  # Critical: use to_dict() method
```

#### Return Type Expectations
```python
# run_complete_pipeline returns a dict with these keys:
{
    "tasks_completed": int,
    "discovered_paths": int,
    "average_score": float,
    "dataset_path": str,
    "paths": List[Dict]
}
```

#### AWS Strands SDK Integration
When `strands-agents` package is available:
- OpenTelemetry spans automatically captured
- Traces converted to JSONL format
- StrandsTraceProcessor handles span lifecycle
- Golden paths identified by evaluation scores > 0.8

### Module Dependencies

**Core Dependencies Flow:**
- `core.py` → `agent_orchestrator.py` → `ollama_provider.py`
- `core.py` → `evaluation.py` → LLM judge ensemble
- `core.py` → `dataset.py` → JSONL output

**Telemetry Pipeline:**
- `strands_telemetry.py` → `jsonl_stream_writer.py` → compressed JSONL files
- `process_miner.py` → reads JSONL → generates DPO pairs

### Error Handling Patterns

The codebase uses structured logging with `structlog`:
```python
import structlog
logger = structlog.get_logger()
logger.info("event_name", key1=value1, key2=value2)
logger.warning("skipped_item", reason="missing_field")
logger.error("operation_failed", error=str(e))
```

File operations use atomic writes with temp files:
```python
# Pattern used throughout for safe file writes
temp_path = target_path.with_suffix('.tmp')
with open(temp_path, 'w') as f:
    # write data
temp_path.replace(target_path)  # Atomic move
```

Async operations use `gather` with exception handling:
```python
results = await asyncio.gather(*tasks, return_exceptions=True)
# Filter out exceptions from results
valid_results = [r for r in results if not isinstance(r, Exception)]
```

### State Management

1. **Agent History**: Each Agent instance maintains execution history
2. **Session Tracking**: Exploration sessions tracked via session_id
3. **Trace Aggregation**: DecisionPoints accumulated until trace completion
4. **File Rotation**: JSONL files rotate by size (100MB) and time (24h)
5. **Error Continuity**: Workflows continue despite individual step failures

## Common Development Tasks

### Adding a New Agent Type
1. Add enum value to `AgentType` in `agent_orchestrator.py`
2. Add default system prompt in `Agent._default_system_prompt()`
3. Register in `AgentOrchestrator._initialize_default_agents()`
4. Create workflow method in `AgentOrchestrator` if needed

### Modifying Evaluation Criteria
Edit evaluation rubric in `core.py` or pass custom rubric:
```python
evaluator = LLMJudgeEvaluator()
evaluation = await evaluator.evaluate_response(
    task, response, custom_rubric={...}
)
```

### Safe Mathematical Expression Evaluation
The `calculate` tool in `examples/strands_integration.py` uses AST-based safe evaluation:
```python
# Never use eval() - use AST parsing instead
import ast
tree = ast.parse(expression, mode='eval')
# Process tree safely
```

### Debugging Ollama Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# List available models
ollama list

# Test model directly
ollama run gpt-oss:20b "Test prompt"

# View Ollama logs
# On macOS: ~/Library/Logs/ollama/
# On Linux: ~/.ollama/logs/
```

### Process Mining Workflow
```python
from golden_path_strands.process_miner import ProcessMiner

# Load traces from JSONL files
miner = ProcessMiner()
miner.load_traces_from_directory("datasets/strands_traces")

# Discover patterns
patterns = miner.discover_patterns()  # Find common successful sequences
bottlenecks = miner.identify_bottlenecks()  # Find performance issues

# Generate DPO training data
pairs = miner.generate_preference_pairs(min_score_difference=0.15)
miner.export_dpo_dataset("datasets/dpo_training.jsonl", pairs)
```

## Important Notes

- **Python 3.10+ Required**: Code uses union types like `Dict | None` (Python 3.10+ syntax)
- **Ollama Must Be Running**: Framework expects Ollama server at localhost:11434
- **Large Model Download**: GPT-OSS:20b is 20GB, ensure sufficient disk space
- **Directory Creation**: Some modules expect `datasets/` and `logs/` directories to exist (created by setup.sh)
- **Async Context**: Most operations are async, use `asyncio.run()` or await properly
- **Resource Cleanup**: File handles and aiohttp sessions properly managed via context managers
- **Config Compatibility**: Always use `Settings.to_dict()` when passing config to core.py
- **Dictionary Access**: Always use `.get()` method with defaults for dictionary access
- **Error Handling**: Workflows continue on error, check result dictionaries for `error` keys
- **Dependencies**: Main production dependencies in requirements.txt, dev dependencies in pyproject.toml[project.optional-dependencies]
- **CI/CD**: GitHub Actions runs pre-commit hooks on push/PR (see .github/workflows/ci.yml)

## Three-Phase Methodology Implementation

1. **Exploration Phase** (`core.py:discover_golden_paths`)
   - Deploys agents to explore task space
   - Logs all decisions to JSONL format
   - Uses AgentOrchestrator for multi-agent workflows
   - Handles errors gracefully, continuing exploration

2. **Evaluation Phase** (`evaluation.py:LLMJudgeEvaluator`)
   - Multi-criteria scoring with configurable weights
   - Consensus scoring to reduce bias
   - Marks paths with scores > 0.8 as "golden"
   - Validates responses against task requirements

3. **Distillation Phase** (`process_miner.py`)
   - Mines patterns from successful traces
   - Generates DPO preference pairs
   - Exports training datasets for fine-tuning
   - Logs skipped/invalid traces for debugging

## Agent Types Available

The framework includes several specialized agents in `agent_orchestrator.py`:
- **Research Agent**: Information gathering and analysis
- **Coding Agent**: Code generation with tests
- **Analysis Agent**: Data analysis and insights
- **Planning Agent**: Task breakdown and planning
- **Testing Agent**: Test generation and QA

Each agent can be used individually or composed into workflows using `execute_sequential()` or `execute_parallel()` methods.

## Critical Patterns and Conventions

### Code Safety Patterns
- **Safe expression evaluation**: Never use `eval()`, always use AST parsing (see examples/strands_integration.py:calculate tool)
- **Atomic file operations**: Write to temp files first, then atomic rename
- **Async error handling**: Use `asyncio.gather(*tasks, return_exceptions=True)` and filter exceptions
- **Dictionary access**: Always use `.get()` with defaults to prevent KeyError
- **Type annotations**: Use `Optional[T]` instead of `T = None` for consistency

### Logging Conventions
- Use `structlog` for all logging with structured key-value pairs
- Log levels: `info` for normal operations, `warning` for skipped items, `error` for failures
- Always include context: `logger.info("event_name", task_id=id, session_id=session)`

### Testing Patterns
- Tests use `pytest-asyncio` for async test functions
- Mock Ollama responses in tests to avoid external dependencies
- Use fixtures for common test setup (see tests/conftest.py if exists)