# Claude Code + Agentic Network v2 Integration

This integrates the open-source LLMCode CLI (a Claude Code alternative) with the Agentic Network v2 framework, providing an intelligent coding assistant that uses sparse multi-agent orchestration.

## Features

- **Interactive CLI**: File operations, directory navigation, workspace context
- **Agentic Intelligence**: Uses the Agentic Network v2 orchestrator for intelligent responses
- **Central Model Configuration**: Single configuration for all AI interactions
- **Workspace Awareness**: Understands your codebase structure and files
- **Multi-Agent Collaboration**: Leverages multiple specialized agents for complex tasks

## Installation

### Quick Start
```bash
# Install dependencies
python setup_claude_cli.py

# Configure API key
python claude_integrated.py
/config set apiKey YOUR_API_KEY

# Exit and restart
/exit
python claude_integrated.py
```

### Manual Installation
```bash
# Install requirements
pip install -r requirements.txt
pip install openai>=1.57.4

# Run the CLI
python claude_integrated.py
```

## Usage

### Basic Commands
```
/help                    Show help message
/exit, /quit            Exit the program
/pwd, /cwd              Print working directory
```

### File Operations
```
/ls [path]              List directory contents
/tree [path]            Show directory structure
/cat <file>             Display file contents
/write <file>           Create/overwrite a file
/append <file>          Append to existing file
```

### Directory Operations
```
/cd <path>              Change directory
/mkdir <path>           Create directory
```

### Configuration
```
/config set <key> <value>  Set configuration value
/config list               List all configurations
/config show               Show active configuration
```

### Context
```
/context [path]          Get workspace context from path
/# [path]                Alias for /context
```

### Agentic Network
```
/agents                  Show agent information
/reload                  Reload agentic network configuration
```

## Configuration

The configuration file is stored at `~/.claude_agentic_config.json`. Configure:

- `api_key`: Your API key (OpenAI or Anthropic)
- `llm_provider`: "openai" or "anthropic"
- `llm_model`: Model name (e.g., "gpt-4o", "claude-3-5-sonnet")
- `llm_base_url`: Base URL for API (default: OpenAI)
- `budget_mode`: "low", "balanced", or "thorough"
- `enable_parallel`: Enable parallel execution
- `max_parallel_agents`: Maximum concurrent agents
- `debug`: Enable debug mode

### Example Configuration
```bash
/config set apiKey sk-...
/config set llm_provider openai
/config set llm_model gpt-4o
/config set budget_mode balanced
/config set enable_parallel true
```

## How It Works

1. **CLI Interface**: Provides interactive file operations and workspace context
2. **Context Gathering**: When you use `/context` or `/#`, it reads your workspace files
3. **Agentic Processing**: Your requests are sent to the Agentic Network orchestrator
4. **Multi-Agent Collaboration**: The orchestrator routes tasks to specialized agents
5. **Intelligent Responses**: Combines multiple agent perspectives for better answers

## Integration Details

### File Structure
```
claude_integrated.py          # Main integrated CLI
setup_claude_cli.py           # Setup script
CLAUDE_CLI_README.md          # This file
llmcode-cli/                  # Original LLMCode CLI source
claude-cli/                   # Original Claude CLI docs/plugins
```

### Key Components
1. **Color Terminal**: ANSI-colored output for better UX
2. **File Edit Mode**: Interactive file creation/editing
3. **Workspace Context**: Reads and caches workspace files
4. **Agentic Network Client**: Bridges CLI to orchestrator
5. **Configuration Management**: Persistent config storage

## Agentic Network Features

The integrated system leverages:
- **Sparse Router**: Activates only valuable agents per task
- **Parallel Executor**: Thread-safe concurrent execution
- **Skill Packs**: Soft specialization without spawning new agents
- **Lifecycle Management**: Creates and prunes specialists based on demand
- **Conflict Arbitration**: Resolves disagreements between agents

## Example Workflow

```bash
# Start the CLI
python claude_integrated.py

# Get workspace context
/# 

# Ask for help with a file
Can you explain the main.py file?

# Create a new file
/write new_script.py
# ... enter content ...
/save

# Get directory listing
/ls

# Ask for code review
Can you review my new_script.py for errors?
```

## Advanced Usage

### Using with Local Models
```bash
/config set llm_provider ollama
/config set llm_model qwen2.5:7b
/config set llm_base_url http://localhost:11434
```

### Debug Mode
```bash
/config set debug true
```

### Custom Budget Modes
- **low**: Single agent only (fastest)
- **balanced**: Sparse multi-agent (recommended)
- **thorough**: More agents allowed (most thorough)

## Troubleshooting

### API Key Issues
```
Error: API key not configured
```
Solution: `/config set apiKey YOUR_API_KEY`

### Import Errors
```
ModuleNotFoundError: No module named 'app'
```
Solution: Run from the project root directory

### Configuration Issues
```
Error loading config
```
Solution: Delete `~/.claude_agentic_config.json` and reconfigure

## Development

### Adding New Commands
Edit `claude_integrated.py` and add to:
1. Command parsing in main loop
2. `print_help()` function
3. Implementation function

### Extending Agentic Integration
Modify `AgenticNetworkClient` class to:
1. Expose more orchestrator features
2. Add new agent management commands
3. Enhance context handling

## License

[Your License Here]

## Acknowledgments

- **LLMCode**: Open-source Claude Code alternative
- **Agentic Network v2**: Sparse multi-agent orchestration framework
- **Anthropic**: Original Claude Code inspiration