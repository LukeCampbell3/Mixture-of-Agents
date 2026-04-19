"""Main entry point for the agentic network."""

import argparse
from app.orchestrator import Orchestrator


def main():
    """Run the agentic network CLI."""
    parser = argparse.ArgumentParser(description="Agentic Network v2")
    parser.add_argument("request", type=str, help="Task request")
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic"],
        help="LLM provider"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model name"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for local LLM API"
    )
    parser.add_argument(
        "--budget",
        type=str,
        default="balanced",
        choices=["low", "balanced", "thorough"],
        help="Budget mode"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = Orchestrator(
        llm_provider=args.provider,
        llm_model=args.model,
        llm_base_url=args.base_url,
        budget_mode=args.budget
    )
    
    # Run task
    print(f"\n{'='*60}")
    print(f"Task: {args.request}")
    print(f"{'='*60}\n")
    
    result = orchestrator.run_task(args.request)
    
    # Display results
    print(f"\n{'='*60}")
    print("FINAL ANSWER")
    print(f"{'='*60}\n")
    print(result.final_answer)
    
    if args.verbose:
        print(f"\n{'='*60}")
        print("EXECUTION DETAILS")
        print(f"{'='*60}\n")
        print(f"Task ID: {result.task_id}")
        print(f"Active Agents: {', '.join(result.active_agents)}")
        print(f"Final State: {result.final_state}")
        print(f"Validation: {result.validation_report['summary']}")
        print(f"\nBudget Usage:")
        budget = result.budget_usage
        print(f"  Agents: {budget['active_agents']}/{budget['max_active_agents']}")
        print(f"  Tokens: {budget['used_prompt_tokens']}/{budget['max_prompt_tokens']}")
        print(f"  Time: {budget['elapsed_seconds']:.2f}s/{budget['max_wall_clock_seconds']}s")


if __name__ == "__main__":
    main()
