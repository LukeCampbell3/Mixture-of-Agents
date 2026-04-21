"""Parallel agent execution with thread safety and async support."""

import asyncio
import threading
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, Future, wait as futures_wait, FIRST_COMPLETED
from dataclasses import dataclass
import time


@dataclass
class AgentTask:
    """A task for an agent to execute."""
    agent_id: str
    agent_instance: Any
    context: Dict[str, Any]
    skill_packs: Optional[List[Any]] = None


@dataclass
class AgentResult:
    """Result from agent execution."""
    agent_id: str
    output: Dict[str, Any]
    elapsed_time: float
    success: bool
    error: Optional[str] = None


class SharedContext:
    """Thread-safe shared context for agent collaboration."""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._context: Dict[str, Any] = {}
        self._agent_outputs: Dict[str, Dict[str, Any]] = {}
        self._update_callbacks: List[Callable] = []
    
    def update(self, key: str, value: Any):
        """Thread-safe update of shared context."""
        with self._lock:
            self._context[key] = value
            # Notify callbacks
            for callback in self._update_callbacks:
                try:
                    callback(key, value)
                except Exception:
                    pass  # Don't let callback errors break execution
    
    def get(self, key: str, default: Any = None) -> Any:
        """Thread-safe read from shared context."""
        with self._lock:
            return self._context.get(key, default)
    
    def add_agent_output(self, agent_id: str, output: Dict[str, Any]):
        """Thread-safe addition of agent output."""
        with self._lock:
            self._agent_outputs[agent_id] = output
            self._context[f"agent_{agent_id}_output"] = output
    
    def get_agent_outputs(self) -> Dict[str, Dict[str, Any]]:
        """Thread-safe read of all agent outputs."""
        with self._lock:
            return self._agent_outputs.copy()
    
    def get_other_agent_outputs(self, exclude_agent_id: str) -> Dict[str, Dict[str, Any]]:
        """Get outputs from all agents except the specified one."""
        with self._lock:
            return {
                agent_id: output
                for agent_id, output in self._agent_outputs.items()
                if agent_id != exclude_agent_id
            }
    
    def register_callback(self, callback: Callable):
        """Register a callback for context updates."""
        with self._lock:
            self._update_callbacks.append(callback)
    
    def get_full_context(self) -> Dict[str, Any]:
        """Get a copy of the full context."""
        with self._lock:
            return self._context.copy()


class ParallelExecutor:
    """Execute agents in parallel with thread safety."""
    
    def __init__(self, max_workers: int = 3):
        """Initialize parallel executor.
        
        Args:
            max_workers: Maximum number of parallel agent executions
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def execute_parallel(
        self,
        tasks: List[AgentTask],
        shared_context: SharedContext,
        timeout: Optional[float] = None
    ) -> List[AgentResult]:
        """Execute multiple agent tasks in parallel.
        
        Args:
            tasks: List of agent tasks to execute
            shared_context: Shared context for collaboration
            timeout: Optional wall-clock timeout in seconds for the entire batch
        
        Returns:
            List of agent results
        """
        if not tasks:
            return []
        
        # Submit all tasks (no per-task timeout inside the worker — we enforce
        # the deadline at the batch level using futures_wait below)
        future_to_task: Dict[Future, AgentTask] = {}
        for task in tasks:
            future = self.executor.submit(
                self._execute_agent_task,
                task,
                shared_context,
                None,  # no inner timeout; outer deadline handles it
            )
            future_to_task[future] = task
        
        # Wait for all futures, respecting the wall-clock deadline
        if timeout is not None:
            done, not_done = futures_wait(
                future_to_task.keys(), timeout=timeout
            )
        else:
            done = set(future_to_task.keys())
            not_done = set()
        
        results = []
        
        # Collect completed futures
        for future in done:
            task = future_to_task[future]
            try:
                result = future.result(timeout=0)  # already done, no wait
                results.append(result)
                if result.success:
                    shared_context.add_agent_output(result.agent_id, result.output)
            except Exception as e:
                results.append(AgentResult(
                    agent_id=task.agent_id,
                    output={},
                    elapsed_time=0.0,
                    success=False,
                    error=str(e)
                ))
        
        # Cancel and record timed-out futures
        for future in not_done:
            task = future_to_task[future]
            future.cancel()
            results.append(AgentResult(
                agent_id=task.agent_id,
                output={},
                elapsed_time=timeout or 0.0,
                success=False,
                error=f"Agent timed out after {timeout}s"
            ))
        
        return results
    
    def execute_sequential_with_sharing(
        self,
        tasks: List[AgentTask],
        shared_context: SharedContext,
        timeout: Optional[float] = None
    ) -> List[AgentResult]:
        """Execute agents sequentially but with immediate context sharing.
        
        This is useful when agents need to see previous outputs before starting.
        
        Args:
            tasks: List of agent tasks to execute
            shared_context: Shared context for collaboration
            timeout: Optional timeout in seconds for each task
        
        Returns:
            List of agent results
        """
        results = []
        
        for task in tasks:
            # Update context with other agents' outputs
            task.context["other_agent_outputs"] = shared_context.get_other_agent_outputs(
                task.agent_id
            )
            
            # Execute task
            result = self._execute_agent_task(task, shared_context, timeout)
            results.append(result)
            
            # Add to shared context immediately
            if result.success:
                shared_context.add_agent_output(result.agent_id, result.output)
        
        return results
    
    def execute_pipeline(
        self,
        tasks: List[AgentTask],
        shared_context: SharedContext,
        pipeline_stages: List[List[int]],
        timeout: Optional[float] = None
    ) -> List[AgentResult]:
        """Execute agents in a pipeline with stages.
        
        Each stage executes in parallel, but stages are sequential.
        
        Args:
            tasks: List of agent tasks to execute
            shared_context: Shared context for collaboration
            pipeline_stages: List of stages, each containing task indices
            timeout: Optional timeout in seconds for each task
        
        Example:
            pipeline_stages = [[0, 1], [2]]  # Tasks 0,1 parallel, then task 2
        
        Returns:
            List of agent results
        """
        all_results = []
        
        for stage_indices in pipeline_stages:
            stage_tasks = [tasks[i] for i in stage_indices if i < len(tasks)]
            
            # Update context for all tasks in this stage
            for task in stage_tasks:
                task.context["other_agent_outputs"] = shared_context.get_other_agent_outputs(
                    task.agent_id
                )
            
            # Execute stage in parallel
            stage_results = self.execute_parallel(stage_tasks, shared_context, timeout)
            all_results.extend(stage_results)
        
        return all_results
    
    def _execute_agent_task(
        self,
        task: AgentTask,
        shared_context: SharedContext,
        timeout: Optional[float]
    ) -> AgentResult:
        """Execute a single agent task with skill packs."""
        start_time = time.time()
        
        try:
            # Apply skill packs if provided
            if task.skill_packs:
                task.context["skill_packs"] = task.skill_packs
            
            # Execute agent
            output = task.agent_instance.execute(task.context)
            
            elapsed = time.time() - start_time
            
            return AgentResult(
                agent_id=task.agent_id,
                output=output,
                elapsed_time=elapsed,
                success=True
            )
        
        except Exception as e:
            elapsed = time.time() - start_time
            return AgentResult(
                agent_id=task.agent_id,
                output={},
                elapsed_time=elapsed,
                success=False,
                error=str(e)
            )
    
    def shutdown(self, wait: bool = True):
        """Shutdown the executor."""
        self.executor.shutdown(wait=wait)


class AsyncParallelExecutor:
    """Async version of parallel executor for async agents."""
    
    def __init__(self, max_concurrent: int = 3):
        """Initialize async parallel executor.
        
        Args:
            max_concurrent: Maximum number of concurrent agent executions
        """
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def execute_parallel_async(
        self,
        tasks: List[AgentTask],
        shared_context: SharedContext,
        timeout: Optional[float] = None
    ) -> List[AgentResult]:
        """Execute multiple agent tasks in parallel asynchronously.
        
        Args:
            tasks: List of agent tasks to execute
            shared_context: Shared context for collaboration
            timeout: Optional timeout in seconds for each task
        
        Returns:
            List of agent results
        """
        if not tasks:
            return []
        
        # Create coroutines for all tasks
        coroutines = [
            self._execute_agent_task_async(task, shared_context, timeout)
            for task in tasks
        ]
        
        # Execute all in parallel
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(AgentResult(
                    agent_id=tasks[i].agent_id,
                    output={},
                    elapsed_time=0.0,
                    success=False,
                    error=str(result)
                ))
            else:
                processed_results.append(result)
                if result.success:
                    shared_context.add_agent_output(result.agent_id, result.output)
        
        return processed_results
    
    async def _execute_agent_task_async(
        self,
        task: AgentTask,
        shared_context: SharedContext,
        timeout: Optional[float]
    ) -> AgentResult:
        """Execute a single agent task asynchronously."""
        async with self.semaphore:
            start_time = time.time()
            
            try:
                # Apply skill packs if provided
                if task.skill_packs:
                    task.context["skill_packs"] = task.skill_packs
                
                # Execute agent (assuming it has async support)
                if hasattr(task.agent_instance, 'execute_async'):
                    output = await task.agent_instance.execute_async(task.context)
                else:
                    # Fallback to sync execution in thread pool
                    loop = asyncio.get_event_loop()
                    output = await loop.run_in_executor(
                        None,
                        task.agent_instance.execute,
                        task.context
                    )
                
                elapsed = time.time() - start_time
                
                return AgentResult(
                    agent_id=task.agent_id,
                    output=output,
                    elapsed_time=elapsed,
                    success=True
                )
            
            except Exception as e:
                elapsed = time.time() - start_time
                return AgentResult(
                    agent_id=task.agent_id,
                    output={},
                    elapsed_time=elapsed,
                    success=False,
                    error=str(e)
                )
