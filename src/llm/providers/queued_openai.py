"""Queue-based OpenAI client using Producer-Consumer pattern."""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Callable
import asyncio
import aiohttp
import json
import time
from openai import AsyncOpenAI


class QueuedOpenAIClient:
    """
    Execute requests using a continuous pipeline with Producer-Consumer pattern.

    Workers continuously pull from queue and process - no wave batching.
    Much more efficient than batch execution when requests have variable latency.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        max_workers: int = 50,
        timeout: int = 120,
        max_retries: int = 3
    ):
        """
        Initialize queued OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model to use (e.g., 'gpt-5-mini')
            max_workers: Number of concurrent worker tasks (default: 50)
            timeout: Request timeout in seconds (default: 120)
            max_retries: Maximum number of retry attempts (default: 3)
        """
        self.api_key = api_key
        self.model = model
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_retries = max_retries
        self.is_gpt5 = 'gpt-5' in model.lower()

    def execute_all(
        self,
        requests_list: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Execute all requests using continuous pipeline.

        This is a synchronous wrapper around the async implementation.

        Args:
            requests_list: List of request dictionaries
            progress_callback: Optional callback(completed, total) for progress updates

        Returns:
            Dictionary mapping custom_id to response
        """
        return asyncio.run(self._execute_all_async(requests_list, progress_callback))

    async def _execute_all_async(
        self,
        requests_list: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Async implementation of continuous pipeline execution.

        Producer: Puts all requests into queue upfront
        Consumers: N workers continuously pull and process

        Args:
            requests_list: List of request dictionaries
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping custom_id to response
        """
        total_requests = len(requests_list)

        # Create queue and results dict
        queue = asyncio.Queue()
        results = {}
        completed_count = 0
        results_lock = asyncio.Lock()

        # Create async OpenAI client
        client = AsyncOpenAI(api_key=self.api_key, timeout=self.timeout)

        async def worker():
            """Worker that continuously processes queue items."""
            nonlocal completed_count

            while True:
                request = await queue.get()

                # Check for sentinel value (stop signal)
                if request is None:
                    queue.task_done()
                    break

                custom_id = request['custom_id']

                # Process request with retries
                for attempt in range(self.max_retries):
                    try:
                        messages = request['messages']

                        if self.is_gpt5:
                            # GPT-5 uses responses API
                            # Combine messages into single input
                            combined_input = "\n\n".join([f"{m['role']}: {m['content']}" for m in messages])
                            combined_input += "\n\nIMPORTANT: Respond with ONLY valid JSON. No explanation, just the JSON object."

                            response = await client.responses.create(
                                model=self.model,
                                input=combined_input,
                                reasoning={"effort": "minimal"},
                                text={"verbosity": "low"},
                                max_output_tokens=max(request.get('max_tokens', 100) * 3, 300)
                            )

                            raw = response.output_text.strip()

                            # Extract JSON if wrapped
                            if "```json" in raw:
                                raw = raw.split("```json")[1].split("```")[0].strip()
                            elif "```" in raw:
                                raw = raw.split("```")[1].split("```")[0].strip()

                            parsed = json.loads(raw)
                            result = custom_id, parsed
                        else:
                            # GPT-4 and earlier use chat API with structured outputs
                            kwargs = {
                                "model": self.model,
                                "messages": messages,
                                "max_tokens": request.get('max_tokens', 100),
                                "temperature": request.get('temperature', 1.0)
                            }

                            if 'response_format' in request:
                                kwargs['response_format'] = request['response_format']

                            response = await client.chat.completions.create(**kwargs)
                            content = response.choices[0].message.content

                            # Try to parse as JSON
                            try:
                                parsed = json.loads(content)
                                result = custom_id, parsed
                            except json.JSONDecodeError:
                                result = custom_id, {"content": content}

                        # Success - store result
                        async with results_lock:
                            results[custom_id] = result[1]
                            completed_count += 1
                            if progress_callback:
                                progress_callback(completed_count, total_requests)

                        break  # Success, exit retry loop

                    except Exception as e:
                        if attempt < self.max_retries - 1:
                            # Retry with exponential backoff
                            wait_time = (2 ** attempt) + (time.time() % 1)
                            await asyncio.sleep(wait_time)
                        else:
                            # Max retries exceeded
                            async with results_lock:
                                results[custom_id] = {
                                    "status": "error",
                                    "error": f"Max retries exceeded: {str(e)}"
                                }
                                completed_count += 1
                                if progress_callback:
                                    progress_callback(completed_count, total_requests)

                queue.task_done()

        # Producer: Add all requests to queue upfront
        for request in requests_list:
            await queue.put(request)

        # Add sentinel values to signal workers to stop
        for _ in range(self.max_workers):
            await queue.put(None)

        # Consumer: Create worker tasks
        workers = [asyncio.create_task(worker()) for _ in range(self.max_workers)]

        # Wait for all tasks to complete
        await queue.join()

        # Wait for all workers to finish
        await asyncio.gather(*workers)

        return results
