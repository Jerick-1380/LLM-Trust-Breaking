"""Queue-based OpenAI client using Producer-Consumer pattern."""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Callable
import asyncio
import aiohttp
import json
import time
from tqdm import tqdm


class QueuedOpenAIClient:
    """
    Execute requests using a continuous pipeline with Producer-Consumer pattern.

    Workers continuously pull from queue and process - no wave batching.
    Much more efficient than batch execution when requests have variable latency.
    """

    # Models that support structured outputs via response_format parameter
    STRUCTURED_OUTPUT_MODELS = {
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5-pro",
        "gpt-5.2",
        "o1-mini",
        "o1-preview",
        "o3-mini",
    }

    # Reasoning models that don't support temperature/seed
    REASONING_MODELS = {
        "o1-mini",
        "o1-preview",
        "o3-mini",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5-pro",
        "gpt-5.2",
    }

    # GPT-5 models that use responses API instead of chat completions
    GPT5_MODELS = {
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5-pro",
        "gpt-5.2",
    }

    def __init__(
        self,
        api_key: str,
        max_workers: int = 50,
        timeout: int = 120,
        max_retries: int = 3
    ):
        """
        Initialize queued OpenAI client.

        Args:
            api_key: OpenAI API key
            max_workers: Number of concurrent worker tasks (default: 50)
            timeout: Request timeout in seconds (default: 120)
            max_retries: Maximum number of retry attempts (default: 3)
        """
        self.api_key = api_key
        self.chat_url = "https://api.openai.com/v1/chat/completions"
        self.responses_url = "https://api.openai.com/v1/responses"
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_retries = max_retries

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

        # Configure connection pooling
        connector = aiohttp.TCPConnector(
            limit=self.max_workers * 2,
            limit_per_host=self.max_workers,
            ttl_dns_cache=300,
            enable_cleanup_closed=True
        )

        timeout_obj = aiohttp.ClientTimeout(total=self.timeout)

        async with aiohttp.ClientSession(timeout=timeout_obj, connector=connector) as session:
            # Producer: Add all requests to queue upfront
            for request in requests_list:
                await queue.put(request)

            # Add sentinel values to signal workers to stop
            for _ in range(self.max_workers):
                await queue.put(None)

            # Consumer: Create worker tasks
            async def worker():
                nonlocal completed_count

                while True:
                    request = await queue.get()

                    # Check for sentinel value
                    if request is None:
                        queue.task_done()
                        break

                    try:
                        # Execute request
                        custom_id, response = await self._execute_single_async(
                            session, request
                        )

                        # Store result
                        async with results_lock:
                            results[custom_id] = response
                            completed_count += 1

                            # Call progress callback
                            if progress_callback:
                                progress_callback(completed_count, total_requests)

                    except Exception as e:
                        # Log error but continue processing
                        custom_id = request.get('custom_id', 'unknown')
                        print(f"Worker error for {custom_id}: {e}")

                        # Store error response
                        async with results_lock:
                            results[custom_id] = {
                                "status": "error",
                                "error": str(e)
                            }

                    finally:
                        queue.task_done()

            # Start all worker tasks
            workers = [asyncio.create_task(worker()) for _ in range(self.max_workers)]

            # Wait for all tasks to complete
            await queue.join()
            await asyncio.gather(*workers)

        return results

    async def _execute_single_async(
        self,
        session: aiohttp.ClientSession,
        request: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any]]:
        """
        Execute a single request with retry logic.

        Args:
            session: aiohttp session
            request: Request dictionary with custom_id, model, messages, etc.

        Returns:
            (custom_id, response_dict)
        """
        custom_id = request["custom_id"]
        model = request["model"]

        # Route to appropriate API based on model
        if model in self.GPT5_MODELS:
            return await self._execute_gpt5_async(session, request)
        else:
            return await self._execute_chat_async(session, request)

    async def _execute_chat_async(
        self,
        session: aiohttp.ClientSession,
        request: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any]]:
        """Execute request using chat completions API (GPT-4, etc.)."""
        custom_id = request["custom_id"]
        model = request["model"]
        messages = request["messages"]
        max_tokens = request.get("max_tokens", 100)
        temperature = request.get("temperature", 1.0)
        response_format = request.get("response_format")

        # Build request payload
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens
        }

        # Add temperature only for non-reasoning models
        if model not in self.REASONING_MODELS:
            payload["temperature"] = temperature

        # Add structured output if supported
        if response_format and model in self.STRUCTURED_OUTPUT_MODELS:
            payload["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Retry loop
        for attempt in range(self.max_retries):
            try:
                async with session.post(
                    self.chat_url,
                    json=payload,
                    headers=headers
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

                    # Extract content
                    content = data["choices"][0]["message"]["content"]

                    # Try to parse as JSON if needed
                    if response_format:
                        try:
                            # Native structured output or markdown-wrapped JSON
                            if "```json" in content:
                                content = content.split("```json")[1].split("```")[0].strip()
                            elif "```" in content:
                                content = content.split("```")[1].split("```")[0].strip()

                            parsed = json.loads(content)
                            return (custom_id, parsed)
                        except (json.JSONDecodeError, ValueError):
                            # Return raw content if JSON parsing fails
                            return (custom_id, {"content": content})
                    else:
                        # Simple text response
                        return (custom_id, {"content": content})

            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                # Retryable errors
                if attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) + (time.time() % 1)
                    await asyncio.sleep(wait_time)
                else:
                    return (custom_id, {
                        "status": "error",
                        "error": f"Max retries exceeded: {str(e)}"
                    })

            except aiohttp.ClientResponseError as e:
                # HTTP errors (rate limiting, server errors)
                if e.status in [429, 500, 502, 503, 504] and attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) + (time.time() % 1)
                    await asyncio.sleep(wait_time)
                else:
                    return (custom_id, {
                        "status": "error",
                        "error": str(e)
                    })

            except Exception as e:
                # Non-retryable errors
                return (custom_id, {
                    "status": "error",
                    "error": str(e)
                })

        # Should never reach here, but just in case
        return (custom_id, {
            "status": "error",
            "error": "Max retries exceeded"
        })

    async def _execute_gpt5_async(
        self,
        session: aiohttp.ClientSession,
        request: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any]]:
        """Execute request using responses API (GPT-5 models)."""
        custom_id = request["custom_id"]
        model = request["model"]
        messages = request["messages"]
        max_tokens = request.get("max_tokens", 100)
        response_format = request.get("response_format")

        # Convert messages to input format for responses API
        # Combine all messages into structured input array
        input_messages = []
        for msg in messages:
            input_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        # If we need JSON output, add instruction to the last user message
        if response_format:
            # Extract field names from schema if available
            field_names = []
            if isinstance(response_format, dict) and "json_schema" in response_format:
                schema = response_format["json_schema"].get("schema", {})
                properties = schema.get("properties", {})
                field_names = list(properties.keys())

            # Add JSON instruction to help GPT-5 understand we need JSON with specific fields
            if input_messages and input_messages[-1]["role"] == "user":
                if field_names:
                    fields_str = ", ".join([f'"{field}"' for field in field_names])
                    input_messages[-1]["content"] += f"\n\nIMPORTANT: Respond with ONLY valid JSON containing these exact fields: {fields_str}. No markdown formatting, no explanation, just the JSON object."
                else:
                    input_messages[-1]["content"] += "\n\nIMPORTANT: Respond with ONLY valid JSON matching the required schema. No markdown formatting, no explanation, just the JSON object."

        # Build request payload for responses API
        payload = {
            "model": model,
            "input": input_messages
        }

        # GPT-5 uses reasoning effort instead of max_tokens for output control
        # We'll use a minimal reasoning effort for faster responses
        # Note: max_output_tokens controls the actual output length
        if max_tokens:
            payload["max_output_tokens"] = max(max_tokens * 3, 300)  # GPT-5 needs more headroom

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Retry loop
        for attempt in range(self.max_retries):
            try:
                async with session.post(
                    self.responses_url,
                    json=payload,
                    headers=headers
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

                    # Extract output from responses API
                    # Response format: {
                    #   "output": [{
                    #     "type": "message",
                    #     "content": [{"type": "output_text", "text": "..."}]
                    #   }]
                    # }
                    if "output" in data and len(data["output"]) > 0:
                        # Navigate: output[0] -> content[0] -> text
                        output_msg = data["output"][0]
                        if "content" in output_msg and len(output_msg["content"]) > 0:
                            output_text = output_msg["content"][0].get("text", "")
                        else:
                            output_text = ""

                        # Try to parse as JSON if we expected structured output
                        if response_format:
                            try:
                                # Clean up potential markdown wrapping
                                content = output_text.strip()
                                if "```json" in content:
                                    content = content.split("```json")[1].split("```")[0].strip()
                                elif "```" in content:
                                    content = content.split("```")[1].split("```")[0].strip()

                                parsed = json.loads(content)
                                return (custom_id, parsed)
                            except (json.JSONDecodeError, ValueError) as e:
                                # Return raw content if JSON parsing fails
                                return (custom_id, {"content": output_text})
                        else:
                            # Simple text response
                            return (custom_id, {"content": output_text})
                    else:
                        # Unexpected response format
                        return (custom_id, {
                            "status": "error",
                            "error": "Unexpected response format from GPT-5 API"
                        })

            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                # Retryable errors
                if attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) + (time.time() % 1)
                    await asyncio.sleep(wait_time)
                else:
                    return (custom_id, {
                        "status": "error",
                        "error": f"Max retries exceeded: {str(e)}"
                    })

            except aiohttp.ClientResponseError as e:
                # HTTP errors (rate limiting, server errors)
                if e.status in [429, 500, 502, 503, 504] and attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) + (time.time() % 1)
                    await asyncio.sleep(wait_time)
                else:
                    return (custom_id, {
                        "status": "error",
                        "error": str(e)
                    })

            except Exception as e:
                # Non-retryable errors
                return (custom_id, {
                    "status": "error",
                    "error": str(e)
                })

        # Should never reach here, but just in case
        return (custom_id, {
            "status": "error",
            "error": "Max retries exceeded"
        })
