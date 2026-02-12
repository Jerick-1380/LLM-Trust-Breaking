"""Async parallel OpenRouter client using asyncio and aiohttp."""
from __future__ import annotations
from typing import List, Dict, Any
import asyncio
import aiohttp
import json
import time


class AsyncOpenRouterParallelClient:
    """
    Execute multiple OpenRouter requests in parallel using asyncio.

    Much more efficient than threading - can handle 100+ concurrent requests
    with minimal memory overhead.
    """

    # Models that support structured outputs via response_format parameter
    STRUCTURED_OUTPUT_MODELS = {
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "openai/gpt-4-turbo",
        "openai/gpt-4",
        "openai/gpt-3.5-turbo",
    }

    def __init__(self, api_key: str, max_concurrent: int = 50, timeout: int = 120, max_retries: int = 3):
        """
        Initialize async parallel OpenRouter client.

        Args:
            api_key: OpenRouter API key
            max_concurrent: Maximum number of concurrent requests (default: 50)
            timeout: Request timeout in seconds (default: 120)
            max_retries: Maximum number of retry attempts (default: 3)
        """
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.max_retries = max_retries

    @property
    def supports_structured_output(self) -> bool:
        """Conservative default - use prompt-based JSON forcing."""
        return False

    def execute_parallel(self, requests_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute multiple requests in parallel using asyncio.

        This is a synchronous wrapper around the async implementation.

        Args:
            requests_list: List of request dictionaries

        Returns:
            Dictionary mapping custom_id to response
        """
        # Run async code in event loop
        return asyncio.run(self._execute_parallel_async(requests_list))

    async def _execute_parallel_async(self, requests_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Async implementation of parallel execution.

        Args:
            requests_list: List of request dictionaries

        Returns:
            Dictionary mapping custom_id to response
        """
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Configure connection pooling
        # TCPConnector manages the connection pool and reuses connections
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent * 2,  # Total connection pool size
            limit_per_host=self.max_concurrent,  # Connections to OpenRouter
            ttl_dns_cache=300,  # Cache DNS for 5 minutes
            enable_cleanup_closed=True  # Clean up closed connections
        )

        # Create aiohttp session with connection pooling
        timeout_obj = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout_obj, connector=connector) as session:
            # Create tasks for all requests
            tasks = [
                self._execute_single_async(session, semaphore, request)
                for request in requests_list
            ]

            # Execute all tasks concurrently
            results_list = await asyncio.gather(*tasks, return_exceptions=True)

            # Convert list to dict
            results = {}
            for result in results_list:
                if isinstance(result, Exception):
                    print(f"Task failed with exception: {result}")
                    continue
                if result:
                    custom_id, response = result
                    results[custom_id] = response

            return results

    async def _execute_single_async(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        request: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any]]:
        """
        Execute a single request asynchronously with retry logic.

        Args:
            session: aiohttp session
            semaphore: Semaphore to limit concurrency
            request: Request dictionary

        Returns:
            Tuple of (custom_id, response)
        """
        custom_id = request['custom_id']

        # Retry with exponential backoff
        for attempt in range(self.max_retries):
            async with semaphore:  # Limit concurrent requests
                try:
                    # Build request data
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }

                    data = {
                        "model": request['model'],
                        "messages": request['messages']
                    }

                    # Add optional parameters
                    if 'temperature' in request:
                        data['temperature'] = request['temperature']

                    if 'max_tokens' in request:
                        data['max_tokens'] = request['max_tokens']
                    elif 'max_completion_tokens' in request:
                        data['max_tokens'] = request['max_completion_tokens']
                    else:
                        data['max_tokens'] = 100

                    # Add response format if supported
                    model = request['model']
                    if 'response_format' in request and model in self.STRUCTURED_OUTPUT_MODELS:
                        data['response_format'] = request['response_format']

                    # Make async HTTP request
                    async with session.post(self.base_url, headers=headers, json=data) as response:
                        response.raise_for_status()
                        response_data = await response.json()

                        content = response_data["choices"][0]["message"]["content"]

                        # Try to parse JSON content
                        try:
                            parsed = json.loads(content)
                            if isinstance(parsed, dict):
                                return custom_id, parsed
                            else:
                                return custom_id, {"content": content}
                        except (json.JSONDecodeError, ValueError):
                            # Not JSON, return as plain content
                            return custom_id, {
                                "status": "success",
                                "content": content
                            }

                except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                    # Retryable errors
                    if attempt < self.max_retries - 1:
                        wait_time = (2 ** attempt) + (time.time() % 1)
                        print(f"Retryable error in {custom_id} (attempt {attempt + 1}/{self.max_retries}): {e}")
                        print(f"Retrying in {wait_time:.2f} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"Error in request {custom_id} after {self.max_retries} attempts: {e}")
                        return custom_id, {
                            "status": "error",
                            "error": f"Max retries exceeded: {str(e)}"
                        }

                except aiohttp.ClientResponseError as e:
                    # HTTP errors (rate limiting, server errors)
                    if e.status in [429, 500, 502, 503, 504] and attempt < self.max_retries - 1:
                        wait_time = (2 ** attempt) + (time.time() % 1)
                        # Silently retry rate limits and server errors
                        await asyncio.sleep(wait_time)
                    else:
                        if attempt == self.max_retries - 1:
                            print(f"HTTP error in request {custom_id} after {self.max_retries} attempts: {e}")
                        return custom_id, {
                            "status": "error",
                            "error": str(e)
                        }

                except Exception as e:
                    # Non-retryable errors
                    print(f"Error in request {custom_id}: {e}")
                    return custom_id, {
                        "status": "error",
                        "error": str(e)
                    }

        # Should never reach here
        return custom_id, {
            "status": "error",
            "error": "Max retries exceeded"
        }


# Backwards compatibility alias
OpenRouterParallelClient = AsyncOpenRouterParallelClient
