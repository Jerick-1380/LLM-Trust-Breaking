"""Parallel OpenRouter client for concurrent API requests."""
from __future__ import annotations
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import json
import time


class OpenRouterParallelClient:
    """
    Execute multiple OpenRouter requests in parallel using threading.

    Similar to ParallelLLMClient but for OpenRouter API.
    """

    # Models that support structured outputs via response_format parameter
    # NOTE: Claude models on OpenRouter do NOT support response_format
    # They need prompt-based JSON forcing instead
    STRUCTURED_OUTPUT_MODELS = {
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "openai/gpt-4-turbo",
        "openai/gpt-4",
        "openai/gpt-3.5-turbo",
        # Claude models removed - they don't support response_format on OpenRouter
        # "anthropic/claude-3.5-sonnet",
        # "anthropic/claude-3-5-sonnet-20240620",
        # "anthropic/claude-3-5-sonnet-20241022",
        # "anthropic/claude-sonnet-4",
        # "anthropic/claude-sonnet-4.5",
        # "anthropic/claude-3-opus",
        # "anthropic/claude-3-haiku",
    }

    def __init__(self, api_key: str, max_workers: int = 20, timeout: int = 120, max_retries: int = 3):
        """
        Initialize parallel OpenRouter client.

        Args:
            api_key: OpenRouter API key
            max_workers: Maximum number of concurrent threads
            timeout: Request timeout in seconds (default: 120)
            max_retries: Maximum number of retry attempts (default: 3)
        """
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_retries = max_retries

    @property
    def supports_structured_output(self) -> bool:
        """
        Check if this client/model supports structured JSON output via response_format.

        For OpenRouter:
        - OpenAI models support response_format
        - Claude and other models do NOT support response_format (need prompt-based JSON)

        Returns False by default since we can't know the model at client init time.
        Individual requests should check the model in execute_parallel.
        """
        return False  # Conservative default - use prompt-based JSON forcing

    def execute_parallel(
        self,
        requests_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute multiple requests in parallel using threading.

        Args:
            requests_list: List of request dictionaries with:
                - custom_id: Unique identifier
                - model: Model name
                - messages: List of message dicts
                - response_format: Optional JSON schema
                - max_tokens: Max tokens
                - temperature: Temperature

        Returns:
            Dictionary mapping custom_id to response
        """
        results = {}

        # Create a shared requests session for connection pooling
        # This reuses TCP/SSL connections across requests
        session = requests.Session()

        # Configure connection pool size to match max_workers
        # This ensures we can reuse connections efficiently
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=self.max_workers,
            pool_maxsize=self.max_workers * 2,
            max_retries=0  # We handle retries manually
        )
        session.mount('https://', adapter)

        def execute_single(request):
            """Execute a single request with retry logic."""
            custom_id = request['custom_id']

            # Retry with exponential backoff
            for attempt in range(self.max_retries):
                try:
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }

                    # Build request data
                    data = {
                        "model": request['model'],
                        "messages": request['messages']
                    }

                    # Add temperature
                    if 'temperature' in request:
                        data['temperature'] = request['temperature']

                    # Add max_tokens
                    if 'max_tokens' in request:
                        data['max_tokens'] = request['max_tokens']
                    elif 'max_completion_tokens' in request:
                        data['max_tokens'] = request['max_completion_tokens']
                    else:
                        data['max_tokens'] = 100

                    # Add response format if model supports structured output
                    model = request['model']
                    if 'response_format' in request and model in self.STRUCTURED_OUTPUT_MODELS:
                        data['response_format'] = request['response_format']

                    # Make request with configurable timeout using shared session
                    # Session reuses TCP/SSL connections for better performance
                    response = session.post(
                        self.base_url,
                        headers=headers,
                        json=data,
                        timeout=self.timeout
                    )
                    response.raise_for_status()

                    response_data = response.json()
                    content = response_data["choices"][0]["message"]["content"]

                    # Try to parse JSON content (for structured outputs)
                    try:
                        import json as json_module
                        parsed = json_module.loads(content)
                        # If it parsed successfully, return the parsed object
                        if isinstance(parsed, dict):
                            return custom_id, parsed
                        else:
                            # If it's not a dict (e.g., just a number or string), wrap it
                            return custom_id, {"content": content}
                    except (json_module.JSONDecodeError, ValueError):
                        # Not JSON, return as plain content
                        return custom_id, {
                            "status": "success",
                            "content": content
                        }

                except (requests.exceptions.Timeout,
                        requests.exceptions.ConnectionError,
                        requests.exceptions.ChunkedEncodingError) as e:
                    # Retryable errors
                    if attempt < self.max_retries - 1:
                        wait_time = (2 ** attempt) + (time.time() % 1)  # Exponential backoff with jitter
                        print(f"Retryable error in {custom_id} (attempt {attempt + 1}/{self.max_retries}): {e}")
                        print(f"Retrying in {wait_time:.2f} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"Error in request {custom_id} after {self.max_retries} attempts: {e}")
                        return custom_id, {
                            "status": "error",
                            "error": f"Max retries exceeded: {str(e)}"
                        }

                except requests.exceptions.HTTPError as e:
                    # Check for rate limiting (429) or server errors (5xx)
                    if hasattr(e, 'response') and e.response is not None:
                        status_code = e.response.status_code
                        if status_code in [429, 500, 502, 503, 504] and attempt < self.max_retries - 1:
                            wait_time = (2 ** attempt) + (time.time() % 1)
                            # Silently retry rate limits and server errors
                            time.sleep(wait_time)
                        else:
                            # Only print if it's a final failure after retries
                            if attempt == self.max_retries - 1:
                                print(f"HTTP error in request {custom_id} after {self.max_retries} attempts: {e}")
                            return custom_id, {
                                "status": "error",
                                "error": str(e)
                            }
                    else:
                        # Only print if it's a final failure
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

        # Execute all requests in parallel
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(execute_single, req): req for req in requests_list}

                for future in as_completed(futures):
                    try:
                        custom_id, result = future.result()
                        results[custom_id] = result
                    except Exception as e:
                        print(f"Future failed: {e}")
        finally:
            # Close session to cleanup connections
            session.close()

        return results
