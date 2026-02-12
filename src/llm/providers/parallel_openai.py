"""Parallel LLM client for concurrent API requests."""
from __future__ import annotations
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import json


class ParallelLLMClient:
    """
    Execute multiple LLM requests in parallel using threading.

    Much faster than batch API for interactive use cases.
    """

    def __init__(self, api_key: str, max_workers: int = 20):
        """
        Initialize parallel client.

        Args:
            api_key: OpenAI API key
            max_workers: Maximum number of concurrent threads
        """
        self.client = OpenAI(api_key=api_key)
        self.max_workers = max_workers

    def execute_parallel(
        self,
        requests: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute multiple requests in parallel using threading.

        Args:
            requests: List of request dictionaries with:
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

        def execute_single(request):
            """Execute a single request."""
            custom_id = request['custom_id']
            try:
                # Build request parameters
                params = {
                    'model': request['model'],
                    'messages': request['messages']
                }

                # Check if this is a reasoning model first
                model = request['model']
                is_reasoning_model = model.startswith('gpt-5') or model.startswith('o1') or model.startswith('o3')

                # Add max_tokens or max_completion_tokens based on model type
                if 'max_completion_tokens' in request:
                    params['max_completion_tokens'] = request['max_completion_tokens']
                elif 'max_tokens' in request:
                    if is_reasoning_model:
                        # Reasoning models require max_completion_tokens, not max_tokens
                        params['max_completion_tokens'] = request['max_tokens']
                    else:
                        params['max_tokens'] = request['max_tokens']
                else:
                    # Default token limit
                    if is_reasoning_model:
                        params['max_completion_tokens'] = 100
                    else:
                        params['max_tokens'] = 100

                if not is_reasoning_model:
                    if 'temperature' in request:
                        params['temperature'] = request['temperature']
                    # Add seed for deterministic outputs (especially important with temperature=0)
                    if 'seed' in request:
                        params['seed'] = request['seed']
                else:
                    # Add reasoning_effort for reasoning models if provided
                    if 'reasoning_effort' in request:
                        params['reasoning_effort'] = request['reasoning_effort']

                # Add response format if provided (not supported by reasoning models)
                if 'response_format' in request:
                    params['response_format'] = request['response_format']

                # Make API call
                response = self.client.chat.completions.create(**params)

                # Extract content
                if response.choices and len(response.choices) > 0:
                    message = response.choices[0].message
                    content = message.content
                    finish_reason = response.choices[0].finish_reason

                    # Extract usage information including reasoning tokens
                    usage_info = {}
                    if hasattr(response, 'usage'):
                        usage = response.usage
                        usage_info['total_tokens'] = usage.total_tokens
                        usage_info['prompt_tokens'] = usage.prompt_tokens
                        usage_info['completion_tokens'] = usage.completion_tokens

                        # Extract reasoning tokens if available (for reasoning models)
                        if hasattr(usage, 'completion_tokens_details'):
                            details = usage.completion_tokens_details
                            if hasattr(details, 'reasoning_tokens') and details.reasoning_tokens:
                                usage_info['reasoning_tokens'] = details.reasoning_tokens

                    # Check if response was truncated
                    if not content or content.strip() == "":
                        error_msg = f"Empty response (finish_reason: {finish_reason})"
                        if usage_info:
                            error_msg += f" [tokens: {usage_info.get('completion_tokens', 0)}]"
                        return (custom_id, {"error": error_msg})

                    # Try to parse as JSON
                    try:
                        parsed = json.loads(content)
                        # Add usage info to parsed result (only for dicts, not arrays)
                        if usage_info and isinstance(parsed, dict):
                            parsed['_usage'] = usage_info
                        return (custom_id, parsed)
                    except json.JSONDecodeError:
                        # For reasoning models, try to extract JSON from text
                        # Look for JSON patterns in the text
                        import re
                        # Try to find JSON object with proper nesting support
                        # Look for {...} pattern, allowing for nested braces
                        json_match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', content, re.DOTALL)
                        if json_match:
                            try:
                                parsed = json.loads(json_match.group())
                                # Add usage info and raw content
                                if usage_info:
                                    parsed['_usage'] = usage_info
                                parsed['_raw_content'] = content
                                return (custom_id, parsed)
                            except json.JSONDecodeError:
                                pass

                        # If that fails, try looking for JSON code block
                        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                        if code_block_match:
                            try:
                                parsed = json.loads(code_block_match.group(1))
                                if usage_info:
                                    parsed['_usage'] = usage_info
                                parsed['_raw_content'] = content
                                return (custom_id, parsed)
                            except json.JSONDecodeError:
                                pass

                        result = {"content": content}
                        if usage_info:
                            result['_usage'] = usage_info
                        return (custom_id, result)
                else:
                    return (custom_id, {"error": "No response"})

            except Exception as e:
                import traceback
                error_details = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                print(f"Error in parallel request {custom_id}: {error_details}")
                return (custom_id, {"error": str(e)})

        # Execute all requests in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(execute_single, req) for req in requests]

            for future in as_completed(futures):
                custom_id, result = future.result()
                results[custom_id] = result

        return results
