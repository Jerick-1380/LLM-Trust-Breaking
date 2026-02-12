"""Batch LLM client using OpenAI Batch API for parallel requests."""
from __future__ import annotations
import json
import time
import tempfile
from typing import List, Dict, Any, TYPE_CHECKING
from openai import OpenAI

if TYPE_CHECKING:
    pass


class BatchLLMClient:
    """
    Handle batched LLM requests via OpenAI Batch API.

    Allows parallel execution of multiple LLM calls by submitting them as a batch,
    waiting for completion, and parsing results.
    """

    def __init__(self, api_key: str, check_interval: int = 5):
        """
        Initialize batch client.

        Args:
            api_key: OpenAI API key
            check_interval: Seconds between batch status checks
        """
        self.client = OpenAI(api_key=api_key)
        self.check_interval = check_interval

    def create_batch_request(
        self,
        custom_id: str,
        model: str,
        messages: List[Dict[str, str]],
        response_format: Dict = None,
        max_tokens: int = 100,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Create a single batch request in the required format.

        Args:
            custom_id: Unique identifier for this request
            model: Model name (e.g., "gpt-4o-mini")
            messages: List of message dicts with role and content
            response_format: Optional JSON schema for structured output
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Returns:
            Dictionary in batch request format
        """
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens
        }

        # Only add temperature if not gpt-5/o1 models
        if not (model.startswith("gpt-5") or model.startswith("o1")):
            body["temperature"] = temperature

        # Add JSON schema if provided
        if response_format:
            body["response_format"] = response_format

        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body
        }

    def execute_batch(
        self,
        requests: List[Dict[str, Any]],
        timeout: int = 3600
    ) -> Dict[str, Any]:
        """
        Execute a batch of requests and wait for completion.

        Args:
            requests: List of batch request dictionaries
            timeout: Maximum seconds to wait for batch completion

        Returns:
            Dictionary mapping custom_id to response content

        Raises:
            TimeoutError: If batch doesn't complete within timeout
            RuntimeError: If batch fails
        """
        if not requests:
            return {}

        # Create temporary JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for request in requests:
                f.write(json.dumps(request) + '\n')
            temp_file_path = f.name

        try:
            # Upload file
            with open(temp_file_path, 'rb') as f:
                file_obj = self.client.files.create(
                    file=f,
                    purpose="batch"
                )

            # Create batch
            batch = self.client.batches.create(
                input_file_id=file_obj.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )

            # Poll for completion
            start_time = time.time()
            while True:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Batch {batch.id} did not complete within {timeout}s")

                batch = self.client.batches.retrieve(batch.id)

                if batch.status == "completed":
                    break
                elif batch.status == "failed":
                    raise RuntimeError(f"Batch {batch.id} failed")
                elif batch.status == "expired":
                    raise RuntimeError(f"Batch {batch.id} expired")
                elif batch.status == "cancelled":
                    raise RuntimeError(f"Batch {batch.id} was cancelled")

                # Still in progress, wait before checking again
                time.sleep(self.check_interval)

            # Download and parse results
            results = {}
            if batch.output_file_id:
                file_response = self.client.files.content(batch.output_file_id)
                file_contents = file_response.text

                # Parse JSONL results
                for line in file_contents.strip().split('\n'):
                    if line:
                        result = json.loads(line)
                        custom_id = result["custom_id"]

                        # Extract the response content
                        if result.get("error"):
                            results[custom_id] = {"error": result["error"]}
                        else:
                            response_body = result["response"]["body"]

                            # Handle both regular and JSON responses
                            if "choices" in response_body and len(response_body["choices"]) > 0:
                                choice = response_body["choices"][0]
                                message = choice.get("message", {})

                                # Try to extract JSON from response_format
                                if "content" in message:
                                    content = message["content"]
                                    # If it's JSON, parse it
                                    try:
                                        results[custom_id] = json.loads(content)
                                    except json.JSONDecodeError:
                                        results[custom_id] = {"content": content}
                                else:
                                    results[custom_id] = message
                            else:
                                results[custom_id] = {"error": "No choices in response"}

            return results

        finally:
            # Clean up temp file
            import os
            try:
                os.unlink(temp_file_path)
            except:
                pass

    def cancel_batch(self, batch_id: str) -> bool:
        """
        Cancel a running batch.

        Args:
            batch_id: ID of the batch to cancel

        Returns:
            True if cancellation was initiated
        """
        try:
            self.client.batches.cancel(batch_id)
            return True
        except Exception:
            return False
