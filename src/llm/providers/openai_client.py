"""OpenAI LLM client implementation."""
from __future__ import annotations
import json
from typing import Dict, Any
from openai import OpenAI
from src.llm.base import BaseLLMClient

class OpenAIClient(BaseLLMClient):
    """OpenAI API client with structured output support."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.7):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model identifier (e.g., 'gpt-4o-mini', 'gpt-4', 'gpt-5', 'gpt-5-mini')
            temperature: Sampling temperature
        """
        super().__init__(model, temperature)
        self.client = OpenAI(api_key=api_key)
        # GPT-5 models use the new responses API
        self.is_gpt5 = model.startswith("gpt-5")
        # o1 models use chat API but with different parameters
        self.is_o1 = model.startswith("o1")

    def call_with_json_schema(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: Dict[str, Any],
        max_tokens: int = 100
    ) -> Dict[str, Any]:
        """
        Call OpenAI with structured JSON output.

        Args:
            system_prompt: System message
            user_prompt: User message
            json_schema: JSON schema for structured output
            max_tokens: Max tokens to generate

        Returns:
            Parsed JSON response
        """
        try:
            # GPT-5 models use the new responses API
            if self.is_gpt5:
                # Combine prompts and request JSON output
                combined_input = f"{system_prompt}\n\n{user_prompt}\n\nIMPORTANT: Respond with ONLY valid JSON. No explanation, just the JSON object."

                # GPT-5 needs more tokens than GPT-4 for same output due to reasoning
                gpt5_tokens = max(max_tokens * 3, 300)  # Triple the tokens for GPT-5

                response = self.client.responses.create(
                    model=self.model,
                    input=combined_input,
                    reasoning={"effort": "low"},
                    text={"verbosity": "low"},
                    max_output_tokens=gpt5_tokens
                )

                raw = response.output_text.strip()

                # Try to extract JSON if wrapped in markdown code blocks
                if "```json" in raw:
                    raw = raw.split("```json")[1].split("```")[0].strip()
                elif "```" in raw:
                    raw = raw.split("```")[1].split("```")[0].strip()

                # Clean up the JSON string - remove any trailing/leading quotes or formatting
                raw = raw.strip()
                if raw.startswith('"') and not raw.endswith('"'):
                    # Fix unterminated strings by finding the last complete JSON object
                    try:
                        # Try to find a complete JSON object
                        brace_count = 0
                        last_complete = 0
                        for i, char in enumerate(raw):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    last_complete = i + 1
                        if last_complete > 0:
                            raw = raw[:last_complete]
                    except:
                        pass

                try:
                    return json.loads(raw)
                except json.JSONDecodeError as e:
                    # If JSON parsing fails, try to extract and repair from the text
                    print(f"JSON decode error: {e}, attempting to parse from text: {raw[:100]}...")
                    import re

                    # Try to find dm_to and message fields manually
                    dm_to_match = re.search(r'"dm_to"\s*:\s*"([^"]*)"', raw)
                    message_match = re.search(r'"message"\s*:\s*"([^"]*)', raw)  # May be unterminated

                    if dm_to_match:
                        dm_to = dm_to_match.group(1)
                        message = message_match.group(1) if message_match else ""
                        # Reconstruct valid JSON
                        return {"dm_to": dm_to, "message": message}

                    # Try to find point_to field manually
                    point_to_match = re.search(r'"point_to"\s*:\s*"([^"]*)"', raw)
                    if point_to_match:
                        return {"point_to": point_to_match.group(1)}

                    # Last resort: look for any complete JSON object
                    json_match = re.search(r'\{[^{}]*\}', raw)
                    if json_match:
                        try:
                            return json.loads(json_match.group(0))
                        except:
                            pass

                    raise

            # o1 models use chat API but no structured outputs
            elif self.is_o1:
                enhanced_user_prompt = user_prompt + "\n\nIMPORTANT: You must respond with valid JSON only, no other text."

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": f"{system_prompt}\n\n{enhanced_user_prompt}"},
                    ],
                    max_completion_tokens=max_tokens
                )

                raw = response.choices[0].message.content.strip()

                # Try to extract JSON if wrapped in markdown code blocks
                if raw.startswith("```json"):
                    raw = raw.split("```json")[1].split("```")[0].strip()
                elif raw.startswith("```"):
                    raw = raw.split("```")[1].split("```")[0].strip()

                return json.loads(raw)

            else:
                # GPT-4 and earlier support structured outputs
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": json_schema
                    },
                    temperature=self.temperature,
                    max_tokens=max_tokens
                )

                raw = response.choices[0].message.content.strip()
                return json.loads(raw)

        except Exception as e:
            # Fallback: return a valid but minimal response
            print(f"Warning: LLM call failed with error: {e}")
            # Extract required fields from schema and provide defaults
            required = json_schema.get("schema", {}).get("required", [])
            return {field: "" for field in required}

    def call_simple(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 150
    ) -> str:
        """
        Call OpenAI for simple text generation.

        Args:
            system_prompt: System message
            user_prompt: User message
            max_tokens: Max tokens to generate

        Returns:
            Generated text
        """
        try:
            # GPT-5 models use the new responses API
            if self.is_gpt5:
                combined_input = f"{system_prompt}\n\n{user_prompt}"

                # GPT-5 needs more tokens than GPT-4 for same output due to reasoning
                gpt5_tokens = max(max_tokens * 3, 450)  # Triple the tokens for GPT-5

                response = self.client.responses.create(
                    model=self.model,
                    input=combined_input,
                    reasoning={"effort": "low"},
                    text={"verbosity": "low"},
                    max_output_tokens=gpt5_tokens
                )

                return response.output_text.strip()

            # o1 models use chat API but no system messages or temperature
            elif self.is_o1:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"},
                    ],
                    max_completion_tokens=max_tokens
                )

                return response.choices[0].message.content.strip()

            else:
                # GPT-4 and earlier models
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=max_tokens
                )

                return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Warning: LLM call failed with error: {e}")
            return ""
