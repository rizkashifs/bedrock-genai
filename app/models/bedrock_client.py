"""
AWS Bedrock client — class-based, production-ready.

Provides:
  BedrockClient.generate(messages, options)   — standard interface used by all agents
  BedrockClient.invoke_model(body, model_id)  — raw model invocation (Titan embeddings)

A module-level singleton `bedrock_client` is exported for convenience.
The raw boto3 client `bedrock` is also exported for legacy callers.
"""
import json
import os
from typing import Any, Dict, List, Optional

import boto3
from dotenv import load_dotenv

from app.config.settings import settings
from app.utils.logger import logger

load_dotenv()


class BedrockClient:
    """
    Thin wrapper around boto3 bedrock-runtime that exposes a unified
    generate(messages, options) interface consumed by every agent.

    Messages use the *standard* format:
        [{"role": "user"|"assistant"|"system", "content": "<text>"}]

    System messages are extracted and forwarded via Bedrock's `system` param.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        region: Optional[str] = None,
    ) -> None:
        self.model_id = model_id or settings.bedrock_model_id
        self.region = region or settings.aws_region

        self._client = boto3.client(
            service_name="bedrock-runtime",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=self.region,
        )
        logger.info("BedrockClient initialised. model=%s region=%s", self.model_id, self.region)

    # ── Public interface ───────────────────────────────────────────────────

    def generate(
        self,
        messages: List[Dict[str, str]],
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Send a conversation to Claude via Bedrock Converse and return the text reply.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": str}
            options:  Optional dict with keys: temperature (float), max_tokens (int)

        Returns:
            Claude's reply as a plain string.
        """
        opts = options or {}
        temperature: float = float(opts.get("temperature", 0.0))
        max_tokens: int = int(opts.get("max_tokens", settings.max_tokens))

        # Split system messages from the conversation
        system_prompts: List[Dict[str, str]] = []
        bedrock_messages: List[Dict[str, Any]] = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompts.append({"text": msg["content"]})
            else:
                bedrock_messages.append(
                    {"role": msg["role"], "content": [{"text": msg["content"]}]}
                )

        kwargs: Dict[str, Any] = {
            "modelId": self.model_id,
            "messages": bedrock_messages,
            "inferenceConfig": {
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
        }
        if system_prompts:
            kwargs["system"] = system_prompts

        try:
            logger.info("Bedrock request. model=%s", self.model_id)
            response = self._client.converse(**kwargs)
            return response["output"]["message"]["content"][0]["text"]
        except Exception as exc:
            logger.error("Bedrock generate failed: %s", exc)
            return f"Bedrock LLM Error: {exc}"

    def invoke_messages(
        self,
        messages: List[Dict[str, str]],
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Alias for generate() — matches interface expected by legacy code."""
        return self.generate(messages, options)

    def invoke_model(self, body: Dict[str, Any], model_id: Optional[str] = None) -> Dict:
        """
        Raw model invocation via InvokeModel API (used for Titan Embeddings).

        Returns the parsed JSON response body.
        """
        target_model = model_id or self.model_id
        try:
            response = self._client.invoke_model(
                modelId=target_model,
                body=json.dumps(body),
            )
            return json.loads(response["body"].read())
        except Exception as exc:
            logger.error("invoke_model failed. model=%s error=%s", target_model, exc)
            raise

    def converse(self, **kwargs) -> Dict:
        """
        Pass-through to boto3 bedrock-runtime.converse().
        For callers that build the full Bedrock message structure themselves.
        """
        try:
            return self._client.converse(**kwargs)
        except Exception as exc:
            logger.error("converse failed: %s", exc)
            raise


# ── Module-level singletons ────────────────────────────────────────────────

bedrock_client = BedrockClient()

# Raw boto3 client exported for legacy callers (document_processing, etc.)
bedrock = bedrock_client._client


# ── Legacy functional API (backwards compatibility) ────────────────────────

def invoke_claude(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 200,
    top_p: float = 0.0,
) -> str:
    """Wrap a single string into a user message and call Claude."""
    return bedrock_client.generate(
        [{"role": "user", "content": prompt}],
        options={"temperature": temperature, "max_tokens": max_tokens},
    )


def invoke_claude_messages(
    messages: List[Dict],
    temperature: float = 0.0,
    max_tokens: int = 200,
    top_p: float = 0.0,
) -> str:
    """
    Accepts either:
      - Standard format: [{"role": "user", "content": "text"}]
      - Bedrock format:  [{"role": "user", "content": [{"text": "text"}]}]
    """
    normalised: List[Dict[str, str]] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            # Bedrock format — flatten text parts
            text = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict) and "text" in part
            )
        else:
            text = str(content)
        normalised.append({"role": msg["role"], "content": text})

    return bedrock_client.generate(
        normalised,
        options={"temperature": temperature, "max_tokens": max_tokens},
    )


if __name__ == "__main__":
    reply = invoke_claude("Tell me a joke about cloud computing.")
    print("Claude says:", reply)
