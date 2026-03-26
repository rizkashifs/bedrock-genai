from app.agents.base import BaseAgent
from app.models.bedrock_client import bedrock_client
from app.utils.logger import logger


class AnswerAgent(BaseAgent):
    """Generates a final natural-language response from retrieved data."""

    def run(self, input_data: dict) -> str:
        query = input_data.get("query")
        retrieved_data = input_data.get("retrieved_data")
        intent = input_data.get("intent")
        file_summary = input_data.get("file_summary", "No summary available.")

        prompt = f"""
Convert the following deterministic data results into a human-readable explanation.

Dataset Summary:
{file_summary}

User Question: {query}
Query Plan: {intent}
Retrieved Data: {retrieved_data}

Rules:
- Never invent numbers.
- Add context and caveats if the data is limited.
- Be concise and factual.
"""
        logger.info("AnswerAgent synthesising response...")
        return bedrock_client.generate(
            [{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
