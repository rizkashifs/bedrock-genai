import json
from app.agents.base import BaseAgent
from app.models.bedrock_client import bedrock_client
from app.utils.logger import logger


class CSVReasoningAgent(BaseAgent):
    """Converts a natural-language question into a structured JSON query plan."""

    def run(self, input_data: dict) -> dict:
        query = input_data.get("query")
        schema_context = input_data.get("schema_context")
        file_summary = input_data.get("file_summary", "No summary available.")

        prompt = f"""
Represent the user's data question as a structured JSON query plan.

Dataset Summary:
{file_summary}

Schema Context:
{schema_context}

User Question: {query}

Return a JSON object with:
- "operation": "sum" | "avg" | "count" | "max" | "min" | "correlation" | "filter" | "none"
- "columns": [list of relevant column names]
- "filters": {{ "column_name": "value" }}
- "group_by": [list of columns]

Rules:
- Output ONLY valid JSON.
- No markdown blocks.
- No natural language explanations.
- Use only columns present in the schema.
"""
        logger.info("CSVReasoningAgent extracting intent for: %s", str(query)[:50])
        response = bedrock_client.generate(
            [{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        try:
            return json.loads(response.strip())
        except Exception:
            return {"operation": "none", "columns": [], "filters": {}, "group_by": []}
