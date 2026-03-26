import json
from app.agents.base import BaseAgent
from app.models.bedrock_client import bedrock_client
from app.utils.logger import logger


class FileSelectorAgent(BaseAgent):
    """Selects the most relevant file(s) from the registry for a given query."""

    def run(self, input_data: dict) -> list:
        query = input_data.get("query")
        file_summaries = input_data.get("file_summaries")

        if not file_summaries:
            return []

        prompt = f"""
You are an expert data router. Given a user query and a set of available data files
(with their summaries), identify which file(s) are necessary to answer the query.

Available Files:
{file_summaries}

User Query: {query}

Return ONLY a JSON list of the relevant File Paths.
Example Output: ["data/sales.csv", "data/inventory.xlsx"]
If none are relevant, return [].
"""
        logger.info("FileSelectorAgent selecting files for: %s", str(query)[:50])
        response = bedrock_client.generate(
            [{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )

        try:
            clean = response.strip()
            if "```json" in clean:
                clean = clean.split("```json")[-1].split("```")[0].strip()
            elif "```" in clean:
                clean = clean.split("```")[-1].split("```")[0].strip()
            return json.loads(clean)
        except Exception as exc:
            logger.error("FileSelectorAgent parse error: %s. Raw: %s", exc, response)
            return []
