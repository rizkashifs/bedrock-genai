from app.agents.base import BaseAgent
from app.models.bedrock_client import bedrock_client
from app.utils.logger import logger


class SummaryAgent(BaseAgent):
    """Generates a concise semantic summary of a dataset from its schema + sample rows."""

    def run(self, input_data: dict) -> str:
        schema_context = input_data.get("schema_context")
        sample_data = input_data.get("sample_data")

        prompt = f"""
Provide a concise, professional summary of the following dataset.
Focus on the semantic meaning you can infer from the first 5 rows and schema.

Schema & Profiling:
{schema_context}

Sample Data (First 5 rows):
{sample_data}

Summary:
"""
        logger.info("SummaryAgent generating dataset summary...")
        return bedrock_client.generate(
            [{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
