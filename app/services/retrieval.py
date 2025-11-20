from app.services.document_processing import search_content
from typing import Optional

class RetrievalService:
    def get_context(self, query: str) -> str:
        """
        Retrieve relevant context for the query using the document processing module.
        """
        # search_content returns a formatted string of relevant content
        context = search_content(query)
        
        if context == "No content stored yet." or context == "Error creating query embedding.":
            return ""
            
        return context
