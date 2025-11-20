from typing import Dict, Any, Optional
from app.services.chat_engine import ChatEngine
from app.models.api_models import ChatRequest, ChatResponse
from app.services.history import HistoryService
from app.services.retrieval import RetrievalService
import uuid

class ChatOrchestrator:
    def __init__(self):
        self.history_service = HistoryService()
        self.retrieval_service = RetrievalService()

    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        """
        Orchestrate the chat flow:
        1. Initialize/Retrieve Chat Engine
        2. (Optional) Retrieve context via Retrieval service
        3. (Optional) Fetch history via History service (if not handled by ChatEngine)
        4. Send message to Chat Engine
        5. Return response
        """
        
        chat_id = request.chat_id or str(uuid.uuid4())
        
        # Retrieve history
        existing_history = self.history_service.get_history(chat_id)
        
        # Retrieve context if needed
        context = ""
        if request.feature in ["qna", "search"]: # Example logic
             context = self.retrieval_service.get_context(request.message)
        
        system_prompt = request.system_prompt
        if context:
            system_prompt = f"{system_prompt}\n\nContext:\n{context}" if system_prompt else f"Context:\n{context}"

        # Initialize ChatEngine
        chat_engine = ChatEngine(
            chat_id=chat_id,
            system_prompt=system_prompt,
            mode="api",
            use_dynamo=False, # Force use of injected history for now since Dynamo is missing
            metadata=request.metadata,
            existing_history=existing_history
        )
        
        # Send message
        response_text = chat_engine.send_message(request.message)
        
        # Save state
        # Since we are using in-memory history service, we need to manually update it
        # because ChatEngine's save_to_dynamo only works with DynamoDB
        self.history_service.add_exchange(chat_id, request.message, response_text)
        
        return ChatResponse(
            response=response_text,
            chat_id=chat_id,
            history=chat_engine.get_complete_history()
        )
