"""
Chat Engine Module

A modular chat engine that provides a clean interface for interacting with Claude via Bedrock.
Integrates with DynamoDB for persistent conversation history.
"""

from typing import List, Dict, Optional, Any, Literal
from app.models.bedrock_client import invoke_claude
import json
import os
import uuid
from datetime import datetime


# Import DynamoDB storage Lambda
try:
    from dynamo_chat_storage import DynamoChatStorage
    DYNAMO_AVAILABLE = True
except ImportError:
    DYNAMO_AVAILABLE = False
    print("Warning: DynamoDB storage not available. Running in memory-only mode.")


# Mode type definition
ModeType = Literal["api", "chat"]


class ChatEngine:
    """
    A chat engine that manages conversations with Claude via AWS Bedrock.
    Supports persistent conversation history via DynamoDB.
    """
    
    def __init__(self, chat_id: str, system_prompt: Optional[str] = None, mode: ModeType = "api", 
                 use_dynamo: bool = True, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the chat engine.
        
        Args:
            chat_id: Unique identifier for this chat session
            system_prompt: Optional system prompt to set context for the conversation
            mode: Operation mode - "api" for structured JSON responses, "chat" for conversational
            use_dynamo: Whether to use DynamoDB for persistent storage
            metadata: Optional metadata to store with the chat (user_id, feature, etc.)
        """ 
       
        self.chat_id = chat_id
        self.system_prompt = system_prompt
        self.mode = mode
        self.use_dynamo = use_dynamo and DYNAMO_AVAILABLE
        self.metadata = metadata or {}
        

        # Initialize storage
        if self.use_dynamo:
            self.dynamo_storage = DynamoChatStorage()
            # Load existing conversation history from DynamoDB
            self.conversation_history = self.dynamo_storage.get_chat_history(chat_id)
        else:
            # In-memory storage only
            self.conversation_history: List[Dict[str, str]] = []
        
        # Track session-only exchanges (not yet saved to DynamoDB)
        self.session_exchanges: List[Dict[str, str]] = []
       

    def send_message(self, user_input: str) -> str:
        """
        Send a message to Claude and get a response.
        Always includes full conversation history (DynamoDB + session).
        
        Args:
            user_input: The user's message
            
        Returns:
            Claude's response as a string
        """
       
        try:
            # Build prompt with complete history (DynamoDB + session)
            full_prompt = self._build_prompt_with_complete_history(user_input)
            
            # Get response from Claude
            response = invoke_claude(full_prompt)
            
            # Store in session exchanges
            exchange = {
                "user": user_input,
                "assistant": response
            }
            self.session_exchanges.append(exchange)
            
            return response
            
        except Exception as e:
            error_msg = f"Error communicating with Claude: {str(e)}"
            print(error_msg)
            return error_msg
    
    def save_to_dynamo(self) -> bool:
        """
        Save all session exchanges to DynamoDB.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.use_dynamo or not self.session_exchanges:
            return False
        
        try:
            # Append all session exchanges to DynamoDB
            for exchange in self.session_exchanges:
                success = self.dynamo_storage.append_exchange(
                    self.chat_id,
                    exchange["user"],
                    exchange["assistant"],
                    self.metadata
                )
                if not success:
                    return False
            
            # Clear session exchanges after successful save
            self.session_exchanges.clear()
            
            # Reload conversation history to include newly saved exchanges
            self.conversation_history = self.dynamo_storage.get_chat_history(self.chat_id)
            
            return True
            
        except Exception as e:
            print(f"Error saving to DynamoDB: {e}")
            return False
    
    def get_complete_history(self) -> List[Dict[str, str]]:
        """
        Get complete conversation history (DynamoDB + current session).
        
        Returns:
            Complete conversation history
        """
        return self.conversation_history + self.session_exchanges
    
    def _build_prompt_with_complete_history(self, user_input: str) -> str:
        """
        Build a prompt that includes complete conversation history.
        """
        prompt_parts = []
        
        if self.system_prompt:
            prompt_parts.append(self.system_prompt)
        
        # Add complete conversation history (DynamoDB + session)
        complete_history = self.get_complete_history()
        for exchange in complete_history:
            prompt_parts.append(f"User: {exchange['user']}")
            prompt_parts.append(f"Assistant: {exchange['assistant']}")
        
        # Add current user input
        prompt_parts.append(f"User: {user_input}")
        
        return "\n\n".join(prompt_parts)
    
    def get_session_history(self) -> List[Dict[str, str]]:
        """Get only the current session history (not yet saved to DynamoDB)."""
        return self.session_exchanges.copy()
    
    def get_dynamo_history(self) -> List[Dict[str, str]]:
        """Get only the DynamoDB persisted history."""
        return self.conversation_history.copy()
    
    def set_system_prompt(self, system_prompt: str):
        """Update the system prompt."""
        self.system_prompt = system_prompt
    
def create_interactive_chat(chat_id: Optional[str] = None):
    """
    Create an interactive chat session.
    
    Args:
        chat_id: Optional chat ID for testing. If None, generates a test ID.
    """

    
    if chat_id is None:
        # Generate a test chat ID for interactive mode
        chat_id = f"interactive_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
       
    print("Claude Chat Engine")
    print("Type 'quit', 'exit', or 'bye' to end the conversation")
    print("-" * 50)
    
    # Initialize chat engine in CHAT mode for interactive use
    chat = ChatEngine(
        chat_id=chat_id,
        system_prompt="You are a helpful AI assistant. Provide clear, concise, and helpful responses.",
        mode="api", 
        use_dynamo=True,
        metadata={"source": "interactive_testing"}
    )
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
            
            
            # Get response from Claude
            print("\nClaude:", end=" ")
            response = chat.send_message(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n Goodbye!")
            break
        except Exception as e:
            print(f"\n Error: {e}")

# Global variable to cache loaded prompts
_SYSTEM_PROMPTS = None

def load_system_prompts(json_file_path: str = "system_prompts.json") -> Dict[str, str]:
    """
    Load system prompts from a JSON file.
    
    Args:
        json_file_path: Path to the JSON file containing system prompts
        
    Returns:
        Dictionary of feature -> system prompt mappings
    """
    global _SYSTEM_PROMPTS
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            _SYSTEM_PROMPTS = json.load(f)
        return _SYSTEM_PROMPTS
    except FileNotFoundError:
        #print(f"Warning: {json_file_path} not found. Using default prompts.")
        #return get_default_prompts()
        raise FileNotFoundError(f"Required system prompts file '{json_file_path}' not found. Fallback is disabled.")
    except json.JSONDecodeError as e:
        print(f"Error parsing {json_file_path}: {e}. Using default prompts.")
        return get_default_prompts()


def get_default_prompts() -> Dict[str, str]:
    """
    Get default system prompts as fallback.
    
    Returns:
        Dictionary of default system prompts
    """
    return {
        "docComparison": "You are an expert document analysis assistant specialized in comparing documents and identifying differences.",
        "qna": "You are a knowledgeable Q&A assistant that provides accurate and comprehensive answers.",
        "search": "You are a search and information retrieval assistant that helps users find relevant information efficiently.",
        "codeReview": "You are an expert code reviewer that analyzes code for bugs, best practices, and improvements."
    }


def get_system_prompts() -> Dict[str, str]:
    """
    Get the loaded system prompts, loading them if not already cached.
    
    Returns:
        Dictionary of system prompts
    """
    global _SYSTEM_PROMPTS
    
    if _SYSTEM_PROMPTS is None:
        _SYSTEM_PROMPTS = load_system_prompts()
    
    return _SYSTEM_PROMPTS


def get_system_prompt(feature: str, mode: ModeType = "api") -> str:
    """
    Get the appropriate system prompt for a feature and mode.
    
    Args:
        feature: The feature name (docComparison, qna, search, codeReview)
        mode: The operation mode ("api" or "chat")
        
    Returns:
        The corresponding system prompt with appropriate output format instructions
    """
    prompts = get_system_prompts()
    base_prompt = prompts.get(feature, prompts.get("qna", "You are a helpful AI assistant."))
    
    # Add mode-specific output format instructions
    if mode == "api":
        output_format = get_api_output_format(feature)
        return f"{base_prompt}\n\n{output_format}"
    else:
        # Chat mode - natural conversational format
        return f"{base_prompt}\n\nProvide responses in a natural, conversational format suitable for chat interfaces."


def get_api_output_format(feature: str) -> str:
    """
    Get the API output format instructions for a specific feature.
    
    Args:
        feature: The feature name
        
    Returns:
        Output format instructions for API mode
    """
    api_formats = {
        "docComparison": """OUTPUT FORMAT: Return your response as a valid JSON object with this structure:
{
  "summary": "Brief summary of the comparison",
  "differences": [
    {
      "type": "addition|deletion|modification",
      "location": "section/page reference",
      "description": "what changed",
      "importance": "high|medium|low"
    }
  ],
  "similarities": ["list of key similarities"],
  "recommendations": ["list of recommended actions"]
}""",
        
        "qna": """OUTPUT FORMAT: Return your response as a valid JSON object with this structure:
{
  "answer": "The main answer to the question",
  "confidence": "high|medium|low",
  "sources": ["list of relevant sources if applicable"],
  "related_topics": ["list of related topics for further exploration"],
  "clarifying_questions": ["questions to ask for more specific help"]
}""",
        
        "search": """OUTPUT FORMAT: Return your response as a valid JSON object with this structure:
{
  "search_strategy": "recommended approach for finding information",
  "keywords": ["list of suggested search keywords"],
  "sources": ["list of recommended sources/databases"],
  "refined_query": "optimized search query",
  "next_steps": ["actionable steps for the user"]
}""",
        
        "codeReview": """OUTPUT FORMAT: Return your response as a valid JSON object with this structure:
{
  "overall_rating": "excellent|good|needs_improvement|poor",
  "summary": "Brief summary of code quality",
  "issues": [
    {
      "type": "bug|performance|security|style|maintainability",
      "severity": "critical|high|medium|low",
      "line": "line number if applicable",
      "description": "what the issue is",
      "suggestion": "how to fix it"
    }
  ],
  "strengths": ["list of code strengths"],
  "recommendations": ["list of improvement recommendations"]
}"""
    }
    
    return api_formats.get(feature, api_formats["qna"])

def main():
    """
    Main function - runs interactive chat for local testing.
    Module is designed to be imported in Lambda functions.
    """
    print("Chat engine module loaded successfully.")
    print("Use: from chat_engine import ChatEngine")

    create_interactive_chat()


if __name__ == "__main__":
    main()