''' How to run:
    python test.py --mode quick   # Run a quick one-off test
    python test.py --mode both    # Run both the quick test and then start chat
    python test.py --mode chat --prompt codeReview   # Interactive chat using codeReview prompt:
    python test.py --mode chat --prompt qna   # Use a different system prompt from system_prompts.json
    
'''


import argparse
from app.services.chat_engine import ChatEngine, get_system_prompt


def run_quick_test(prompt='qna'):
    chat = ChatEngine(
        chat_id="test123",
        system_prompt=get_system_prompt("qna", mode="chat"),
        mode="api",
        use_dynamo=True  # set False to stay in memory-only mode
    )
    print(">>> Running quick test")
    response = chat.send_message("What is the capital of France?")
    print("Claude:", response)
    print(">>> Quick test complete\n")


def run_repl(prompt='qna'):
    chat = ChatEngine(
        chat_id="test123",
        system_prompt=get_system_prompt("qna", mode="chat"),
        mode="chat",
        use_dynamo=True
    )
    print(">>> Starting interactive chat (type 'exit' or 'quit' to end)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            print("Exiting chat.")
            break
        response = chat.send_message(user_input)
        print("Claude:", response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat testing tool")
    parser.add_argument(
        "--mode",
        choices=["quick", "chat", "both"],
        default="quick",
        help="Run mode: quick (one-off test), chat (interactive), both (default)"
    )
    parser.add_argument(
        "--prompt",
        default="qna",
        help="System prompt name to use from system_prompts.json (default: qna)"
    )
    args = parser.parse_args()

    if args.mode in ("quick", "both"):
        run_quick_test(args.prompt)

    if args.mode in ("chat", "both"):
        run_repl(args.prompt)