from chat import ChatEngine, get_system_prompt

def lambda_handler(event, context):
    user_input = event.get("message", "")
    chat_id = event.get("chat_id", "default_session")

    chat = ChatEngine(
        chat_id=chat_id,
        system_prompt=get_system_prompt("qna", mode="api"),
        mode="api",
        use_dynamo=True,
        metadata={"source": "lambda"}
    )

    response = chat.send_message(user_input)
    chat.save_to_dynamo()

    return {
        "statusCode": 200,
        "body": {
            "response": response,
            "history": chat.get_complete_history()
        }
    }