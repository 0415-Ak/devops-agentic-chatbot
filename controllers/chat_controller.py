import asyncio
from db.database import SessionLocal, User, Chat, Message, generate_uuid
from services.agent_service import get_agent
from services.callbacks import StreamingCallbackHandler

def handle_chat_request(query, user_id):
    """
    Handles a user's chat message, generates an agent response, and saves both messages.
    """
    callback = StreamingCallbackHandler()
    agent = get_agent(user_id, query.chat_id, callback=callback)

    if agent is None:
        yield {"error": "Agent not initialized properly."}
        return

    db = SessionLocal()
    try:
        # Ensure user exists
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            user = User(user_id=user_id)
            db.add(user)
            db.commit()

        # Ensure chat exists
        chat = db.query(Chat).filter(Chat.chat_id == query.chat_id).first()
        if not chat:
            yield {"error": "Chat not found. Please start a chat using /start_chat."}
            return

        # Update chat name on first message
        if not chat.chat_name or chat.chat_name.strip() == "New Chat":
            clean_msg = " ".join(query.message.strip().split())
            short_title = clean_msg[:20] + ("..." if len(clean_msg) > 20 else "")
            chat.chat_name = short_title
            db.commit()

        # Save user message
        user_msg = Message(
            chat_id=query.chat_id,
            user_id=user_id,
            role="user",
            content=query.message
        )
        db.add(user_msg)
        db.commit()
        db.refresh(user_msg)

        # Run agent in the background
        task = asyncio.create_task(agent.arun(query.message))

        async def token_stream():
            full_response = ""
            async for token in callback.aiter():
                yield token
                full_response += token
            await task

            # Save full assistant message after streaming completes
            if "AI:" in full_response:
                full_response = full_response.split("AI:", 1)[1].strip()

            bot_msg = Message(
                chat_id=query.chat_id,
                user_id=user_id,
                role="assistant",
                content=full_response
            )
            db.add(bot_msg)
            db.commit()

        return token_stream()

    except Exception as e:
        db.rollback()
        yield {"error": str(e)}

    finally:
        db.close()


def start_new_chat(user_id):
    """
    Creates a new chat session for a user.
    """
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            user = User(user_id=user_id)
            db.add(user)
            db.commit()

        chat_id = generate_uuid()
        chat = Chat(
            chat_id=chat_id,
            user_id=user_id,
            chat_name="New Chat",
            source="chatbot"
        )
        db.add(chat)
        db.commit()

        return {
            "chat_id": chat.chat_id,
            "source": chat.source,
            "created_at": chat.created_at
        }

    finally:
        db.close()

def get_all_user_chats(user_id):
    """
    Retrieves all chats for a specific user.
    """
    db = SessionLocal()
    try:
        chats = db.query(Chat).filter(Chat.user_id == user_id).order_by(Chat.created_at.desc()).all()
        return [
            {
                "chat_id": chat.chat_id,
                "chat_name": chat.chat_name,
                "source": chat.source,
                "created_at": chat.created_at
            }
            for chat in chats
        ]
    finally:
        db.close()

def get_chat_history(chat_id, user_id):
    """
    Retrieves the message history for a specific chat session.
    """
    db = SessionLocal()
    try:
        messages = db.query(Message).filter(
            Message.user_id == user_id,
            Message.chat_id == chat_id
        ).order_by(Message.timestamp.asc()).all()

        return {
            "chat_id": chat_id,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp
                }
                for msg in messages
            ]
        }
    finally:
        db.close()
