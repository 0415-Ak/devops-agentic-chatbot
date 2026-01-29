from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from controllers import chat_controller
from middleware.auth_middleware import get_current_user
from schemas import Query, ChatHistoryRequest

router = APIRouter()

@router.post("/chat", summary="Send a message to the chatbot agent")
async def chat_with_agent_route(
    query: Query,
    user_id: str = Depends(get_current_user)
):
    """
    Handles the chat request. It streams the response from the LLM agent.
    """
    return StreamingResponse(
        chat_controller.handle_chat_request(query, user_id),
        media_type="text/plain"
    )

@router.post("/start_chat", summary="Create a new chat session")
async def start_chat_route(
    user_id: str = Depends(get_current_user)
):
    """
    Creates a new chat entry in the database and returns the chat ID.
    """
    return chat_controller.start_new_chat(user_id)

@router.get("/user_chats", summary="Get a list of all user's chats")
async def get_user_chats_route(
    user_id: str = Depends(get_current_user)
):
    """
    Retrieves all chat sessions for the authenticated user.
    """
    return chat_controller.get_all_user_chats(user_id)

@router.get("/chat_history", summary="Get the message history for a specific chat")
async def get_chat_history_route(
    data: ChatHistoryRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Retrieves the complete message history for a given chat ID.
    """
    return chat_controller.get_chat_history(data.chat_id, user_id)
