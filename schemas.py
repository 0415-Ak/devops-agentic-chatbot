from pydantic import BaseModel

# Request schema for /chat
class Query(BaseModel):
    chat_id: str
    message: str

# Request schema for /chat_history
class ChatHistoryRequest(BaseModel):
    chat_id: str
