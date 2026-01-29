import asyncio
from langchain.callbacks.base import AsyncCallbackHandler

class StreamingCallbackHandler(AsyncCallbackHandler):
    """
    Callback handler for streaming LLM responses.
    """
    def __init__(self):
        self.queue = asyncio.Queue()

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """
        Adds a new token to the queue as it is received.
        """
        await self.queue.put(token)

    async def aiter(self):
        """
        An async iterator to yield tokens from the queue.
        """
        while True:
            token = await self.queue.get()
            if token is None:
                break
            yield token

    async def aclose(self):
        """
        Closes the stream by adding a sentinel value to the queue.
        """
        await self.queue.put(None)
