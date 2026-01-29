from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

from db.database import init_db
from api.routes import router

# Initialize DB tables on startup
init_db()

app = FastAPI(title="Fixora API")

# Allow CORS for all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the main API router
app.include_router(router)

@app.get("/")
async def root():
    """
    Root endpoint for the API.
    """
    return {"message": "Fixora API is running"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # Correctly reference the new folder name "ChatBot"
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
