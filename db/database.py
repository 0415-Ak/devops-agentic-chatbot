from sqlalchemy import create_engine, Column, String, Text, ForeignKey, TIMESTAMP
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime
from uuid6 import uuid7
from dotenv import load_dotenv
import os

load_dotenv()

# Load DB config from environment variables
host = os.getenv("DB_HOST")
dbname = os.getenv("DB_NAME")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
port = os.getenv("DB_PORT")

DATABASE_URL = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

# Setup DB engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# UUID generator function
def generate_uuid():
    """Generates a UUID7 string."""
    return str(uuid7())

# User table model
class User(Base):
    __tablename__ = 'users'
    user_id = Column(String, primary_key=True, default=generate_uuid)
    username = Column(Text)
    created_at = Column(TIMESTAMP, default=datetime.datetime.utcnow)

# Chat table model
class Chat(Base):
    __tablename__ = 'chats'
    chat_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey('users.user_id'))
    chat_name = Column(Text)
    source = Column(String, default='chatbot')
    created_at = Column(TIMESTAMP, default=datetime.datetime.utcnow)

# Message table model
class Message(Base):
    __tablename__ = 'messages'
    message_id = Column(String, primary_key=True, default=generate_uuid)
    chat_id = Column(String, ForeignKey('chats.chat_id'))
    user_id = Column(String, ForeignKey('users.user_id'))
    role = Column(Text)
    content = Column(Text)
    timestamp = Column(TIMESTAMP, default=datetime.datetime.utcnow)

def init_db():
    """Initializes the database by creating all defined tables."""
    Base.metadata.create_all(engine)
