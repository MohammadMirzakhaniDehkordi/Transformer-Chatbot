from fastapi import FastAPI
from pydantic import BaseModel
from chatbot_transformer import reply


# -- FastAPI Setup (راه‌اندازی فست‌اپی)
# FastAPI application instance
app = FastAPI()


class Query(BaseModel):
    message: str


# -- Chat API (پاسخ به سوالات)
# Function to handle chat queries
# This function takes a query, processes it, and returns a response.
@app.post("/chat")
def chat(query: Query):
    return {"response": reply(query.message)}
