from fastapi import FastAPI
from pydantic import BaseModel
from chatbot_transformer import reply


# -- FastAPI Setup (راه‌اندازی فست‌اپی)
app=FastAPI()
class Query(BaseModel):
    message:str


# -- Chat API (پاسخ به سوالات)
@app.post("/chat")
def chat(query:Query):
    return {"response": reply(query.message)}
