from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ollama import chat

app = FastAPI()

ollama_model = "qwen3:8b"

class ChatReq(BaseModel):
    input: str

@app.post("/chat")
def ollama_chat(req: ChatReq):
    response = chat(model=ollama_model, messages=[{"role":"user", "content": req.input}])
    print(f"{response}")
    return {"response": f"{response.message.content}"}
