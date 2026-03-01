from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ollama import chat

app = FastAPI()

ollama_model = "qwen3:8b"

class Item(BaseModel):
    id:int
    name: str

temp_items = {
    1: Item(name="item1", id=1),
    2: Item(name="item2", id=2),
}

@app.get("/")
def root():
    return {"message": "Hello World"}  

@app.get("/items/{item_id}")
def read_item(item_id: int):
    if item_id not in temp_items:
        raise HTTPException(status_code=404, detail="Item not found")
    else:
        return temp_items[item_id]


# class ChatReq(BaseModel):
#     input: str

# @app.post("/chat")
# def ollama_chat(req: ChatReq):
#     response = chat(model=ollama_model, messages=[{"role":"user", "content": req.input}])
#     print(f"{response}")
#     return {"response": f"{response.message.content}"}
