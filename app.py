from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import torch

app = FastAPI()

generator = pipeline(
    "text-generation",
    model="distilgpt2",
    device=-1,              # CPU
    torch_dtype=torch.float32
)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50

@app.on_event("startup")
def warmup():
    generator(
        "Hello",
        max_new_tokens=10,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate")
def generate(req: GenerateRequest):
    tokens = min(req.max_tokens, 50)

    result = generator(
        req.prompt,
        max_new_tokens=tokens,
        do_sample=True,
        temperature=0.7,          # lower = more focused
        top_p=0.9,                # nucleus sampling
        repetition_penalty=1.2,   # reduces looping
        eos_token_id=50256
    )

    return {
        "prompt": req.prompt,
        "output": result[0]["generated_text"]
    }
