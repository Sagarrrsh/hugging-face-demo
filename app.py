from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Tiny model = small + fast
generator = pipeline(
    "text-generation",
    model="sshleifer/tiny-gpt2"
)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 20

@app.on_event("startup")
def warmup():
    generator("hello", max_new_tokens=5)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate")
def generate(req: GenerateRequest):
    tokens = min(req.max_tokens, 20)
    result = generator(req.prompt, max_new_tokens=tokens)
    return {"output": result[0]["generated_text"]}
