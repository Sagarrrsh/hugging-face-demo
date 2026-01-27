from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Load tiny model (VERY IMPORTANT for size + speed)
generator = pipeline(
    "text-generation",
    model="sshleifer/tiny-gpt2"
)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 20

@app.on_event("startup")
def warmup():
    # Warm up model to avoid first-request slowness
    generator("hello", max_new_tokens=5)

@app.post("/generate")
def generate(req: GenerateRequest):
    tokens = min(req.max_tokens, 20)  # hard limit
    result = generator(
        req.prompt,
        max_new_tokens=tokens
    )
    return {"output": result[0]["generated_text"]}

@app.get("/health")
def health():
    return {"status": "ok"}
