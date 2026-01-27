from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Text Generator API")

# Load model ONCE at startup (important)
generator = pipeline(
    "text-generation",
    model="sshleifer/tiny-gpt2"
)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50

@app.post("/generate")
def generate(req: GenerateRequest):
    result = generator(
        req.prompt,
        max_new_tokens=req.max_tokens
    )
    return {
        "output": result[0]["generated_text"]
    }

@app.get("/health")
def health():
    return {"status": "ok"}
