import warnings
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import torch

# Silence Hugging Face future warnings (clean logs)
warnings.filterwarnings("ignore", category=FutureWarning)

app = FastAPI()

# Load model (CPU, efficient, better than tiny-gpt2)
generator = pipeline(
    "text-generation",
    model="distilgpt2",
    device=-1,  # CPU
    torch_dtype=torch.float32
)

# Explicitly set pad token (removes warning)
generator.tokenizer.pad_token = generator.tokenizer.eos_token


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 40


@app.on_event("startup")
def warmup():
    # Warm up model (reduces first-request latency)
    generator(
        "Hello",
        max_new_tokens=5,
        do_sample=False,
        pad_token_id=generator.tokenizer.eos_token_id
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate(req: GenerateRequest):
    max_tokens = min(req.max_tokens, 50)

    result = generator(
        req.prompt,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=generator.tokenizer.eos_token_id,
        eos_token_id=generator.tokenizer.eos_token_id,
    )

    return {
        "prompt": req.prompt,
        "output": result[0]["generated_text"]
    }
