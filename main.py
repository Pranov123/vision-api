import base64
import binascii
import os
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MODEL_ID = os.environ.get("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

SYSTEM_PROMPT = (
    "You are an information-extraction engine. You will be shown an image "
    "(a table, invoice, receipt, chart, or similar document) and a question "
    "about it. Read the image carefully and answer the question.\n\n"
    "STRICT OUTPUT RULES:\n"
    "1. Reply with ONLY the raw answer value. No explanation, no restated "
    "question, no leading/trailing words.\n"
    "2. If the answer is a number, output just the number (e.g. 4089.35). "
    "Do NOT include currency symbols, commas as thousands separators, units, "
    "or percent signs.\n"
    "3. If the answer is text (e.g. a category or vendor name), output the "
    "exact label as it appears in the image, with no extra commentary.\n"
    "4. Never wrap the answer in quotes, markdown, or punctuation unless it "
    "is part of the value itself."
)

app = FastAPI(title="Pandal Vision QA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


class AnswerImageRequest(BaseModel):
    image_base64: str
    question: str


class AnswerImageResponse(BaseModel):
    answer: str


def _normalize_base64(raw: str) -> str:
    """Strip a data: URI prefix if present and return the bare base64 payload."""
    if raw.startswith("data:"):
        match = re.match(r"^data:[^;]+;base64,(.*)$", raw, re.DOTALL)
        if match:
            return match.group(1)
    return raw


def _clean_answer(text: str) -> str:
    """Post-process the model's reply into a bare answer string."""
    answer = text.strip()
    # Strip surrounding quotes / backticks if the model added them anyway.
    answer = answer.strip("`\"' \n\t")
    # If it's purely numeric with stray commas/currency symbols, clean those.
    numeric_candidate = re.sub(r"[,$₹€£\s]", "", answer)
    if re.fullmatch(r"-?\d+(\.\d+)?", numeric_candidate):
        return numeric_candidate
    return answer


@app.get("/")
def health_check():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/answer-image", response_model=AnswerImageResponse)
def answer_image(payload: AnswerImageRequest):
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="Server misconfiguration: GROQ_API_KEY is not set.",
        )

    if not payload.image_base64 or not payload.question:
        raise HTTPException(
            status_code=400, detail="Both image_base64 and question are required."
        )

    b64_payload = _normalize_base64(payload.image_base64)

    # Validate it's actually decodable base64 before sending upstream.
    try:
        base64.b64decode(b64_payload, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="image_base64 is not valid base64.")

    data_url = f"data:image/png;base64,{b64_payload}"

    try:
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": payload.question},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            temperature=0,
            max_completion_tokens=256,
            top_p=1,
            stream=False,
        )
    except Exception as exc:  # surface upstream errors clearly to the caller
        raise HTTPException(status_code=502, detail=f"Groq API error: {exc}")

    raw_text = completion.choices[0].message.content or ""
    answer = _clean_answer(raw_text)

    return AnswerImageResponse(answer=answer)