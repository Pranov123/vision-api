import base64
import io
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import google.generativeai as genai

# ----------------------------
# Gemini
# ----------------------------

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

# ----------------------------
# FastAPI
# ----------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Models
# ----------------------------

class ImageRequest(BaseModel):
    image_base64: str
    question: str


class ImageResponse(BaseModel):
    answer: str


# ----------------------------
# Endpoint
# ----------------------------

@app.post("/answer-image", response_model=ImageResponse)
async def answer_image(req: ImageRequest):
    try:
        image_bytes = base64.b64decode(req.image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return {"answer": ""}

    prompt = f"""
You are an OCR and visual reasoning assistant.

Answer ONLY the user's question.

Question:
{req.question}

Rules:
- Return ONLY the final answer.
- No explanation.
- No markdown.
- No surrounding quotes.
- If the answer is a number, return only the number.
- Never include currency symbols or units unless explicitly requested.
- Read tables, receipts, invoices, charts and scanned documents accurately.
"""

    try:
        response = model.generate_content([prompt, image])

        answer = response.text.strip()

        if answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]

        return {"answer": answer}

    except Exception:
        return {"answer": ""}