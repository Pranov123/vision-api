import base64
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

# ----------------------------
# Groq Setup
# ----------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ----------------------------
# FastAPI
# ----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageRequest(BaseModel):
    image_base64: str
    question: str

class ImageResponse(BaseModel):
    answer: str

@app.post("/answer-image", response_model=ImageResponse)
async def answer_image(req: ImageRequest):
    prompt = f"""
You are an OCR and visual reasoning assistant.
Answer ONLY the user's question.
Question: {req.question}

Rules:
- Return ONLY the final answer.
- No explanation.
- No markdown.
- No surrounding quotes.
- If the answer is a number, return only the number.
- Never include currency symbols or units unless explicitly requested.
"""

    try:
        # Llama 3.2 Vision supports base64 encoded images directly in the data URL format
        response = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{req.image_base64}"},
                        },
                    ],
                }
            ],
        )

        answer = response.choices[0].message.content.strip()
        
        # Cleanup quotes if they appear
        if answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]

        return {"answer": answer}

    except Exception as e:
        print(f"Error: {e}")
        return {"answer": ""}