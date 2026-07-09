import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

# ----------------------------
# Groq Setup
# ----------------------------
# Ensure GROQ_API_KEY is set in your Render environment variables
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
        # Use a currently supported multimodal model
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            # Assuming incoming base64 string does NOT have the data-uri prefix.
                            # If it does, remove the prefix string below.
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
        # Log the error to the console so you can see it in Render logs
        print(f"Error during extraction: {e}")
        return {"answer": ""}