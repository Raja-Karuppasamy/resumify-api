import json
import re
from typing import List, Dict
import openai
import os

def ensure_openai_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing from environment variables")
    openai.api_key = api_key


def _extract_content_from_choice(choice) -> str:
    msg = choice.message
    if hasattr(msg, "content"):
        return msg.content
    return msg["content"]


def parse_resumes_batch(resume_texts: List[str]) -> List[Dict]:
    """
    Parse multiple resumes in ONE GPT call (10x cheaper)
    Supports up to 5 resumes in a batch.
    """

    if len(resume_texts) > 5:
        raise ValueError("Maximum 5 resumes per batch")

    batch_text = ""
    for i, text in enumerate(resume_texts):
        batch_text += (
            f"\n\n--- RESUME {i+1} START ---\n"
            f"{text[:4000]}\n"
            f"--- RESUME {i+1} END ---\n"
        )

    prompt = f"""
Extract information from these {len(resume_texts)} resumes and return a JSON array.

{batch_text}

Return a JSON array with EXACTLY {len(resume_texts)} objects.
Each JSON object must follow EXACTLY this schema:

{{
  "name": "Full Name",
  "email": "email@example.com",
  "phone": "(123) 456-7890",
  "location": "City, State",
  "experience": [],
  "education": [],
  "skills": [],
  "certifications": [],
  "summary": ""
}}

RULES:
- Only return valid JSON
- No markdown, no explanations, no comments
- Output MUST be a JSON array of {len(resume_texts)} objects
"""

    try:
        ensure_openai_key()

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4000
        )

        content = _extract_content_from_choice(response.choices[0]).strip()
        content = re.sub(r"```json|```", "", content).strip()

        results = json.loads(content)
        return results

    except Exception as e:
        print("\n⚠️ Batch error:", e)
        print("→ Falling back to single-resume parser\n")

        from parser import parse_resume_with_gpt
        fallback_results = []
        for text in resume_texts:
            try:
                fallback_results.append(parse_resume_with_gpt(text))
            except Exception as e2:
                print("Fallback single parser failed:", e2)
                fallback_results.append({})
        return fallback_results
