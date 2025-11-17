import json
import re
from typing import List, Dict
from openai import OpenAI
import os

# Singleton OpenAI client
_openai_client = None

def get_openai_client():
    """Create or return global OpenAI client."""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing from environment variables")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def parse_resumes_batch(resume_texts: List[str]) -> List[Dict]:
    """
    Parse multiple resumes in ONE GPT call (10x cheaper)
    Supports up to 5 resumes in a batch.
    """

    if len(resume_texts) > 5:
        raise ValueError("Maximum 5 resumes per batch")

    # Build combined text
    batch_text = ""
    for i, text in enumerate(resume_texts):
        batch_text += (
            f"\n\n--- RESUME {i+1} START ---\n"
            f"{text[:4000]}\n"
            f"--- RESUME {i+1} END ---\n"
        )

    # Unified batch prompt
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
        client = get_openai_client()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4000
        )

        # FIX: Correct extraction for openai==1.x
        content = response.choices[0].message["content"].strip()

        # Remove accidental markdown
        content = re.sub(r"```json|```", "", content).strip()

        results = json.loads(content)
        return results

    except Exception as e:
        print("\n⚠️ Batch error:", e)
        print("→ Falling back to single-resume parser\n")

        # Fallback calls (existing logic)
        from parser import parse_resume_with_gpt
        fallback_results = []
        for text in resume_texts:
            try:
                fallback_results.append(parse_resume_with_gpt(text))
            except Exception as e2:
                print("Fallback single parser failed:", e2)
                fallback_results.append({})
        return fallback_results
