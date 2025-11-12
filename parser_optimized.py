import json
import re
from typing import List, Dict
from openai import OpenAI
import os

_openai_client = None

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv('OPENAI_API_KEY')
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def parse_resumes_batch(resume_texts: List[str]) -> List[Dict]:
    """Parse multiple resumes in ONE API call (10x cheaper!)"""
    
    if len(resume_texts) > 5:
        raise ValueError("Maximum 5 resumes per batch")
    
    batch_text = ""
    for i, text in enumerate(resume_texts):
        batch_text += f"\n\n--- RESUME {i+1} START ---\n{text[:4000]}\n--- RESUME {i+1} END ---\n"
    
    prompt = f"""Extract information from these {len(resume_texts)} resumes and return a JSON array.

{batch_text}

Return a JSON array with {len(resume_texts)} objects. Each object should have:
{{
  "name": "Full Name",
  "email": "email@example.com",
  "phone": "(123) 456-7890",
  "location": "City, State",
  "experience": [{{"job_title": "...", "company": "...", "start_date": "...", "end_date": "...", "responsibilities": ["..."]}},],
  "education": [{{"degree": "...", "major": "...", "institution": "...", "start_date": "...", "end_date": "..."}}],
  "skills": ["skill1", "skill2"],
  "summary": "..."
}}

Return ONLY the JSON array of {len(resume_texts)} objects. No markdown, no explanations."""

    try:
        client = get_openai_client()
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4000
        )
        
        json_str = response.choices[0].message.content.strip()
        json_str = re.sub(r'`{3}(?:json)?\s*', '', json_str).strip()
        
        results = json.loads(json_str)
        return results
        
    except Exception as e:
        print(f"Batch error: {e}")
        from parser import parse_resume_with_gpt
        return [parse_resume_with_gpt(text) for text in resume_texts]
