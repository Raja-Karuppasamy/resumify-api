# Updated: Nov 19, 2025 - v3.1 (Fixed OpenAI 1.0+ API compatibility)

import os
import re
import json
import base64
import pdfplumber
import docx
import statistics
from io import BytesIO
from openai import OpenAI
from typing import List, Dict, Optional, Tuple

# Try loading OCR libraries
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False
    print("⚠️ OCR dependencies missing (pytesseract/pdf2image/PIL). Vision fallback will still work.")

# -----------------------------
# 🔐 OpenAI Client Setup
# -----------------------------
_client = None

def get_openai_client() -> OpenAI:
    """Get or create OpenAI client instance"""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY is not set")
        _client = OpenAI(api_key=api_key)
    return _client
# -----------------------------
# 🔠 Useful regex patterns
# -----------------------------
EMAIL_REGEX = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
PHONE_REGEX = (
    r'(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?'
    r'\d{3,4}[-.\s]?\d{4}'
)

# Skill database
TECH_SKILLS = [
    "python", "java", "javascript", "c++", "c#", "ruby", "go", "rust", "swift", "kotlin",
    "php", "typescript", "scala", "perl", "r", "matlab", "html", "css", "sql", "nosql",
    "react", "angular", "vue", "node.js", "express", "django", "flask", "spring", "rails",
    ".net", "asp.net", "laravel", "fastapi", "next.js", "nuxt.js", "svelte",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible", "jenkins",
    "git", "github", "gitlab", "bitbucket", "jira", "confluence",
    "mongodb", "postgresql", "mysql", "redis", "elasticsearch", "cassandra", "dynamodb",
    "linux", "unix", "windows", "bash", "powershell", "vim", "emacs",
    "machine learning", "deep learning", "nlp", "computer vision", "data science",
    "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "matplotlib",
    "agile", "scrum", "kanban", "devops", "ci/cd", "microservices", "rest", "graphql",
    "communication", "leadership", "teamwork", "problem solving", "critical thinking",
    "project management", "time management", "presentation", "negotiation",
    "microsoft office", "excel", "powerpoint", "word", "outlook",
    "figma", "sketch", "adobe xd", "photoshop", "illustrator", "indesign",
    "seo", "sem", "google analytics", "salesforce", "hubspot", "tableau", "power bi"
]

SECTION_HEADERS = [
    "experience", "education", "skills", "certifications",
    "projects", "work experience", "employment history",
    "technical skills", "core competencies", "professional experience",
    "work history", "career history", "qualifications"
]

# -----------------------------
# 🔧 Utility: Base64 encode PIL image for Vision API
# -----------------------------
def pil_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"
# ============================================================
# 🧠 Hybrid OCR Engine (Tesseract → GPT-4o Vision fallback)
# ============================================================

def pdfplumber_text_quality(text: str) -> float:
    """
    Quick heuristic to measure whether pdfplumber text is clean enough.
    Returns a ratio of alphabetic characters vs noise.
    """
    if not text:
        return 0
    
    total = len(text)
    alpha = sum(c.isalpha() for c in text)
    ratio = alpha / max(total, 1)
    return ratio

def vision_ocr_page(image: Image.Image) -> str:
    """
    Use GPT-4o Vision to read a single PDF page image.
    """
    try:
        client = get_openai_client()
        encoded_image = pil_to_base64(image)

        prompt = (
            "You are reading a scanned resume page. "
            "Extract ALL visible text exactly as-is. "
            "Do NOT add or rewrite content. "
            "Maintain natural line breaks. "
            "Return ONLY the extracted text."
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": encoded_image}
                        }
                    ]
                }
            ],
            max_tokens=2000,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"❌ GPT-4o Vision OCR failed: {e}")
        return ""

def tesseract_ocr_page(image: Image.Image) -> str:
    """
    Use Tesseract to perform fast OCR on a single page.
    """
    try:
        return pytesseract.image_to_string(
            image,
            config="--psm 6"
        )
    except Exception as e:
        print(f"❌ Tesseract failed: {e}")
        return ""
def extract_text_from_pdf(file_path: str) -> str:
    """
    Hybrid extraction:
      1) Try pdfplumber → If clean, return
      2) Convert pages to images
      3) For each page:
          - Tesseract first
          - If low quality → GPT-4o Vision fallback
      4) Combine cleaned text
    """
    text = ""

    # Step 1: Try pdfplumber textual extraction
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        quality = pdfplumber_text_quality(text)
        if len(text.strip()) >= 200 and quality > 0.45:
            print("📄 Using pdfplumber text (clean enough)")
            return text

        print(f"⚠️ pdfplumber text too noisy (quality={quality:.2f}), switching to OCR...")

    except Exception as e:
        print(f"❌ pdfplumber failed: {e}")

    # Step 2: Convert PDF → images
    if not OCR_AVAILABLE:
        print("❌ OCR libraries missing. Returning pdfplumber text.")
        return text if text else ""

    try:
        images = convert_from_path(file_path)
    except Exception as e:
        print(f"❌ pdf2image failed: {e}")
        return text if text else ""

    # Step 3: Per-page hybrid OCR
    rebuilt_pages = []

    for idx, image in enumerate(images):
        print(f"\n🖼️ Processing page {idx+1}/{len(images)}")

        tesseract_text = tesseract_ocr_page(image)
        t_quality = pdfplumber_text_quality(tesseract_text)

        if len(tesseract_text.strip()) >= 80 and t_quality > 0.25:
            print(f"✔ Tesseract accepted (quality={t_quality:.2f})")
            rebuilt_pages.append(tesseract_text)
            continue

        print(f"⚠️ Tesseract too noisy (quality={t_quality:.2f}), using GPT-4o Vision fallback...")

        vision_text = vision_ocr_page(image)

        if vision_text.strip():
            print("✔ GPT-4o Vision returned clean text")
            rebuilt_pages.append(vision_text)
        else:
            print("❌ Vision OCR failed → using Tesseract fallback text")
            rebuilt_pages.append(tesseract_text)

    combined = "\n\n".join(rebuilt_pages)
    print(f"\n🧾 Final OCR'd text length: {len(combined)} characters")
    return combined
# ============================================================
# 🧼 GPT Cleanup Pass
# ============================================================

def clean_ocr_text_with_gpt(ocr_text: str) -> str:
    """
    Cleans OCR noise but strictly avoids hallucinating content.
    """
    if not ocr_text or not ocr_text.strip():
        return ocr_text

    try:
        client = get_openai_client()

        prompt = f"""
You are cleaning noisy OCR text extracted from a resume.

STRICT RULES:
- DO NOT invent or change any information.
- DO NOT add skills, rewrite sentences, or guess content.
- ONLY fix:
    - broken words
    - missing spaces
    - weird symbols (|, —, ·, °, ®)
    - random line breaks
- Preserve section headers as they appear.
- Keep all original resume data EXACTLY the same.

Return ONLY the cleaned resume text.

OCR Text:
{ocr_text}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=2000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )

        cleaned = response.choices[0].message.content.strip()
        return cleaned or ocr_text

    except Exception as e:
        print(f"❌ GPT cleanup failed: {e}")
        return ocr_text

# ============================================================
# 🧠 GPT Structured Extraction (Resume → JSON)
# ============================================================

def parse_resume_with_gpt(text: str) -> dict:
    """
    Extract ALL resume fields using gpt-4o-mini.
    Ensures clean JSON output with fallbacks.
    """
    if not text or not text.strip():
        return {}

    text_sample = text[:8000]

    prompt = f"""
Extract ALL information from this resume and return ONLY valid JSON.

Resume text:
{text_sample}

RETURN JSON WITH THIS EXACT STRUCTURE:

{{
  "name": "Full Name",
  "email": "email@example.com",
  "phone": "(123) 456-7890",
  "location": "City, State",
  "linkedin": "linkedin.com/in/username",
  "experience": [
    {{
      "job_title": "",
      "company": "",
      "location": "",
      "start_date": "",
      "end_date": "",
      "responsibilities": []
    }}
  ],
  "education": [
    {{
      "degree": "",
      "major": "",
      "institution": "",
      "location": "",
      "start_date": "",
      "end_date": ""
    }}
  ],
  "skills": [],
  "certifications": [],
  "summary": ""
}}

STRICT RULE:
- Return ONLY raw JSON. No markdown, no backticks, no explanations.
"""

    try:
        client = get_openai_client()
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.choices[0].message.content.strip()
        content = re.sub(r"``````", "", content).strip()

        try:
            return json.loads(content)
        except:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(content[start:end+1])
            print("❌ JSON recovery failed")
            return {}

    except Exception as e:
        print(f"❌ GPT-4o-mini extraction failed: {e}")
        return {}
# ============================================================
# 📬 EMAIL + PHONE EXTRACTION
# ============================================================

def calculate_confidence(extracted_value: str, method: str, context: str = "") -> float:
    """Basic confidence score"""
    if not extracted_value:
        return 0.0

    weights = {
        "regex": 0.95,
        "section": 0.85,
        "keyword": 0.75,
        "fallback": 0.5
    }

    return weights.get(method, 0.5)

def extract_email(text: str) -> Tuple[Optional[str], float]:
    matches = re.findall(EMAIL_REGEX, text)
    if matches:
        email = matches[0]
        return email, calculate_confidence(email, "regex")
    return None, 0.0

def extract_phone(text: str) -> Tuple[Optional[str], float]:
    matches = re.findall(PHONE_REGEX, text)
    if matches:
        raw = matches[0]
        phone = "".join(raw) if isinstance(raw, tuple) else raw
        return phone.strip(), calculate_confidence(phone, "regex")
    return None, 0.0

# ============================================================
# 🧍 NAME EXTRACTION
# ============================================================

def extract_name(text: str) -> Tuple[Optional[str], float]:
    sample = text[:500]

    try:
        client = get_openai_client()
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=50,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Extract ONLY the candidate's full name from this resume text. "
                        "Return just the name, no explanations:\n\n" + sample
                    ),
                }
            ],
        )

        name = response.choices[0].message.content.strip()
        name = name.replace("*", "").replace("**", "")
        name = name.strip('"').strip("'")

        if 2 <= len(name) <= 50 and "@" not in name and not any(c.isdigit() for c in name):
            return name, 0.95

        return extract_name_fallback(text)

    except:
        return extract_name_fallback(text)

def extract_name_fallback(text: str) -> Tuple[Optional[str], float]:
    lines = [l.strip() for l in text.split("\n")[:10] if len(l.strip()) > 2]

    for line in lines:
        if any(x in line.lower() for x in ["resume", "cv", "@", "phone", "email"]):
            continue
        words = line.split()
        if 2 <= len(words) <= 4 and all(w.isalpha() for w in words):
            return line, 0.7

    return None, 0.0
# ============================================================
# 📑 SECTION SPLITTING
# ============================================================

def split_sections(text: str) -> Dict[str, str]:
    sections = {}
    current = None
    buffer = []

    lines = text.split("\n")

    for line in lines:
        low = line.lower().strip()

        if any(low.startswith(h) for h in SECTION_HEADERS):
            if current:
                sections[current] = "\n".join(buffer)
            current = low
            buffer = []
        else:
            if current:
                buffer.append(line)

    if current and buffer:
        sections[current] = "\n".join(buffer)

    return sections

# ============================================================
# 💼 EXPERIENCE EXTRACTION
# ============================================================

def extract_experience(text: str) -> List[Dict]:
    """Fallback experience extractor"""
    if not text:
        return []

    date_pattern = r"(\d{4})\s*[-–—]\s*(\d{4}|present|current)"
    matches = list(re.finditer(date_pattern, text, re.IGNORECASE))

    if not matches:
        return []

    experiences = []

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]

        lines = [l.strip() for l in block.split("\n") if l.strip()]

        if not lines:
            continue

        job = lines[0] if len(lines) > 0 else ""
        company = lines[1] if len(lines) > 1 else ""

        experiences.append({
            "job_title": job,
            "company": company,
            "location": "",
            "start_date": m.group(1),
            "end_date": m.group(2),
            "responsibilities": lines[2:]
        })

    return experiences

# ============================================================
# 🎓 EDUCATION EXTRACTION
# ============================================================

def extract_education(text: str) -> List[Dict]:
    if not text:
        return []

    blocks = re.split(r'\n\s*\n', text)
    results = []

    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if not lines:
            continue

        degree = lines[0]
        inst = lines[1] if len(lines) > 1 else ""
        dates = re.findall(r"\b\d{4}\b", block)

        start = dates[0] if len(dates) > 0 else ""
        end = dates[1] if len(dates) > 1 else start

        results.append({
            "degree": degree,
            "major": "",
            "institution": inst,
            "location": "",
            "start_date": start,
            "end_date": end,
        })

    return results

# ============================================================
# 🛠 SKILLS EXTRACTION
# ============================================================

def extract_skills_advanced(text: str, sections: dict) -> List[Dict]:
    found = []

    combined = (
        sections.get("skills", "") +
        sections.get("technical skills", "") +
        text
    ).lower()

    for skill in TECH_SKILLS:
        if skill.lower() in combined:
            found.append({
                "skill": skill,
                "confidence": calculate_confidence(skill, "keyword")
            })

    return found

# ============================================================
# 🎯 OVERALL CONFIDENCE SCORE
# ============================================================

def calculate_overall_confidence(
    name_conf,
    email_conf,
    phone_conf,
    exp_conf,
    edu_conf,
    skill_conf
) -> float:
    scores = [
        name_conf, email_conf, phone_conf,
        *exp_conf, *edu_conf, *skill_conf
    ]

    scores = [s for s in scores if s > 0]
    if not scores:
        return 0.0

    return round(statistics.mean(scores), 2)
