import re
import json
import pdfplumber
import docx
from typing import Optional, List, Dict, Tuple
import statistics
from openai import OpenAI
import os
# Make OCR dependencies optional
try:
    #import pytesseract
    #from pdf2image import convert_from_path
    #from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("OCR libraries not available. Scanned PDF support disabled.")


# OpenAI client singleton
_openai_client = None

# Regex patterns for extraction
EMAIL_REGEX = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
PHONE_REGEX = r'(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}|\(\d{3}\)\s?\d{3}-\d{4}|\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'

# Comprehensive skills database
TECH_SKILLS = [
    # Programming Languages
    'python', 'java', 'javascript', 'typescript', 'go', 'rust', 'c++', 'c#', 'php', 'ruby',
    'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql', 'html', 'css', 'sass', 'less',
    
    # Frameworks & Libraries  
    'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring', 'laravel',
    'rails', 'asp.net', 'jquery', 'bootstrap', 'tailwind', 'next.js', 'nuxt.js',
    
    # Databases
    'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch', 'cassandra', 'dynamodb',
    'oracle', 'sqlite', 'neo4j', 'firebase',
    
    # Cloud & DevOps
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'gitlab', 'github', 'terraform',
    'ansible', 'nginx', 'apache', 'linux', 'ubuntu', 'centos', 'bash', 'powershell',
    
    # Data & AI
    'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn',
    'data analysis', 'data science', 'big data', 'hadoop', 'spark', 'tableau', 'power bi',
    
    # Soft Skills
    'leadership', 'project management', 'agile', 'scrum', 'communication', 'problem solving',
    'team management', 'strategic planning', 'analytical thinking',
    
    # Healthcare/Nursing
    'mckesson', 'ge corometrics', 'baxter', 'masimo', 'carescape',
    'abbott', 'siemens', 'patient care', 'ehr', 'phlebotomy',
    'blood analysis', 'fetal monitoring', 'neonatal', 'obstetrics', 'phlebotomist',
    
    # Design & Creative
    'adobe', 'photoshop', 'illustrator', 'sketch', 'figma',
    'visual design', 'brand identity', 'motion graphics', 'after effects', 'premiere', 'ux', 'ui',
    
    # Data & Analytics  
    'google analytics', 'bigquery', 'looker', 'stata', 'powerbi',
    
    # Education
    'classroom management', 'lesson planning', 'curriculum development',
    'student assessment', 'google classroom', 'canvas', 'blackboard',
    
    # Restaurant/Hospitality  
    'pos systems', 'inventory management', 'food safety', 'haccp',
    'menu planning', 'cost control', 'staff training',
    
    # General Business
    'microsoft office', 'excel', 'powerpoint', 'outlook', 'visio',
    'customer service', 'team leadership', 'project coordination',
    'budgeting', 'scheduling', 'communication', 'presentation',
    'time management', 'problem solving',
    
    # Miscellaneous
    'canva', 'pet care', 'water flow', 'hydroponics', 'pet footwear', 
    'secchi disks', 'sensory evaluations', 'water flow meters', 'coolclimate'
]

SECTION_HEADERS = [
    'experience', 'work experience', 'professional experience', 'employment history', 'work history',
    'education', 'academic background', 'qualifications', 'educational background', 'academic qualifications',
    'skills', 'technical skills', 'core competencies', 'key skills', 'areas of expertise',
    'certifications', 'certificates', 'licenses',
    'projects', 'key projects'
]


def get_openai_client():
    """Get or create OpenAI client"""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv('OPENAI_API_KEY', 'sk-proj-YOUR-KEY-HERE')  # Replace with your actual key
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def parse_resume_with_gpt(text: str) -> dict:
    """Use GPT-4o-mini to extract ALL resume data with high accuracy"""
    
    # Truncate text to fit in context window (8000 chars ≈ full resume)
    sample = text[:8000]
    
    prompt = """Extract ALL information from this resume and return ONLY valid JSON (no markdown, no explanations).

Resume text:
""" + sample + """

Return JSON with this EXACT structure:
{
  "name": "Full Name",
  "email": "email@example.com",
  "phone": "(123) 456-7890",
  "location": "City, State",
  "experience": [
    {
      "job_title": "Job Title",
      "company": "Company Name",
      "location": "City, State",
      "start_date": "Month Year",
      "end_date": "Month Year or Present",
      "responsibilities": ["responsibility 1", "responsibility 2"]
    }
  ],
  "education": [
    {
      "degree": "Degree Name",
      "major": "Field of Study",
      "institution": "University Name",
      "location": "City, State",
      "start_date": "Year",
      "end_date": "Year"
    }
  ],
  "skills": ["skill1", "skill2", "skill3"],
  "certifications": ["cert1", "cert2"],
  "summary": "Brief professional summary"
}

RULES:
- Extract ALL work experience entries (don't skip any jobs)
- Extract ALL education entries
- Extract ALL skills mentioned (technical and soft skills)
- If a field is missing, use null or empty array []
- Return ONLY the JSON object, nothing else
- Do not use markdown code blocks"""

    try:
        client = get_openai_client()
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2000
        )
        
        json_str = response.choices[0].message.content.strip()

            # Clean any markdown formatting
        json_str = re.sub(r'`{3}(?:json)?\s*', '', json_str)  # Remove ``````json
        json_str = json_str.strip()

            # Parse JSON
        data = json.loads(json_str)

        return data


        
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Response: {json_str[:200] if 'json_str' in locals() else 'No response'}")
        return None
    except Exception as e:
        print(f"GPT parsing error: {e}")
        return None


def calculate_confidence(extracted_value: str, extraction_method: str, context: str = "") -> float:
    """Calculate confidence score for extracted data"""
    if not extracted_value or not extracted_value.strip():
        return 0.0
    
    confidence_scores = {
        'regex_match': 0.95,
        'section_match': 0.85,
        'keyword_match': 0.75,
        'heuristic': 0.65,
        'fallback': 0.35
    }
    
    base_score = confidence_scores.get(extraction_method, 0.5)
    
    if extraction_method == 'regex_match':
        if '@' in extracted_value and '.' in extracted_value:
            return min(base_score + 0.05, 1.0)
        elif re.match(r'^\+?[\d\s\-$$$$]+$', extracted_value):
            return min(base_score + 0.05, 1.0)
    
    if context and any(header in context.lower() for header in SECTION_HEADERS):
        base_score = min(base_score + 0.1, 1.0)
    
    return round(base_score, 2)


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file"""
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_email(text: str) -> Tuple[Optional[str], float]:
    """Extract first email address found with confidence"""
    matches = re.findall(EMAIL_REGEX, text)
    if matches:
        email = matches
        confidence = calculate_confidence(email, 'regex_match')
        return email, confidence
    return None, 0.0


def extract_phone(text: str) -> Tuple[Optional[str], float]:
    """Extract first phone number found with confidence"""
    matches = re.findall(PHONE_REGEX, text)
    if matches:
        phone = matches if isinstance(matches, str) else ''.join(matches)
        confidence = calculate_confidence(phone, 'regex_match')
        return phone.strip(), confidence
    return None, 0.0


def extract_name(text: str) -> Tuple[Optional[str], float]:
    """Extract name using GPT-4o-mini for high accuracy"""
    sample = text[:600]
    
    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Extract ONLY the person's full name from this resume. Return just the name, nothing else. Do not include job titles, locations, or contact info:\n\n{sample}"
            }],
            temperature=0,
            max_tokens=30
        )
        
        name = response.choices.message.content.strip()
        name = name.replace('**', '').replace('*', '')
        name = name.strip('"').strip("'")
        
        if 2 <= len(name) <= 50 and '@' not in name and not any(c.isdigit() for c in name[:3]):
            return name, 0.95
        else:
            return extract_name_fallback(text)
            
    except Exception as e:
        print(f"GPT name extraction failed: {e}, using fallback")
        return extract_name_fallback(text)


def extract_name_fallback(text: str) -> Tuple[Optional[str], float]:
    """Fallback regex-based name extraction"""
    lines = [l.strip() for l in text.split('\n')[:15] if l.strip() and len(l.strip()) > 2]
    
    for line in lines:
        if any(word in line.lower() for word in ['resume', 'cv', '@', 'phone', 'email', 'http']):
            continue
        
        words = line.split()
        if 2 <= len(words) <= 4:
            if all(w.isupper() if w and w.isalpha() else False for w in words):
                return line, 0.7
    
    return None, 0.0


def split_sections(text: str) -> dict:
    """Split resume text into sections based on common headers"""
    lines = text.split('\n')
    sections = {}
    current_section = None
    current_content = []
    
    for line in lines:
        line_lower = line.lower().strip()
        
        is_header = False
        for header in SECTION_HEADERS:
            if (line_lower == header or 
                line_lower.startswith(header) or
                (header in line_lower and len(line.strip()) < 50)):
                is_header = True
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                current_section = header
                current_content = []
                break
        
        if not is_header and current_section:
            current_content.append(line)
    
    if current_section and current_content:
        sections[current_section] = '\n'.join(current_content)
    
    return sections


def extract_experience(text: str) -> List[Dict]:
    """Extract work experience with date-based separation"""
    if not text or len(text.strip()) < 20:
        return []
    
    experiences = []
    date_pattern = r'(\d{4}|\w{3,9}\s+\d{4})\s*[-–—]\s*(\d{4}|\w{3,9}\s+\d{4}|Present|present|Current|current)'
    matches = list(re.finditer(date_pattern, text, re.IGNORECASE))
    
    if not matches:
        return [{
            'title': 'Experience',
            'company': '',
            'start_date': '',
            'end_date': '',
            'description': text.strip()[:500]
        }]
    
    for i, match in enumerate(matches):
        search_start = max(0, match.start() - 150)
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        
        job_block = text[search_start:block_end]
        lines = [l.strip() for l in job_block.split('\n') if l.strip()]
        
        if len(lines) < 2:
            continue
        
        job_title = None
        company = None
        
        for line in lines[:5]:
            if match.group() in line or len(line) < 3:
                continue
            if not job_title:
                job_title = line
            elif not company:
                company = line
                break
        
        desc_start = match.end()
        description = text[desc_start:block_end].strip()
        description = ' '.join(description.split()[:50])
        
        experiences.append({
            'title': job_title or 'Position',
            'company': company or '',
            'start_date': match.group(1),
            'end_date': match.group(2),
            'description': description
        })
    
    return experiences


def extract_education(text: str) -> List[Dict]:
    """Extract structured education data"""
    if not text.strip():
        return []
    
    education = []
    edu_blocks = re.split(r'\n\s*\n', text)
    date_pattern = r'(\d{4})[^\n]*?(\d{4}|Present|present|Current|current)'
    
    for block in edu_blocks:
        if len(block.strip()) < 5:
            continue
            
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if not lines:
            continue
        
        degree = lines
        institution = ""
        start_date = ""
        end_date = ""
        
        for line in lines[1:]:
            date_match = re.search(date_pattern, line, re.IGNORECASE)
            if date_match:
                start_date = date_match.group(1)
                end_date = date_match.group(2)
                inst_line = re.sub(date_pattern, '', line, flags=re.IGNORECASE).strip(' ,-')
                if inst_line and not institution:
                    institution = inst_line
            elif not institution:
                institution = line
        
        confidence_factors = []
        if degree: confidence_factors.append(0.9)
        if institution: confidence_factors.append(0.8)
        if start_date: confidence_factors.append(0.7)
        
        overall_confidence = statistics.mean(confidence_factors) if confidence_factors else 0.3
        
        education.append({
            'degree': degree,
            'institution': institution,
            'start_date': start_date,
            'end_date': end_date,
            'confidence': round(overall_confidence, 2)
        })
    
    return education


def extract_skills_advanced(text: str, sections: dict) -> List[Dict]:
    """Extract skills with confidence scoring"""
    skills_with_confidence = []
    skills_text = sections.get('skills', '') + sections.get('technical skills', '') + sections.get('core competencies', '')
    
    for skill in TECH_SKILLS:
        if skill.lower() in skills_text.lower():
            skills_with_confidence.append({
                'skill': skill.title(),
                'confidence': calculate_confidence(skill, 'section_match', 'skills')
            })
    
    full_text_lower = text.lower()
    existing_skills = [s['skill'].lower() for s in skills_with_confidence]
    
    for skill in TECH_SKILLS:
        if skill.lower() in full_text_lower and skill.lower() not in existing_skills:
            skills_with_confidence.append({
                'skill': skill.title(), 
                'confidence': calculate_confidence(skill, 'keyword_match')
            })
    
    unique_skills = {}
    for skill_data in skills_with_confidence:
        skill_name = skill_data['skill']
        if skill_name not in unique_skills or skill_data['confidence'] > unique_skills[skill_name]['confidence']:
            unique_skills[skill_name] = skill_data
    
    return list(unique_skills.values())


def calculate_overall_confidence(name_conf: float, email_conf: float, phone_conf: float, 
                               exp_confidences: List[float], edu_confidences: List[float], 
                               skill_confidences: List[float]) -> float:
    """Calculate overall resume parsing confidence"""
    all_confidences = [name_conf, email_conf, phone_conf]
    all_confidences.extend(exp_confidences)
    all_confidences.extend(edu_confidences)
    all_confidences.extend(skill_confidences)
    
    valid_confidences = [c for c in all_confidences if c > 0]
    
    if not valid_confidences:
        return 0.0
    
    return round(statistics.mean(valid_confidences), 2)


def extract_text_from_image(image_path: str) -> str:
    """Extract text from a single image file using OCR"""
    if not OCR_AVAILABLE:
        return "OCR not available"
    
    img = Image.open(image_path)
    return pytesseract.image_to_string(img)


def extract_text_from_scanned_pdf(pdf_path: str) -> str:
    """Extract text from scanned PDF using OCR"""
    if not OCR_AVAILABLE:
        return "OCR not available"
    
    text = ""
    images = convert_from_path(pdf_path)
    for img in images:
        ocr_result = pytesseract.image_to_string(img)
        text += ocr_result + "\n"
    return text


def is_scanned_pdf(pdf_path: str) -> bool:
    """Check if PDF is scanned (image-based)"""
    with pdfplumber.open(pdf_path) as pdf:
        first_page = pdf.pages
        page_text = first_page.extract_text()
        if not page_text or len(page_text.strip()) < 30:
            return True
    return False
