from pydantic import BaseModel
from typing import Optional, List

class ExperienceItem(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None
    confidence: Optional[float] = None

class EducationItem(BaseModel):
    degree: Optional[str] = None
    institution: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    confidence: Optional[float] = None

class SkillItem(BaseModel):
    skill: str
    confidence: float

class ResumeParsed(BaseModel):
    name: Optional[str] = None
    name_confidence: Optional[float] = None
    email: Optional[str] = None
    email_confidence: Optional[float] = None
    phone: Optional[str] = None
    phone_confidence: Optional[float] = None
    experience: List[ExperienceItem] = []
    education: List[EducationItem] = []
    skills: List[SkillItem] = []
    overall_confidence: Optional[float] = None

# New schemas for batch processing
class BatchParseRequest(BaseModel):
    files: List[str]  # List of file URLs or base64 encoded files

class BatchParseResponse(BaseModel):
    results: List[ResumeParsed]
    processed_count: int
    failed_count: int
    processing_time: float
