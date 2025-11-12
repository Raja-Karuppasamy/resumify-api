import json
import os
import sys
from parser import (
    extract_text_from_pdf,
    extract_email,
    extract_phone,
    extract_name,
    split_sections,
    extract_experience,
    extract_education,
    extract_skills_advanced
)

def parse_resume_wrapper(pdf_path: str) -> dict:
    """Wrapper function - uses GPT for everything"""
    from parser import extract_text_from_pdf, parse_resume_with_gpt
    
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Use GPT to parse everything
    result = parse_resume_with_gpt(text)
    
    if not result:
        # Fallback to empty structure if GPT fails
        return {
            'name': None,
            'email': None,
            'phone': None,
            'experience': [],
            'education': [],
            'skills': []
        }
    
    return result

def compare_field(correct_val, parsed_val, field_name):
    """Compare a single field and return score"""
    if correct_val is None and parsed_val is None:
        return 1.0
    if correct_val is None or parsed_val is None:
        print(f"    ❌ {field_name}: One is missing")
        return 0.0
    
    correct_str = str(correct_val).lower().strip()
    parsed_str = str(parsed_val).lower().strip()
    
    if correct_str == parsed_str:
        print(f"    ✅ {field_name}: '{parsed_val}'")
        return 1.0
    else:
        print(f"    ❌ {field_name}: Expected '{correct_val}', got '{parsed_val}'")
        return 0.0


def compare_resumes(correct_data, parsed_data):
    """Compare parsed resume against ground truth"""
    total_score = 0
    max_score = 0
    
    print("  Basic Fields:")
    for field in ['name', 'email', 'phone']:
        max_score += 1
        score = compare_field(
            correct_data.get(field),
            parsed_data.get(field),
            field
        )
        total_score += score
    
    print("\n  Experience:")
    correct_exp_count = len(correct_data.get('experience', []))
    parsed_exp_count = len(parsed_data.get('experience', []))
    max_score += correct_exp_count
    
    matched_jobs = min(correct_exp_count, parsed_exp_count)
    total_score += matched_jobs
    print(f"    Found {parsed_exp_count}/{correct_exp_count} job entries")
    
    print("\n  Education:")
    correct_edu_count = len(correct_data.get('education', []))
    parsed_edu_count = len(parsed_data.get('education', []))
    max_score += correct_edu_count
    
    matched_edu = min(correct_edu_count, parsed_edu_count)
    total_score += matched_edu
    print(f"    Found {parsed_edu_count}/{correct_edu_count} education entries")
    
    print("\n  Skills:")
    correct_skills = set(s.lower() for s in correct_data.get('skills', []))
    parsed_skills = set(s.lower() for s in parsed_data.get('skills', []))
    
    if correct_skills:
        matched_skills = len(correct_skills & parsed_skills)
        max_score += len(correct_skills)
        total_score += matched_skills
        print(f"    Found {matched_skills}/{len(correct_skills)} skills")
    else:
        print(f"    No skills to compare")
    
    if max_score == 0:
        return 0.0
    
    accuracy = (total_score / max_score) * 100
    return accuracy


# Main testing loop
print("=" * 60)
print("RESUME PARSER ACCURACY TEST")
print("=" * 60)

test_dir = "test_data"
total_accuracy = 0
num_resumes = 33  # Change this to test more resumes
successful_tests = 0

for i in range(1, num_resumes + 1):
    print(f"\n{'=' * 60}")
    print(f"TESTING RESUME {i:02d}")
    print(f"{'=' * 60}")
    
    resume_file = os.path.join(test_dir, f"resume_{i:02d}.pdf")
    correct_file = os.path.join(test_dir, f"resume_{i:02d}_correct.json")
    parsed_file = os.path.join(test_dir, f"resume_{i:02d}_parsed.json")
    
    if not os.path.exists(resume_file):
        print(f"❌ Resume file not found: {resume_file}")
        continue
    
    if not os.path.exists(correct_file):
        print(f"❌ Ground truth not found: {correct_file}")
        continue
    
    try:
        print(f"\n📄 Parsing {resume_file}...")
        parsed_data = parse_resume_wrapper(resume_file)
        
        with open(parsed_file, 'w') as f:
            json.dump(parsed_data, f, indent=2)
        print(f"✅ Saved parsed output to {parsed_file}")
        
        with open(correct_file) as f:
            correct_data = json.load(f)
        
        print(f"\n📊 Comparing Results:")
        accuracy = compare_resumes(correct_data, parsed_data)
        
        print(f"\n{'=' * 60}")
        print(f"RESUME {i:02d} ACCURACY: {accuracy:.1f}%")
        print(f"{'=' * 60}")
        
        total_accuracy += accuracy
        successful_tests += 1
        
    except Exception as e:
        print(f"❌ Error processing resume {i:02d}: {str(e)}")
        import traceback
        traceback.print_exc()

print(f"\n\n{'=' * 60}")
print(f"FINAL RESULTS")
print(f"{'=' * 60}")
print(f"Resumes tested: {successful_tests}/{num_resumes}")
if successful_tests > 0:
    overall_accuracy = total_accuracy / successful_tests
    print(f"OVERALL ACCURACY: {overall_accuracy:.1f}%")
    print(f"{'=' * 60}")
    
    if overall_accuracy >= 90:
        print("🎉 EXCELLENT! Your parser is performing very well!")
    elif overall_accuracy >= 75:
        print("👍 GOOD! Some improvements needed but solid baseline.")
    elif overall_accuracy >= 60:
        print("📈 FAIR. Significant improvements needed.")
    else:
        print("⚠️  NEEDS WORK. Focus on fixing major extraction issues.")
else:
    print("❌ No resumes were successfully tested.")
