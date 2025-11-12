import os
import json
from openai import OpenAI
import pdfplumber

client = OpenAI(api_key='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')

def extract_pdf_text(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

def make_json(pdf_path):
    text = extract_pdf_text(pdf_path)
    if len(text) < 50:
        return None
    
    msg = "Extract this resume data as JSON only:\n\n" + text[:4000]
    msg += '\n\nFormat: {"name":"Name","email":"email@test.com","phone":"123-456-7890",'
    msg += '"experience":[{"job_title":"Title","company":"Co","start_date":"2020",'
    msg += '"end_date":"2022","responsibilities":["task1"]}],'
    msg += '"education":[{"degree":"BS","institution":"School","start_date":"2016","end_date":"2020"}],'
    msg += '"skills":["skill1","skill2"]}'
    
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role":"user","content":msg}],
        temperature=0
    )
    
    result = resp.choices[0].message.content.strip()
    
    # Clean markdown
    if "json" in result:
        result = result.split("json")[1]
    if result.startswith("`"):
        result = result.strip("`").strip()
    
    return json.loads(result)

# Main
test_dir = 'test_data'
pdfs = sorted([f for f in os.listdir(test_dir) if f.endswith('.pdf')])

print(f"Found {len(pdfs)} PDFs\n")

done = 0
skip = 0

for pdf in pdfs:
    base = pdf.replace('.pdf', '')
    out = f"{base}_correct.json"
    
    if os.path.exists(f"{test_dir}/{out}"):
        print(f"Skip {pdf}")
        skip += 1
        continue
    
    print(f"Processing {pdf}...")
    try:
        data = make_json(f"{test_dir}/{pdf}")
        if data:
            with open(f"{test_dir}/{out}", 'w') as f:
                json.dump(data, f, indent=2)
            print(f"  ✅ Done")
            done += 1
        else:
            print(f"  ❌ Failed")
    except Exception as e:
        print(f"  ❌ Error: {e}")

print(f"\n✅ Created: {done}")
print(f"⏭ Skipped: {skip}")
