import google.generativeai as genai
import json
import re

# 🔧 Configure the Gemini model
def configure_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")

# ✅ Main resume feedback generator
def get_resume_feedback(resume_text, job_description, api_key):
    model = configure_gemini(api_key)
    prompt = f"""
You are a professional resume reviewer.
Analyze the resume and suggest improvements in structure, formatting, and alignment with job description.

Resume:
\"\"\"{resume_text}\"\"\"

Job Description:
\"\"\"{job_description or "N/A"}\"\"\"
"""
    return model.generate_content(prompt).text

# 👁 Recruiter eye-view summary
def get_recruiter_view(resume_text, api_key):
    model = configure_gemini(api_key)
    prompt = f"""
Pretend you are a recruiter scanning the resume for 30 seconds.
Give your first impression, strengths, and concerns.

Resume:
\"\"\"{resume_text}\"\"\"
"""
    return model.generate_content(prompt).text

# ✍️ Bullet point rewriter
def rewrite_bullet_point(bullet_point, api_key):
    model = configure_gemini(api_key)
    prompt = f"""
Rewrite this resume bullet point to be more professional, concise, and impactful:

\"{bullet_point}\"
"""
    return model.generate_content(prompt).text

# 🧠 Soft skills and tone analyzer
def analyze_soft_skills_and_tone(resume_text, api_key):
    model = configure_gemini(api_key)
    prompt = f"""
Analyze the tone and soft skills in the following resume.
Classify tone (e.g., Aggressive, Passive, Balanced) and list soft skills demonstrated.

Resume:
\"\"\"{resume_text}\"\"\"
"""
    return model.generate_content(prompt).text

# 📈 Career progression estimator
def estimate_career_progression(resume_text, api_key):
    model = configure_gemini(api_key)
    prompt = f"""
Based on the resume, suggest possible next career roles and a 1–3 year development roadmap.

Resume:
\"\"\"{resume_text}\"\"\"
"""
    return model.generate_content(prompt).text

# 📊 Resume scorecard function (for progress meters)
def get_resume_scorecard(resume_text, job_description, api_key):
    model = configure_gemini(api_key)

    # ✂️ Truncate long inputs to avoid timeout
    trimmed_resume = resume_text[:1000]
    trimmed_jd = (job_description or "N/A")[:800]

    prompt = f"""
Score this resume on the following aspects (out of 10), and give a total score (out of 100):

1. Structure
2. Clarity
3. Relevance to Job Description
4. Formatting

Return as JSON like:
{{
  "Structure": 8,
  "Clarity": 7,
  "Relevance": 6,
  "Formatting": 9,
  "Total": 75
}}

Resume:
\"\"\"{trimmed_resume}\"\"\"

Job Description:
\"\"\"{trimmed_jd}\"\"\"
"""
    try:
        response = model.generate_content(prompt).text
        json_text = re.search(r"\{.*\}", response, re.DOTALL).group()
        return json.loads(json_text)
    except Exception as e:
        print("⚠️ Error from Gemini (scorecard):", e)
        return {
            "Structure": 7,
            "Clarity": 7,
            "Relevance": 7,
            "Formatting": 7,
            "Total": 70,
            "error": str(e)
        }
