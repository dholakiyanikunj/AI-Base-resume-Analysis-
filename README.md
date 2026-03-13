# 💼 AI Resume Analyzer

An AI-powered web application that analyzes resumes, offers enhancement suggestions, rewrites bullet points, scores structure & formatting, and even estimates future career paths — all powered by **Google Gemini** and **NLP**.

> 🚀 Built with Streamlit, spaCy, Gemini API, and love.

---

## 🔍 Features

| Feature                             | Description                                                                 |
|------------------------------------|-----------------------------------------------------------------------------|
| 📊 **Resume Scorecard**            | Structure, clarity, relevance & formatting scores with visual progress bars |
| 👁 **Recruiter Eye-View**          | First-impression summary from a recruiter's perspective                     |
| ✍️ **Bullet Point Rewriter**       | Rewrites weak bullet points into sharp, impactful statements                |
| 🧠 **Soft Skills & Tone Analyzer** | Extracts tone (e.g., aggressive/balanced) & soft skills from the resume     |
| 📈 **Career Progression Estimator**| Suggests next roles and 1–3 year roadmap                                    |
| 📚 **NLP-based Skill Extractor**   | Uses spaCy to extract names, skills, designations, and degrees              |
| 📄 **Download Report (PDF)**       | Download a personalized resume feedback report as a professional PDF        |

---

## 🛠 Tech Stack

- **Frontend**: Streamlit
- **LLM Integration**: Google Gemini 1.5 Pro API
- **NLP Engine**: spaCy (`en_core_web_sm`)
- **PDF Export**: FPDF
- **Resume Parsing**: PyMuPDF
- **Environment Management**: `.env` (local) and Streamlit Secrets (cloud)

---

## 🚀 Getting Started

### 🧪 Local Setup

```bash
git clone https://github.com/jaychafale/Resume_analyzer.git
cd Resume_analyzer
pip install -r requirements.txt
```

Create a .env file and add your Gemini API key:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

Download the spaCy language model:
```bash
python -m spacy download en_core_web_sm
```

Run the app:
```bash
streamlit run main.py
```
