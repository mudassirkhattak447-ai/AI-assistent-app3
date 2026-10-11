# 📄 ATS Resume Checker

Upload a resume (PDF, DOCX or TXT) and get:

- an **ATS score out of 100**, calculated from five weighted categories
- **specific, prioritised improvements** (High / Medium / Low)
- **missing keywords** and **before/after bullet rewrites**
- an optional **job-description match score** if you paste a job posting
- **automated checks** (contact details, standard sections, bullets, metrics, length)
- a downloadable Markdown report

Built with [Streamlit](https://streamlit.io) and Google's Gemini Flash model.

## How the score works

1. The app extracts text from your resume (what an ATS would "see").
2. It runs deterministic checks (email, phone, sections, bullets, metrics, length).
3. Gemini scores five categories from 0-100: keywords, content & impact, formatting,
   structure, readability (plus job match if a job description is provided).
4. The overall score is a **weighted average computed in code**, so it is consistent
   and cannot be directly set by text inside the resume.

> The score is an estimate. Real ATS products differ, so use it as guidance, not a guarantee.

## Run locally

```bash
git clone https://github.com/<your-username>/ats-resume-checker.git
cd ats-resume-checker

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Get a free API key at <https://aistudio.google.com/apikey>, then provide it in **one** of these ways:

**Option A - secrets file (recommended)**

Create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-key-here"
# GEMINI_MODEL = "gemini-2.5-flash"   # optional override
```

**Option B - environment variable**

```bash
export GEMINI_API_KEY="your-key-here"     # Windows PowerShell: $env:GEMINI_API_KEY="your-key-here"
```

**Option C** - paste the key into the app's sidebar.

Start the app:

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (never commit your API key).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app** → choose your repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and paste:

   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**. Later `git push` commits redeploy the app automatically.

## Configuration

| Setting | Where | Default |
|---|---|---|
| `GEMINI_API_KEY` | secrets, env var, or sidebar | - |
| `GEMINI_MODEL` | secrets, env var, or sidebar | `gemini-flash-latest` |

## Limitations

- Scanned / image-only PDFs have no extractable text and are rejected (an ATS could not read them either).
- Only the first ~15,000 characters of a resume are analysed.
- Uploaded resumes are sent to the Gemini API. Don't use this with data you aren't allowed to share.
  The free Gemini API tier may use submitted content to improve Google products - check Google's current terms.

## Project structure

```
app.py             # Streamlit app + analysis logic
requirements.txt   # Python dependencies
README.md
.gitignore         # keeps secrets out of Git
```
