# 📄 Resume Review Agent

A beginner-friendly app that compares a resume with a job description and gives an honest score plus an action plan.
It never invents skills: anything not shown in the resume is marked **Unknown / Not Demonstrated**.

**Tech:** Streamlit (UI) • CrewAI (1 Agent, 1 Task, 1 Crew) • Groq (`openai/gpt-oss-120b`) • pypdf

## Files
```
resume-review-agent/
├── app.py                 # the whole app
├── requirements.txt       # libraries to install
├── README.md              # this guide
├── .gitignore             # keeps secrets out of GitHub
└── secrets.toml.example   # example of the secrets format
```

## Run on your computer (optional)
1. Install Python 3.11
2. `pip install -r requirements.txt`
3. Create a folder `.streamlit` and inside it a file `secrets.toml` with:
   ```
   GROQ_API_KEY = "your_key"
   GROQ_MODEL = "openai/gpt-oss-120b"
   ```
4. `streamlit run app.py`

## Deploy on Streamlit Community Cloud
1. Get a free API key from https://console.groq.com/keys
2. Create a GitHub repo and upload `app.py`, `requirements.txt`, `README.md`, `.gitignore`, `secrets.toml.example`
3. Go to https://share.streamlit.io → **Create app** → pick your repo → main file `app.py`
4. Open **Advanced settings** → choose **Python 3.11** → paste your secrets:
   ```
   GROQ_API_KEY = "your_key"
   GROQ_MODEL = "openai/gpt-oss-120b"
   ```
5. Click **Deploy**

## Common problems
| Message | What to do |
|---|---|
| API key missing | Add `GROQ_API_KEY` in Secrets |
| Rate limit | Wait 1 minute, use shorter text |
| No readable text in PDF | PDF is a scan; paste the text instead |
| Model not available | Set `GROQ_MODEL` to an active Groq model |
