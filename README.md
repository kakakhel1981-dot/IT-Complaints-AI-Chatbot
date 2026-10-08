# AI Customer Complaint Handling Assistant

A Streamlit + Groq knowledge-based chatbot that helps new team members troubleshoot customer complaints using approved SOPs, FAQs, and troubleshooting documents.

## AI Process Flow

Upload documents
-> Extract text
-> Split into chunks
-> Retrieve relevant sections using TF-IDF similarity
-> Send complaint + retrieved evidence to Groq
-> Generate step-by-step troubleshooting guidance
-> Show source documents and escalation guidance

## Supported files

- PDF
- DOCX
- TXT

## Project files

- `app.py` - Streamlit application
- `requirements.txt` - Python dependencies
- `.gitignore` - protects local secrets
- `sample_knowledge_base.txt` - sample test knowledge base

## 1. Test in Google Colab

Upload `app.py`, `requirements.txt`, and `sample_knowledge_base.txt` to Colab.

Install packages:

```python
!pip install -r requirements.txt
```

Set your Groq API key:

```python
import os
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"
```

Run Streamlit in Colab using your preferred tunnel method, or test the Python functions directly.

## 2. Run locally

Install Python 3.12.

```bash
pip install -r requirements.txt
```

Set the API key as an environment variable, then run:

```bash
streamlit run app.py
```

## 3. Streamlit secrets

For local Streamlit development, create:

`.streamlit/secrets.toml`

with:

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```

Do NOT commit this file to GitHub.

## 4. GitHub

Create a new repository and upload:

- app.py
- requirements.txt
- .gitignore
- sample_knowledge_base.txt

Do not upload:

- `.streamlit/secrets.toml`
- API keys
- `.env`

## 5. Streamlit Cloud

Deploy the GitHub repository through Streamlit Community Cloud.

Set the application secret:

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```

Optional model override:

```toml
GROQ_MODEL = "llama-3.3-70b-versatile"
```

The app reads the key using Streamlit secrets.

## Important

This first version is designed for internal troubleshooting support. It should not execute database commands or make irreversible changes automatically. Team members should verify recommendations against the latest approved SOP before taking action.
