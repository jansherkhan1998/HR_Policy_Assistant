# 📘 HR Policy Assistant

A beginner-friendly Retrieval-Augmented Generation (RAG) application that allows HR users to upload an HR Policy PDF and ask questions about its contents.

The application extracts text from the PDF, divides the document into chunks, creates embeddings using Sentence Transformers, stores the embeddings in FAISS, retrieves the most relevant policy sections, and sends only the retrieved context to Groq's `openai/gpt-oss-20b` model.

---

## 🚀 Features

* Upload HR Policy PDF
* Extract PDF text using PyMuPDF
* Clean extracted text
* Split documents into manageable chunks
* Preserve PDF page numbers
* Generate embeddings using Sentence Transformers
* Use `all-MiniLM-L6-v2` embeddings
* Store embeddings in FAISS
* Perform similarity search
* Select 3–8 retrieved chunks
* Ask multiple questions
* Streamlit chat interface
* Maintain chat history during the session
* Generate answers using Groq
* Use `openai/gpt-oss-20b`
* Display retrieved policy sections
* Display source page numbers
* Secure API key using Streamlit Secrets

---

# 🧠 RAG Architecture

```text
HR Policy PDF
      ↓
PyMuPDF
      ↓
Text Extraction
      ↓
Text Cleaning
      ↓
Text Chunks
      ↓
Sentence Transformers
all-MiniLM-L6-v2
      ↓
FAISS
      ↓
Similarity Search
      ↓
Retrieved Policy Context
      ↓
Groq
      ↓
openai/gpt-oss-20b
      ↓
HR Policy Answer
```

---

# 🛠️ Technology Stack

| Technology                | Purpose                  |
| ------------------------- | ------------------------ |
| Python                    | Application programming  |
| Streamlit                 | Web interface            |
| PyMuPDF                   | PDF text extraction      |
| Sentence Transformers     | Text embeddings          |
| all-MiniLM-L6-v2          | Embedding model          |
| FAISS                     | Vector similarity search |
| NumPy                     | Numerical operations     |
| Groq                      | LLM inference            |
| openai/gpt-oss-20b        | LLM                      |
| GitHub                    | Source-code repository   |
| Streamlit Community Cloud | Deployment               |

---

# 📂 Project Structure

```text
hr-policy-assistant/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🔄 How the Application Works

## 1. Upload PDF

The user uploads an HR Policy PDF through the Streamlit interface.

## 2. Extract Text

PyMuPDF reads every page of the PDF and extracts available text.

## 3. Clean Text

Extra spaces and unnecessary line breaks are removed.

## 4. Create Chunks

The document is divided into chunks of approximately 700 words with approximately 100 words of overlap.

## 5. Create Embeddings

Sentence Transformers converts each text chunk into a numerical vector using:

```text
all-MiniLM-L6-v2
```

## 6. Create FAISS Index

The embeddings are stored in a FAISS similarity-search index.

## 7. Ask a Question

The user asks an HR policy question.

## 8. Embed the Question

The question is converted into an embedding using the same Sentence Transformer model.

## 9. Retrieve Relevant Chunks

FAISS searches for the most similar policy chunks.

The user can select between 3 and 8 retrieved chunks.

## 10. Generate Answer

The retrieved policy context and question are sent to:

```text
openai/gpt-oss-20b
```

through the Groq API.

## 11. Display Sources

The application displays the retrieved policy sections and their PDF page numbers.

---

# 🔐 Groq API Key

Never place your Groq API key directly inside `app.py`.

Do NOT write:

```python
GROQ_API_KEY = "actual-api-key"
```

inside the source code.

Do NOT put the key in:

* GitHub
* README.md
* requirements.txt
* `.gitignore`
* any Python source file

Instead, use Streamlit Secrets.

---

# 🔑 Streamlit Secrets

For Streamlit Community Cloud, add:

```toml
GROQ_API_KEY = "your_actual_api_key"
```

The application reads the key using:

```python
st.secrets["GROQ_API_KEY"]
```

Streamlit recommends keeping secrets outside the Git repository and adding them through the Community Cloud secrets interface.

---

# ☁️ GitHub Deployment

## Step 1 — Create GitHub Repository

Go to:

https://github.com/

Create a new repository named:

```text
hr-policy-assistant
```

---

## Step 2 — Add Files

Using the GitHub website, create these four files:

```text
app.py
requirements.txt
README.md
.gitignore
```

Paste the corresponding content into each file.

The repository should look like:

```text
hr-policy-assistant/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

Do not upload private HR Policy PDFs to the repository.

---

# 🚀 Streamlit Community Cloud Deployment

Go to:

https://share.streamlit.io/

Sign in with GitHub.

Streamlit's current deployment process is:

```text
Create app
↓
Yup, I have an app
↓
Select GitHub repository
↓
Select branch
↓
Select app.py
↓
Advanced settings
↓
Add Secrets
↓
Deploy
```

Streamlit's current documentation confirms this repository/branch/entrypoint deployment workflow.

---

# 🔑 Add Streamlit Secret

During deployment, open:

```text
Advanced settings
```

Find the:

```text
Secrets
```

field.

Paste:

```toml
GROQ_API_KEY = "your_actual_api_key"
```

Then save and deploy.

Streamlit Community Cloud specifically supports adding the contents of `secrets.toml` through the deployment Advanced Settings interface.

---

# 🐍 Python Version

If Streamlit asks for a Python version, use a currently supported version such as Python 3.12.

Community Cloud currently defaults to Python 3.12 and allows the Python version to be selected from Advanced settings during deployment.

---

# 🧪 Testing

After deployment, upload an HR Policy PDF and test questions such as:

1. What is the annual leave policy?

2. How many sick leaves are allowed?

3. What is the maternity leave policy?

4. What is the emergency leave procedure?

5. What are the organization's working hours?

6. What is the resignation notice period?

7. What is the disciplinary procedure?

8. What is the probation period?

9. Who is eligible for annual leave?

10. What is the overtime policy?

Also test a question that is **not contained in the PDF**.

The expected behavior is:

```text
I could not find this information in the uploaded HR policy.
```

---

# 🐛 Troubleshooting

## ModuleNotFoundError

Example:

```text
ModuleNotFoundError: No module named 'faiss'
```

Check that `requirements.txt` contains:

```text
faiss-cpu
```

Then commit the change to GitHub.

Community Cloud detects dependency changes and reinstalls the environment.

---

## Missing GROQ_API_KEY

If you see:

```text
GROQ_API_KEY is missing.
```

Go to your Streamlit app settings and add:

```toml
GROQ_API_KEY = "your_actual_api_key"
```

Do not put the API key in `app.py`.

---

## PDF Extraction Problem

If the application says:

```text
No extractable text was found in the PDF.
```

the PDF may be a scanned/image-only document.

This version requires a text-extractable PDF. OCR can be added as a future feature.

---

## FAISS Installation Problem

Make sure the dependency is:

```text
faiss-cpu
```

not:

```text
faiss
```

---

## Groq API Error

Check:

1. The API key is correct.
2. The secret is named exactly:

```text
GROQ_API_KEY
```

3. The selected model is:

```text
openai/gpt-oss-20b
```

4. The Groq service/model is currently available.

The current Groq documentation confirms the model ID and Python SDK usage.

---

## Empty Retrieval Results

Make sure:

1. The PDF contains selectable text.
2. The document was processed successfully.
3. The FAISS status says:

```text
FAISS ready
```

4. Try a question that clearly relates to the policy.

---

## Streamlit Deployment Error

Check that the repository root contains:

```text
app.py
requirements.txt
README.md
.gitignore
```

and that Streamlit's selected main file is:

```text
app.py
```

The dependency file should be in the repository root or alongside the entrypoint.

---

# 🔒 Security Reminder

HR policies can contain confidential information.

Do not commit:

```text
GROQ API keys
Employee information
Confidential HR documents
Private PDF files
.streamlit/secrets.toml
```

Use Streamlit Secrets for API credentials.

---

# 📌 Important RAG Limitation

This application intentionally does not guarantee that an LLM can never hallucinate. The design reduces hallucination by:

```text
Question
↓
FAISS retrieval
↓
Relevant policy chunks
↓
Strict policy-only prompt
↓
Groq
```

The model is explicitly instructed to state:

```text
I could not find this information in the uploaded HR policy.
```

when the supplied context does not contain the answer.

For high-stakes HR decisions, the retrieved source pages should still be checked by an authorized HR professional.

```
```
