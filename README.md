# 🧠 RAG Document Chatbot

A high-performance, production-ready Retrieval-Augmented Generation (RAG) web application. This app allows you to upload large PDFs and documents, index them using FAISS, and intelligently chat with them using OpenAI or DeepSeek models.

It features a stunning, modern dark-mode UI built on **Gradio 5** and a lightning-fast asynchronous backend powered by **FastAPI** and **LangChain**.

---

## ✨ Features
- **Multi-Document Support:** Upload PDFs, TXT, DOCX, Markdown, HTML, and CSVs.
- **Lightning Fast Vector DB:** Uses local FAISS for millisecond retrieval times.
- **Intelligent Chunking:** Configurable chunk sizes and overlaps to prevent data loss across tables and complex PDF layouts.
- **Dynamic API Switching:** Instantly swap between OpenAI (`gpt-4o`, `gpt-4o-mini`) and DeepSeek from the frontend UI without restarting the server.
- **Smart Citations:** Automatically attaches a collapsible `<details>` block at the bottom of the AI's response showing exactly which documents it used.
- **Persistent Memory:** Chat history and Vector Databases survive container restarts via mounted volumes.

---

## 🚀 Running Locally (Windows / VS Code)

The fastest way to test changes before pushing to production is running the app locally on your machine.

1. **Clone the repository and open in VS Code.**
2. **Create a Virtual Environment (Optional but recommended):**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Create a `.env` file:**
   Create a `.env` file in the root directory and add your API keys:
   ```env
   OPENAI_API_KEY=sk-your-key-here
   DEEPSEEK_API_KEY=sk-your-key-here
   
   CHUNK_SIZE=3000
   CHUNK_OVERLAP=600
   ```
5. **Start the Server:**
   ```bash
   python api.py
   ```
6. **Open your browser:** Go to `http://localhost:8000`

---

## ☁️ Deploying to Google Cloud (Compute Engine VM)

This application is containerized with Docker, making it perfectly suited for a GCP Virtual Machine.

### 1. Initial VM Setup
When creating your VM in the Google Cloud Console, **you must check the "Allow HTTP traffic" firewall box**.

### 2. Install Docker & Clone
SSH into your VM and run:
```bash
sudo apt-get update
sudo apt-get install -y docker.io git
sudo systemctl enable --now docker

git clone <your-repo-url>
cd <your-repo-folder>
```

### 3. Setup Persistent Volumes & Environment Variables
Create the folders that will survive Docker restarts:
```bash
mkdir faiss_index
mkdir uploads
nano .env # Paste your API keys here!
```

### 4. Build and Run
```bash
sudo docker build -t my-rag-app .

sudo docker run -d \
  --name my-rag-app \
  -p 80:8000 \
  -v $(pwd)/faiss_index:/app/faiss_index \
  -v $(pwd)/uploads:/app/uploads \
  --env-file .env \
  --restart unless-stopped \
  my-rag-app
```
*(You can now visit your VM's External IP address in your browser! Make sure to explicitly type `http://` and NOT `https://`)*

---

## 🛠️ Troubleshooting & Known Quirks

- **"Device or resource busy: 'faiss_index'" Error:**
  If you attempt to clear the knowledge base and see this error, ensure your code is updated. The app is designed to delete the *contents* of the folder rather than the folder itself, because Linux prevents you from deleting active Docker volume mounts.
  
- **Browser Connection Timeout (GCP):**
  If your VM is running perfectly (`curl -I http://localhost` returns 200 OK) but you cannot access it from your browser, you are likely clicking the IP link directly from the GCP Dashboard. GCP automatically prepends `https://`, which breaks the connection. Always manually type `http://<YOUR_IP>` in a new tab.

- **Orange UI / Freezing:**
  This application is pinned to **Gradio 5**. Do not upgrade to Gradio 6, as it introduces severe breaking changes to the CSS engine and Server-Sent Events (SSE) which will break the streaming UI.
