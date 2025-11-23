A simple chatbot that lets you talk to your PDF documents.

## What it does

Upload PDFs, ask questions, get answers. That's it.

The app uses RAG (Retrieval Augmented Generation) to find relevant parts of your documents and generate accurate responses based on actual content from your files.

## Setup

1. Clone this repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Get a free API key from [Groq](https://console.groq.com)
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Features

- Multiple chat sessions
- Upload as many PDFs as you want
- Rename or delete chats
- All data stored in memory (nothing saved to disk)

## How to use

1. Enter your Groq API key on the welcome screen
2. Create a new chat
3. Upload your PDFs
4. Start asking questions

## Tech stack

- Streamlit for the UI
- LangChain for the RAG pipeline
- ChromaDB for vector storage
- Groq (llama-3.1-8b-instant) for the LLM
- HuggingFace embeddings

## Notes

The app doesn't save anything permanently. When you restart it, everything resets. This is by design to keep things simple and avoid dealing with file storage.

If you need help or find bugs, open an issue.

## License

www.github.com/pranavsebastianlouis

Try it Out : https://docuchat-end-to-end-rag-pipeline-chroma.onrender.com/
