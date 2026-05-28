# RAG Chatbot with Text Files

A Streamlit-based application that allows you to chat with the content of text documents using Retrieval Augmented Generation (RAG) technology powered by OpenAI's language models.
<img width="1875" height="778" alt="sample" src="https://github.com/user-attachments/assets/04cc8b21-5bc3-452f-baa7-d29588e89221" />


## Features

- **File Upload**: Upload any text file to chat with its content
- **Multiple AI Models**: Choose from various OpenAI models (GPT-5, GPT-4, GPT-4o, GPT-4o Mini)
- **Customizable Embeddings**: Select different embedding models for optimal performance
- **Temperature Control**: Adjust the creativity/determinism of responses
- **Reranking Option**: Enable NVIDIA reranking for improved answer quality
- **Streaming Responses**: Real-time response generation for better user experience
- **Document Preview**: View your uploaded document content

## How to Use

1. **Set API Keys**: 
   - Enter your OpenAI API key (required)
   - Optionally provide a NVIDIA API key for reranking functionality

2. **Select Models**:
   - Choose a chat model for response generation
   - Select an embedding model for document processing

3. **Adjust Settings**:
   - Set temperature (0.0 for deterministic, 1.0 for creative responses)
   - Enable/disable reranking if you have a NVIDIA API key

4. **Upload Document**:
   - Drag and drop or browse to upload a text file
   - Click "Process Document" to build the search indexes

5. **Ask Questions**:
   - Type questions about your document content
   - Receive answers based solely on the document content

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd rag-chatbot
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Requirements

The application requires the following Python packages:
- streamlit
- langchain
- langchain-community
- langchain-openai
- langchain-text-splitters
- faiss-cpu
- langchain-nvidia-ai-endpoints

## API Keys

You'll need to provide:
- **OpenAI API Key**: Required for accessing language models
- **NVIDIA API Key**: Optional, for enhanced reranking functionality

## Model Options

### Chat Models
- GPT-5: latest model
- GPT-4: More capable for complex tasks
- GPT-4o: OpenAI's most advanced model
- GPT-4o Mini: Smaller, cost-effective version

### Embedding Models
- text-embedding-3-small: Fastest option
- text-embedding-3-large: Higher quality embeddings
- text-embedding-ada-002: Balanced option

## Technical Details

This application uses:
- **FAISS** for efficient vector similarity search
- **BM25** for keyword-based retrieval
- **Ensemble Retrieval** combining both methods
- **NVIDIA Reranker** for result quality improvement (optional)
- **OpenAI Embeddings** for document processing

## Limitations

- Currently supports only text files (.txt)
- Document processing required before querying
- Larger documents may take longer to process

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.



---

Built with ❤️ using LangChain, OpenAI, FAISS, and Streamlit.


