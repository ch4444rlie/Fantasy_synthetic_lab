## Welcome

## Purpose
This lab demonstrates synthetic data generation and usage in AML investigations, ensuring privacy and realism without real-world data. You’ll generate transactions, summon documents, analyze networks, and query findings in a magical setting.

## Key Components
- **Synthetic Data Generation**:
  - **Gaussian Copula**: Models statistical dependencies to generate realistic transaction distributions.
  - **CTGAN**: Uses GANs to capture complex, non-linear patterns in transaction data.
  - **Graph Network**: Constructs transaction graphs to model account relationships and flows.
  - **TVAE**: Employs variational autoencoders to generate diverse, high-fidelity transaction features.
- **Document Creation**:
  - **Markovify**: Generates unique, domain-specific text for whistleblower reports and SARs.
- **Tools**:
  - **Mission 1**: Creates synthetic transactions (`mission1_synthetic_data.csv`) using the above methods.
  - **Mission 2**: Summons PDFs (whistleblower reports, bank statements, SARs) with `reportlab>=4.2.2` and `markovify>=0.9.4`.
  - **Mission 3**: Trains a Prophecy Familiar (`AMLDetectionSystem`) with `torch>=2.8.0` and `torch-geometric>=2.6.1` to detect K1C5 patterns.
  - **Mission 4**: Deploys `FantasyRAGChatbot` with `rank-bm25>=0.2.2`, `sentence-transformers==3.2.0`, `faiss-cpu==1.9.0`, and `ollama>=0.5.3` to query evidence.

 ## Setup
1. **Install Dependencies**:
   ```bash
   pip install pandas==2.2.3 numpy>=1.26.4 torch>=2.8.0 torch-geometric>=2.6.1 networkx>=3.4.2 reportlab>=4.2.2 markovify>=0.9.4 pdfplumber>=0.11.7 rank-bm25>=0.2.2 sentence-transformers==3.2.0 faiss-cpu==1.9.0 ollama>=0.5.3
2. Install Ollama
ollama pull mistral:7b-instruct-v0.3-q4_0
ollama run mistral:7b-instruct-v0.3-q4_0
3. Ensure synthetic data is generated in the same directory as the notebook


## Usage
Run missions sequentially in a Jupyter notebook:

Mission 1: Generate synthetic transactions with realistic AML patterns.
Mission 2: Generate PDFs (FantasyDocumentForge) in mission2_documents/.
Mission 3: Train a Graph autoencoder (Prophecy Familiar) to detect money laundering patterns in Kingdom 1 City 5.
Mission 4: Query transactions and documents with a RAG Chatbot.
