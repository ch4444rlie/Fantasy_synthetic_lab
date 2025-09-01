
import pandas as pd
import numpy as np
from pathlib import Path
import pdfplumber
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
from IPython.display import display, Markdown
import ollama
import signal
import sys

# Signal handler for graceful exit
def signal_handler(sig, frame):
    display(Markdown("🪄 **Arcane Investigation Terminated.**"))
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

display(Markdown("🔮 **Initiating Valdris Financial Investigation System...**"))
display(Markdown("=" * 80))

class FantasyRAGChatbot:
    def __init__(self, transaction_csv="mission1_synthetic_data.csv", docs_folder="mission2_documents"):
        self.transaction_csv = transaction_csv
        self.docs_folder = docs_folder
        self.documents = []
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.faiss_index = None
        self.bm25 = None
        self.model_name = "mistral:7b-instruct-v0.3-q4_0"
        self.df = None
        display(Markdown("✓ **Initialized Arcane RAG System for Transaction and Document Analysis**"))

    def load_transaction_data(self):
        """Load and process transaction data from CSV"""
        display(Markdown(f"\n1️⃣ **Loading Enchanted Ledger from {self.transaction_csv}...**"))
        try:
            self.df = pd.read_csv(self.transaction_csv)
        except FileNotFoundError:
            display(Markdown(f"⚠️ **Error: {self.transaction_csv} not found. Ensure Mission 1 data is generated.**"))
            return []
        transaction_docs = []
        for idx, row in self.df.iterrows():
            doc_text = f"""
TRANSACTION #{row['transaction_id']}
Amount: ${row['amount']:.2f}
Sender: {row.get('customer_id', row['sender_location'])} ({row['sender_location']})
Receiver: {row.get('merchant_id', row['receiver_location'])} ({row['receiver_location']})
Type: {row['transaction_type']}
Suspicious: {row['is_money_laundering']}
Risk Score: {row['merchant_risk_score']:.2f}
Cross Border: {row['cross_border']}
Cash Equivalent: {row['cash_equivalent']}
Customer Segment: {row['customer_segment']}
Account Age Days: {row['account_age_days']}
Transaction Hour: {row['transaction_hour']}
Day of Week: {row['day_of_week']}
Transactions Last 24h: {row['transactions_last_24h']}
Balance Ratio: {row['account_balance_ratio']:.2f}
            """.strip()
            metadata = {
                "type": "transaction",
                "transaction_id": str(row['transaction_id']),
                "is_money_laundering": bool(row['is_money_laundering']),
                "sender_location": row['sender_location'],
                "receiver_location": row['receiver_location'],
                "amount": float(row['amount']),
                "customer_id": str(row.get('customer_id', '')),
                "merchant_id": str(row.get('merchant_id', ''))
            }
            transaction_docs.append({"text": doc_text, "metadata": metadata})
        display(Markdown(f"✓ **Loaded {len(transaction_docs)} ledger entries**"))
        return transaction_docs

    def load_investigation_documents(self):
        """Extract text and tables from PDFs"""
        display(Markdown(f"\n2️⃣ **Deciphering Ancient Scrolls from {self.docs_folder}...**"))
        docs_path = Path(self.docs_folder)
        if not docs_path.exists():
            display(Markdown(f"⚠️ **Error: {self.docs_folder} not found. Run Mission 2 to generate documents.**"))
            return []
        investigation_docs = []
        for pdf_file in docs_path.glob("*.pdf"):
            display(Markdown(f"📜 **Decoding {pdf_file.name}...**"))
            with pdfplumber.open(pdf_file) as pdf:
                full_text = ""
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    tables = page.extract_tables()
                    for table in tables:
                        table_str = "\n".join([",".join(map(str, row)) for row in table])
                        text += f"\nTable:\n{table_str}\n"
                    full_text += text + "\n"
                chunks = [full_text[i:i+500] for i in range(0, len(full_text), 500)]
                doc_type = "unknown"
                if "whistleblower" in pdf_file.name.lower():
                    doc_type = "whistleblower_report"
                elif "bank_statement" in pdf_file.name.lower():
                    doc_type = "bank_statement"
                elif "sar" in pdf_file.name.lower():
                    doc_type = "suspicious_activity_report"
                for chunk_idx, chunk in enumerate(chunks):
                    metadata = {"type": "document", "document_type": doc_type, "filename": pdf_file.name, "chunk_id": chunk_idx}
                    investigation_docs.append({"text": chunk, "metadata": metadata})
        display(Markdown(f"✓ **Processed {len(investigation_docs)} scroll fragments**"))
        return investigation_docs

    def setup_index(self):
        """Build FAISS and BM25 indices"""
        self.documents = self.load_transaction_data() + self.load_investigation_documents()
        if not self.documents:
            display(Markdown("⚠️ **Error: No documents loaded. Check data and document paths.**"))
            return
        # Embed documents
        for doc in self.documents:
            doc["embedding"] = self.embedder.encode(doc["text"], convert_to_numpy=True)
        # FAISS index
        embeddings = np.array([doc["embedding"] for doc in self.documents]).astype('float32')
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings)
        # BM25 index
        tokenized_docs = [doc["text"].lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        display(Markdown(f"✓ **Indexed {len(self.documents)} artifacts with FAISS and BM25**"))

    def search_documents(self, query, top_k=5):
        """Hybrid search with BM25 and SentenceTransformers"""
        query_embedding = self.embedder.encode(query, convert_to_numpy=True).astype('float32')
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        distances, indices = self.faiss_index.search(query_embedding.reshape(1, -1), top_k * 2)
        semantic_scores = [1 / (1 + d) for d in distances[0]]
        combined_scores = [(i, 0.4 * bm25_scores[i] + 0.6 * semantic_scores[j]) 
                          for j, i in enumerate(indices[0])]
        top_indices = sorted(combined_scores, key=lambda x: x[1], reverse=True)[:top_k]
        results = {
            "documents": [self.documents[i]["text"] for i, _ in top_indices],
            "metadatas": [self.documents[i]["metadata"] for i, _ in top_indices]
        }
        return results

    def generate_response(self, query, search_results):
        """Generate response with Ollama, tied to transactions and documents"""
        context = "\n\n".join(search_results["documents"][:5])
        # Dynamic transaction summary
        if "suspicious" in query.lower() or "K1C5" in query.lower() or "money laundering" in query.lower():
            suspicious_df = self.df[self.df['is_money_laundering'] == 1]
            if "K1C5" in query.lower():
                suspicious_df = suspicious_df[(suspicious_df['sender_location'] == 'K1C5') | 
                                            (suspicious_df['receiver_location'] == 'K1C5')]
            summary = f"""
Transaction Summary:
- Suspicious Transactions: {len(suspicious_df)}
- Total Amount: ${suspicious_df['amount'].sum():.2f}
- Top Locations: {suspicious_df['sender_location'].value_counts().head(3).to_dict()}
- Patterns: {suspicious_df['transaction_type'].value_counts().head(2).to_dict()}
"""
            context += "\n" + summary
        prompt = f"""
By the decree of the Valdris Council, you are the Arcane Investigator, tasked with exposing financial crimes in the Kingdom.
CONTEXT (prioritize transactions, then documents; K1C5 is Kingdom 1 City 5):
{context}
QUESTION: {query}
Respond in 150 words or less:
- Suspicious Patterns: [List, e.g., frequent K1C5 transactions]
- Key Entities: [Accounts, locations, segments]
- Money Laundering Indicators: [Evidence from transactions or documents]
- Risk Assessment: [Recommendation]
Be specific, using data from context.
"""
        try:
            response = ollama.chat(model=self.model_name, messages=[{'role': 'user', 'content': prompt}])
            return response['message']['content']
        except Exception as e:
            display(Markdown(f"⚠️ **Error: Ollama failed - {str(e)}. Ensure Ollama server is running.**"))
            return "Unable to generate response due to Ollama error."

    def investigate(self, query):
        """Run investigation query"""
        display(Markdown(f"\n🔍 **Arcane Query: {query}**"))
        if not self.documents or self.faiss_index is None:
            display(Markdown("⚠️ **Error: Index not set up. Run setup_index first.**"))
            return ""
        search_results = self.search_documents(query, top_k=10)
        response = self.generate_response(query, search_results)
        display(Markdown(f"📜 **Arcane Findings**:\n{response}"))
        return response

    def run(self):
        """Main loop for interactive investigation"""
        self.setup_index()
        if not self.documents:
            display(Markdown("⚠️ **Error: No data loaded. Investigation halted.**"))
            return
        display(Markdown("\n🏰 **Arcane Investigation Chamber Active**"))
        display(Markdown("Type your query or 'exit' to close the chamber."))
        display(Markdown("Example Queries:"))
        display(Markdown("- Suspicious transactions in K1C5\n- Details in whistleblower reports\n- Cross-border activity in bank statements"))
        question_count = 0
        while True:
            question = input(f"🪄 Query #{question_count + 1}: ").strip()
            if question.lower() in ['exit', 'quit', 'stop']:
                display(Markdown(f"\n👋 **Chamber Closed. {question_count} queries investigated.**"))
                break
            if question:
                self.investigate(question)
                question_count += 1
            else:
                display(Markdown("💡 **Enter a query or 'exit'.**"))

if __name__ == "__main__":
    chatbot = FantasyRAGChatbot()
    chatbot.run()
