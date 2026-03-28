"""
FrontierAgent — uses GPT-5.4-mini with RAG from ChromaDB.
Retrieves similar products with known prices, then asks GPT to estimate.
"""

import logging
import os
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv(override=True)

BG_BLUE = "\033[44m"
CYAN = "\033[36m"
RESET = "\033[0m"

DB = "products_vectorstore"
MODEL = "gpt-5.4-mini"


class FrontierAgent:
    def __init__(self, collection=None):
        self.log("Initializing FrontierAgent...")

        # Connect to OpenAI
        self.openai = OpenAI()

        # Load the same embedding model used to build the vector store
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Connect to ChromaDB — either use passed collection or load from disk
        if collection:
            self.collection = collection
        else:
            client = chromadb.PersistentClient(path=DB)
            self.collection = client.get_or_create_collection("products")

        self.log(f"Ready! {self.collection.count():,} products in vector store")

    def log(self, message: str):
        text = BG_BLUE + CYAN + "[Frontier] " + message + RESET
        logging.info(text)

    def find_similars(self, description: str, n_results: int = 5):
        """Find similar products in ChromaDB using vector similarity."""
        # Convert the description to a vector
        vector = self.encoder.encode(description).tolist()

        # Search ChromaDB for the closest vectors
        results = self.collection.query(
            query_embeddings=[vector],
            n_results=n_results
        )

        documents = results['documents'][0]
        prices = [m['price'] for m in results['metadatas'][0]]
        return documents, prices

    def make_context(self, similars, prices):
        """Build context string from similar products for GPT."""
        message = "For context, here are some similar products and their prices:\n\n"
        for similar, price in zip(similars, prices):
            message += f"Product: {similar[:200]}\nPrice: ${price:.2f}\n\n"
        return message

    def price(self, description: str) -> float:
        """Estimate price using GPT with RAG context."""
        import re

        self.log(f"Finding similar products for: {description[:50]}...")

        # Step 1: Find similar products (RAG retrieval)
        similars, prices = self.find_similars(description)
        self.log(f"Found {len(similars)} similar products")

        # Step 2: Build the prompt with context
        context = self.make_context(similars, prices)
        user_message = f"Estimate the price of this product. Respond with just the price number, no explanation.\n\n{description}\n\n{context}"

        # Step 3: Ask GPT-5.4-mini
        self.log(f"Asking {MODEL}...")
        response = self.openai.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": user_message}],
            seed=42,
        )

        # Step 4: Extract the price from GPT's response
        reply = response.choices[0].message.content
        reply = reply.replace("$", "").replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", reply)
        result = float(match.group()) if match else 0

        self.log(f"Estimated: ${result:.2f}")
        return result