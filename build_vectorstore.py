"""
Build the ChromaDB vector store with product embeddings.
Run this once: python3 build_vectorstore.py
"""

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm

# Configuration
DB = "products_vectorstore"
DATASET_NAME = "ed-donner/items_full"
COLLECTION_NAME = "products"
BATCH_SIZE = 500

def main():
    # Step 1: Load the dataset
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME)
    train = dataset['train']
    print(f"Loaded {len(train):,} training items")

    # Step 2: Load the embedding model
    # all-MiniLM-L6-v2 converts text into 384-dimensional vectors
    # Similar products will have similar vectors (close together in space)
    print("Loading embedding model...")
    encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Step 3: Create ChromaDB
    client = chromadb.PersistentClient(path=DB)
    
    # Delete existing collection if it exists, so we start fresh
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection")
    
    collection = client.create_collection(COLLECTION_NAME)
    print(f"Created collection '{COLLECTION_NAME}'")

    # Step 4: Process and insert in batches
    print(f"Processing {len(train):,} items in batches of {BATCH_SIZE}...")
    
    for i in tqdm(range(0, len(train), BATCH_SIZE)):
        batch = train[i:i + BATCH_SIZE]
        
        # Filter out items with missing summaries
        valid_indices = []
        summaries = []
        for j in range(len(batch['summary'])):
            if batch['summary'][j] is not None and len(batch['summary'][j].strip()) > 0:
                valid_indices.append(j)
                summaries.append(batch['summary'][j])
        
        if not summaries:
            continue

        # Compute embeddings — this converts text to 384-dim vectors
        embeddings = encoder.encode(summaries).tolist()

        # Prepare data for ChromaDB
        ids = [f"item_{i + j}" for j in valid_indices]
        metadatas = [
            {
                "category": batch['category'][j],
                "price": float(batch['price'][j])
            }
            for j in valid_indices
        ]

        # Insert into ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=summaries,
            metadatas=metadatas,
        )

    total = collection.count()
    print(f"\nDone! Stored {total:,} products in '{DB}/{COLLECTION_NAME}'")
    
    # Quick test — find products similar to "wireless headphones"
    print("\n── Quick test: products similar to 'wireless headphones' ──")
    test_vec = encoder.encode(["wireless bluetooth headphones"]).tolist()
    results = collection.query(query_embeddings=test_vec, n_results=3)
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        print(f"  ${meta['price']:.2f} | {doc[:80]}...")

if __name__ == "__main__":
    main()