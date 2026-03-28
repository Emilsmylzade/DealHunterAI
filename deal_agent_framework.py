"""
DealAgentFramework — orchestrates the entire deal-hunting pipeline.
Manages memory of past deals and coordinates all agents.
"""

import os
import sys
import logging
import json
from typing import List, Optional
from dotenv import load_dotenv
import chromadb
from agents.planning_agent import PlanningAgent
from agents.deals import Opportunity
import numpy as np

load_dotenv(override=True)

BG_BLUE = "\033[44m"
WHITE = "\033[37m"
RESET = "\033[0m"

CATEGORIES = [
    "Appliances", "Automotive", "Cell_Phones_and_Accessories",
    "Electronics", "Musical_Instruments", "Office_Products",
    "Tools_and_Home_Improvement", "Toys_and_Games",
]
COLORS = ["red", "blue", "brown", "orange", "yellow", "green", "purple", "cyan"]


def init_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)


class DealAgentFramework:
    DB = "products_vectorstore"
    MEMORY_FILENAME = "memory.json"

    def __init__(self):
        init_logging()
        self.log("Connecting to vector store...")
        client = chromadb.PersistentClient(path=self.DB)
        self.collection = client.get_or_create_collection("products")
        self.log(f"Vector store has {self.collection.count():,} products")
        self.memory = self.read_memory()
        self.log(f"Loaded {len(self.memory)} deals from memory")
        self.planner = None

    def init_agents_as_needed(self):
        """Lazy initialization — only create agents when first needed."""
        if not self.planner:
            self.log("Initializing Agent Framework...")
            self.planner = PlanningAgent(self.collection)
            self.log("Agent Framework is ready")

    def read_memory(self) -> List[Opportunity]:
        """Load previously discovered deals from disk."""
        if os.path.exists(self.MEMORY_FILENAME):
            with open(self.MEMORY_FILENAME, "r") as file:
                data = json.load(file)
            return [Opportunity(**item) for item in data]
        return []

    def write_memory(self) -> None:
        """Save discovered deals to disk so they persist between runs."""
        data = [opp.model_dump() for opp in self.memory]
        with open(self.MEMORY_FILENAME, "w") as file:
            json.dump(data, file, indent=2)

    @classmethod
    def reset_memory(cls) -> None:
        """Reset memory back to empty (or first 2 deals)."""
        data = []
        if os.path.exists(cls.MEMORY_FILENAME):
            with open(cls.MEMORY_FILENAME, "r") as file:
                data = json.load(file)
        truncated = data[:2]
        with open(cls.MEMORY_FILENAME, "w") as file:
            json.dump(truncated, file, indent=2)

    def log(self, message: str):
        text = BG_BLUE + WHITE + "[Agent Framework] " + message + RESET
        logging.info(text)

    def run(self) -> List[Opportunity]:
        """Run one cycle of the deal-hunting agent."""
        self.init_agents_as_needed()
        self.log("Kicking off Planning Agent...")
        result = self.planner.plan(memory=self.memory)
        self.log(f"Planning Agent completed. Result: {result is not None}")
        if result:
            self.memory.append(result)
            self.write_memory()
            self.log(f"New deal saved! Total deals in memory: {len(self.memory)}")
        return self.memory

    @classmethod
    def get_plot_data(cls, max_datapoints=2000):
        """Get data for the 3D vector visualization in the Gradio UI."""
        from sklearn.manifold import TSNE

        client = chromadb.PersistentClient(path=cls.DB)
        collection = client.get_or_create_collection("products")
        result = collection.get(
            include=["embeddings", "documents", "metadatas"],
            limit=max_datapoints
        )
        vectors = np.array(result["embeddings"])
        documents = result["documents"]
        categories = [metadata["category"] for metadata in result["metadatas"]]
        colors = [COLORS[CATEGORIES.index(c)] for c in categories]
        tsne = TSNE(n_components=3, random_state=42, n_jobs=-1)
        reduced_vectors = tsne.fit_transform(vectors)
        return documents, reduced_vectors, colors


if __name__ == "__main__":
    DealAgentFramework().run()