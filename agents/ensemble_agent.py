"""
EnsembleAgent — combines Specialist and Frontier predictions.
"""

import logging

from agents.specialist_agent import SpecialistAgent
from agents.frontier_agent import FrontierAgent

BG_BLUE = "\033[44m"
MAGENTA = "\033[35m"
RESET = "\033[0m"


class EnsembleAgent:
    FRONTIER_WEIGHT = 0.90
    SPECIALIST_WEIGHT = 0.10

    def __init__(self, collection=None):
        self.log("Initializing EnsembleAgent...")
        self.specialist = SpecialistAgent()
        self.frontier = FrontierAgent(collection)
        self.log("Ready!")

    def log(self, message: str):
        text = BG_BLUE + MAGENTA + "[Ensemble] " + message + RESET
        logging.info(text)

    def price(self, description: str) -> float:
        """Get price estimates from both agents and combine them."""
        self.log(f"Getting estimates for: {description[:50]}...")

        frontier_price = self.frontier.price(description)
        specialist_price = self.specialist.price(description)

        ensemble_price = (
            frontier_price * self.FRONTIER_WEIGHT +
            specialist_price * self.SPECIALIST_WEIGHT
        )

        self.log(f"Frontier: ${frontier_price:.2f} | Specialist: ${specialist_price:.2f}")
        self.log(f"Ensemble ({self.FRONTIER_WEIGHT}/{self.SPECIALIST_WEIGHT}): ${ensemble_price:.2f}")

        return ensemble_price