"""
SpecialistAgent — uses the fine-tuned Llama 3.2 model deployed on Modal.
"""

import logging
import modal

BG_BLUE = "\033[44m"
WHITE = "\033[37m"
RESET = "\033[0m"


class SpecialistAgent:
    def __init__(self):
        self.log("Connecting to Modal pricer-service...")
        Pricer = modal.Cls.from_name("pricer-service", "Pricer")
        self.pricer = Pricer()
        self.log("Connected!")

    def log(self, message: str):
        text = BG_BLUE + WHITE + "[Specialist] " + message + RESET
        logging.info(text)

    def price(self, description: str) -> float:
        """Predict a price using the fine-tuned model on Modal."""
        self.log(f"Estimating price for: {description[:50]}...")
        result = self.pricer.price.remote(description)
        self.log(f"Estimated: ${result:.2f}")
        return result