"""
MessagingAgent — notifies the user of great deals.
Currently logs to console. Can be extended with Pushover for phone notifications.
"""

import logging

BG_BLUE = "\033[44m"
GREEN = "\033[32m"
RESET = "\033[0m"


class MessagingAgent:
    def __init__(self):
        self.log("MessagingAgent ready (console mode)")

    def log(self, message: str):
        text = BG_BLUE + GREEN + "[Messenger] " + message + RESET
        logging.info(text)

    def alert(self, opportunity):
        """Alert the user about a great deal opportunity."""
        deal = opportunity.deal
        message = (
            f"\n{'='*60}\n"
            f"🔥 DEAL ALERT!\n"
            f"{'='*60}\n"
            f"Product: {deal.product_description[:200]}\n"
            f"Price:    ${deal.price:.2f}\n"
            f"Estimate: ${opportunity.estimate:.2f}\n"
            f"Discount: ${opportunity.discount:.2f}\n"
            f"URL:      {deal.url}\n"
            f"{'='*60}"
        )
        self.log(message)
        print(message)