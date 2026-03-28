"""
ScannerAgent — scrapes RSS feeds for deals and uses GPT to curate the best ones.
"""

import json
import logging
from openai import OpenAI
from dotenv import load_dotenv
from agents.deals import ScrapedDeal, Deal, DealSelection

load_dotenv(override=True)

BG_BLUE = "\033[44m"
YELLOW = "\033[33m"
RESET = "\033[0m"

MODEL = "gpt-5.4-mini"

SYSTEM_PROMPT = """You identify and summarize the 5 most detailed deals from a list, by selecting deals that have the most detailed, high quality description and the most clear price.
Respond strictly in JSON with no explanation, using this format. You should provide the price as a number derived from the description. If the price of a deal isn't clear, do not include that deal in your response.
Most important is that you respond with the 5 deals that have the most detailed product description with price. It's not important to mention the terms of the deal; most important is a thorough description of the product.
Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price.
"""

USER_PROMPT_PREFIX = """Respond with the most promising 5 deals from this list, selecting those which have the most detailed, high quality product description and a clear price that is greater than 0.
You should rephrase the description to be a summary of the product itself, not the terms of the deal.
Remember to respond with a short paragraph of text in the product_description field for each of the 5 items that you select.
Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price.

Deals:

"""

USER_PROMPT_SUFFIX = "\n\nInclude exactly 5 deals, no more."


class ScannerAgent:
    def __init__(self):
        self.openai = OpenAI()
        self.log("ScannerAgent ready")

    def log(self, message: str):
        text = BG_BLUE + YELLOW + "[Scanner] " + message + RESET
        logging.info(text)

    def scan(self) -> DealSelection:
        """Scrape RSS feeds and curate the best deals using GPT."""
        # Step 1: Fetch raw deals from RSS feeds
        self.log("Scanning RSS feeds for deals...")
        scraped = ScrapedDeal.fetch(show_progress=True)
        self.log(f"Found {len(scraped)} raw deals")

        if not scraped:
            self.log("No deals found!")
            return DealSelection(deals=[])

        # Step 2: Build the prompt with all scraped deals
        user_prompt = USER_PROMPT_PREFIX
        user_prompt += '\n\n'.join([scrape.describe() for scrape in scraped])
        user_prompt += USER_PROMPT_SUFFIX

        # Step 3: Ask GPT to curate the top 5
        self.log(f"Asking {MODEL} to curate top 5 deals...")
        
        try:
            response = self.openai.chat.completions.parse(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=DealSelection,
            )
            result = response.choices[0].message.parsed
        except Exception as e:
            self.log(f"Structured output failed: {e}")
            self.log("Falling back to manual JSON parsing...")
            result = None

        # Fallback: if structured output didn't work, try regular completion
        if result is None:
            response = self.openai.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
            )
            reply = response.choices[0].message.content
            self.log(f"Raw GPT response: {reply[:200]}...")
            
            try:
                # Try to parse JSON from the response
                data = json.loads(reply)
                if 'deals' in data:
                    deals = [Deal(**d) for d in data['deals']]
                else:
                    deals = [Deal(**d) for d in data]
                result = DealSelection(deals=deals)
            except Exception as e2:
                self.log(f"JSON parsing also failed: {e2}")
                return DealSelection(deals=[])

        self.log(f"Curated {len(result.deals)} deals")
        for deal in result.deals:
            self.log(f"  ${deal.price:.2f} — {deal.product_description[:60]}...")

        return result