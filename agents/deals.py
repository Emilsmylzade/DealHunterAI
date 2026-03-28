"""
Data classes for deals and opportunities.
"""

from pydantic import BaseModel
from typing import List, Optional


class Deal(BaseModel):
    """A product being sold at a certain price."""
    product_description: str
    price: float
    url: str


class Opportunity(BaseModel):
    """A deal where the price is lower than the estimated true value."""
    deal: Deal
    estimate: float  # What we think it's actually worth
    discount: float  # estimate - price (how much of a bargain)


class ScrapedDeal(BaseModel):
    """A deal scraped from an RSS feed before processing."""
    title: str
    summary: str
    link: str

    def describe(self) -> str:
        return f"Title: {self.title}\nSummary: {self.summary}\nLink: {self.link}"

    @classmethod
    def fetch(cls, show_progress=False) -> List['ScrapedDeal']:
        """Fetch deals from DealNews RSS feeds."""
        import requests
        import xml.etree.ElementTree as ET

        feeds = [
            "https://www.dealnews.com/?rss=1",
            "https://www.dealnews.com/c142/Electronics/?rss=1",
            "https://www.dealnews.com/c39/Computers/?rss=1",
        ]

        deals = []
        for feed_url in feeds:
            try:
                response = requests.get(feed_url, timeout=10)
                root = ET.fromstring(response.content)
                for item in root.findall('.//item'):
                    title = item.find('title')
                    description = item.find('description')
                    link = item.find('link')
                    if title is not None and description is not None and link is not None:
                        deals.append(cls(
                            title=title.text or "",
                            summary=description.text or "",
                            link=link.text or "",
                        ))
            except Exception as e:
                if show_progress:
                    print(f"Warning: couldn't fetch {feed_url}: {e}")
        return deals


class DealSelection(BaseModel):
    """GPT's curated selection of the best deals."""
    deals: List[Deal]