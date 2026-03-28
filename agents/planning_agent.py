"""
AutonomousPlanningAgent — uses GPT tool-calling to orchestrate all agents.
GPT decides which tools to call and in what order.
"""

import json
import logging
from openai import OpenAI
from dotenv import load_dotenv
from agents.scanner_agent import ScannerAgent
from agents.ensemble_agent import EnsembleAgent
from agents.messaging_agent import MessagingAgent
from agents.deals import Deal, Opportunity

load_dotenv(override=True)

BG_BLUE = "\033[44m"
WHITE = "\033[37m"
RESET = "\033[0m"

MODEL = "gpt-5.4-mini"

# ── Tool definitions ──
# These tell GPT what functions are available and what parameters they need.
# GPT reads these descriptions and decides when to call each one.

scan_function = {
    "name": "scan_the_internet_for_bargains",
    "description": "Returns top bargains scraped from the internet along with the price each item is being offered for",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False
    }
}

estimate_function = {
    "name": "estimate_true_value",
    "description": "Given the description of an item, estimate how much it is actually worth",
    "parameters": {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "The description of the item to be estimated"
            },
        },
        "required": ["description"],
        "additionalProperties": False
    }
}

notify_function = {
    "name": "notify_user_of_deal",
    "description": "Send the user a push notification about the single most compelling deal; only call this one time",
    "parameters": {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "The description of the item itself scraped from the internet"
            },
            "deal_price": {
                "type": "number",
                "description": "The price offered by this deal scraped from the internet"
            },
            "estimated_true_value": {
                "type": "number",
                "description": "The estimated actual value that this is worth"
            },
            "url": {
                "type": "string",
                "description": "The URL of this deal as scraped from the internet"
            }
        },
        "required": ["description", "deal_price", "estimated_true_value", "url"],
        "additionalProperties": False
    }
}

TOOLS = [
    {"type": "function", "function": scan_function},
    {"type": "function", "function": estimate_function},
    {"type": "function", "function": notify_function},
]

SYSTEM_MESSAGE = "You find great deals on bargain products using your tools, and notify the user of the best bargain."

USER_MESSAGE = """
First, use your tool to scan the internet for bargain deals. Then for each deal, use your tool to estimate its true value.
Then pick the single most compelling deal where the price is much lower than the estimated true value, and use your tool to notify the user.
Then just reply OK to indicate success.
"""


class PlanningAgent:
    def __init__(self, collection=None):
        self.log("Initializing PlanningAgent...")
        self.openai = OpenAI()
        self.scanner = ScannerAgent()
        self.ensemble = EnsembleAgent(collection)
        self.messenger = MessagingAgent()
        self.log("PlanningAgent ready — all sub-agents initialized")

    def log(self, message: str):
        text = BG_BLUE + WHITE + "[Planner] " + message + RESET
        logging.info(text)

    def handle_tool_call(self, message) -> list:
        """Execute the tool calls that GPT requested."""
        results = []

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            self.log(f"GPT wants to call: {tool_name}")

            if tool_name == "scan_the_internet_for_bargains":
                # Call ScannerAgent to get deals
                deal_selection = self.scanner.scan()
                result = deal_selection.model_dump_json()

            elif tool_name == "estimate_true_value":
                # Call EnsembleAgent to estimate the true value
                description = arguments["description"]
                estimate = self.ensemble.price(description)
                result = json.dumps({
                    "description": description,
                    "estimated_true_value": estimate
                })

            elif tool_name == "notify_user_of_deal":
                # Call MessagingAgent to alert the user
                deal = Deal(
                    product_description=arguments["description"],
                    price=arguments["deal_price"],
                    url=arguments["url"]
                )
                opportunity = Opportunity(
                    deal=deal,
                    estimate=arguments["estimated_true_value"],
                    discount=arguments["estimated_true_value"] - arguments["deal_price"]
                )
                self.messenger.alert(opportunity)
                result = json.dumps({"status": "notification sent"})
                self.last_opportunity = opportunity

            else:
                result = json.dumps({"error": f"Unknown tool: {tool_name}"})

            results.append({
                "role": "tool",
                "content": result,
                "tool_call_id": tool_call.id
            })

        return results

    def plan(self, memory=None) -> Opportunity:
        """Run the autonomous planning loop."""
        self.log("Starting autonomous planning loop...")
        self.last_opportunity = None

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": USER_MESSAGE}
        ]

        # Add memory of past deals if available
        if memory:
            past_deals = [
                f"${opp.deal.price:.2f} - {opp.deal.product_description[:100]}"
                for opp in memory
            ]
            memory_msg = "Previously found deals (don't repeat these):\n" + "\n".join(past_deals)
            messages.append({"role": "user", "content": memory_msg})

        # The agent loop — GPT keeps calling tools until it's done
        done = False
        loop_count = 0
        max_loops = 15  # Safety limit

        while not done and loop_count < max_loops:
            loop_count += 1
            self.log(f"Agent loop iteration {loop_count}...")

            response = self.openai.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
            )

            if response.choices[0].finish_reason == "tool_calls":
                # GPT wants to call a tool — execute it and feed the result back
                message = response.choices[0].message
                tool_results = self.handle_tool_call(message)
                messages.append(message)
                messages.extend(tool_results)
            else:
                # GPT said "OK" or gave a final text response — we're done
                final_reply = response.choices[0].message.content
                self.log(f"Agent finished: {final_reply}")
                done = True

        if loop_count >= max_loops:
            self.log("Hit max loop limit — stopping")

        return self.last_opportunity