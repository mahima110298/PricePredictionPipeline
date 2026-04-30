"""Local demo of the price-prediction pipeline.

Runs the parts of the pipeline that don't require remote infrastructure:
  * ScannerAgent.test_scan() — uses a baked-in set of sample deals (no RSS calls)
  * FrontierAgent — RAG over the local Chroma collection seeded by seed_chroma.py
  * NeuralNetworkAgent — only if ./deep_neural_network.pth exists
  * SpecialistAgent — only if a Modal pricer-service is deployed and reachable

Skipped components are excluded from the ensemble; weights are renormalized.

Run:
    python demo.py
"""

import logging
import os
import sys
from typing import Dict, List

import chromadb
from dotenv import load_dotenv

from agents.frontier_agent import FrontierAgent
from agents.scanner_agent import ScannerAgent

load_dotenv(override=True)

DB_PATH = "products_vectorstore"
COLLECTION = "products"


def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def open_collection():
    if not os.path.isdir(DB_PATH):
        sys.exit(
            f"Chroma database '{DB_PATH}' not found.\n"
            "Run `python seed_chroma.py` first to build it."
        )
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(COLLECTION)


def maybe_neural_network_agent():
    if not os.path.exists("deep_neural_network.pth"):
        print("[demo] deep_neural_network.pth not found - skipping NeuralNetworkAgent")
        return None
    from agents.neural_network_agent import NeuralNetworkAgent
    return NeuralNetworkAgent()


def maybe_specialist_agent():
    if os.getenv("ENABLE_SPECIALIST") != "1":
        print("[demo] ENABLE_SPECIALIST!=1 - skipping SpecialistAgent (set to 1 if Modal pricer-service is deployed)")
        return None
    try:
        from agents.specialist_agent import SpecialistAgent
        return SpecialistAgent()
    except Exception as exc:
        print(f"[demo] SpecialistAgent unavailable ({exc.__class__.__name__}) - skipping")
        return None


def ensemble_price(description: str, frontier, neural, specialist) -> Dict[str, float]:
    estimates = {}
    estimates["frontier"] = frontier.price(description)
    if neural:
        estimates["neural_network"] = neural.price(description)
    if specialist:
        estimates["specialist"] = specialist.price(description)

    weights = {"frontier": 0.8, "specialist": 0.1, "neural_network": 0.1}
    active = {k: weights[k] for k in estimates}
    total = sum(active.values())
    combined = sum(estimates[k] * (active[k] / total) for k in estimates)
    estimates["combined"] = combined
    return estimates


def main():
    init_logging()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")

    collection = open_collection()
    frontier = FrontierAgent(collection)
    neural = maybe_neural_network_agent()
    specialist = maybe_specialist_agent()

    deals = ScannerAgent().test_scan().deals
    print(f"\n=== Pricing {len(deals)} sample deals ===\n")
    for deal in deals:
        estimates = ensemble_price(deal.product_description, frontier, neural, specialist)
        discount = estimates["combined"] - deal.price
        verdict = "DEAL" if discount > 50 else "skip"
        print(f"-- {verdict} (discount ${discount:+.2f}) --")
        print(f"   listed: ${deal.price:.2f}")
        for k, v in estimates.items():
            print(f"   {k:>15}: ${v:.2f}")
        print(f"   {deal.product_description[:120]}...\n")


if __name__ == "__main__":
    main()
