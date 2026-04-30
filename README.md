# Price Prediction Pipeline — Multi-Agent Deal Hunter

An autonomous agent framework that estimates the fair market price of online product deals and surfaces opportunities where the listed price is meaningfully below estimate. Combines a fine-tuned open-source LLM, a frontier model with retrieval-augmented context, a custom deep neural network, and a coordinator that turns the ensemble's estimates into ranked deal alerts.

## Architecture

```
                       ┌────────────────────────┐
                       │     Planning Agent     │
                       │  (orchestrates loop)   │
                       └────────────┬───────────┘
                                    │
        ┌───────────────────────────┼─────────────────────────────┐
        │                           │                             │
┌───────▼────────┐         ┌────────▼─────────┐         ┌─────────▼────────┐
│ Scanner Agent  │         │  Ensemble Agent  │         │ Messaging Agent  │
│  RSS + GPT-5   │         │ weighted average │         │ Pushover + Claude│
│ structured out │         └────────┬─────────┘         └──────────────────┘
└────────────────┘                  │
                  ┌─────────────────┼──────────────────┐
                  │                 │                  │
        ┌─────────▼────────┐  ┌─────▼──────┐  ┌────────▼────────┐
        │ Specialist Agent │  │  Frontier  │  │ Neural Network  │
        │ Llama-3.2-3B+LoRA│  │  GPT + RAG │  │ 10-layer ResNet │
        │  (Modal GPU)     │  │ (ChromaDB) │  │   MLP w/ skips  │
        └──────────────────┘  └────────────┘  └─────────────────┘
```

### What each piece does

| Component | Role |
|---|---|
| **ScannerAgent** | Pulls RSS feeds from DealNews, asks GPT-5-mini to pick the 5 most precisely-described deals using OpenAI structured outputs. |
| **FrontierAgent** | RAG: embeds the description with `all-MiniLM-L6-v2`, retrieves the 5 nearest products from ChromaDB, hands the list to GPT-5.1 with the prices as context. |
| **SpecialistAgent** | Calls a Llama-3.2-3B fine-tuned with QLoRA on a 400k-row product/price dataset, served as a Modal serverless GPU function (4-bit NF4 quant). |
| **NeuralNetworkAgent** | A 10-layer residual MLP (4096-wide hidden, dropout 0.2) over a 5000-dim hashing-vectorizer text representation; outputs `exp(z*σ + μ) - 1`. |
| **EnsembleAgent** | Weighted average: `0.8 * frontier + 0.1 * specialist + 0.1 * NN`. |
| **PlanningAgent** | Drives the loop: scan → score → if discount > $50, push notification with the URL. |
| **MessagingAgent** | Uses Claude to write an exciting 2-3 sentence push notification, sent via Pushover. Falls back to logs-only if Pushover creds aren't set. |
| **`price_is_right.py`** | Gradio Blocks UI: live deal table, agent log stream, 3D t-SNE plot of the vector store. |
| **`pricer_service.py`** | Modal entry point that boots the quantized Llama and exposes `Pricer.price()` over RPC. |

## Quickstart — local demo (no Modal, no Pushover)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in OPENAI_API_KEY

python seed_chroma.py    # builds a ~60-product Chroma collection
python demo.py           # prices 4 sample deals via FrontierAgent (and ensemble peers if available)
```

`demo.py` exercises the FrontierAgent + ensemble logic without needing the Modal-hosted Llama or the trained `.pth`. SpecialistAgent and NeuralNetworkAgent are skipped automatically if their resources aren't present, and the ensemble weights are renormalized over what *is* available.

## Full pipeline

To run the complete autonomous loop with the Gradio UI:

1. **Build a real ChromaDB.** Replace `seed_chroma.py` with embeddings of your full product corpus (the production setup uses ~400k Amazon items). Save to `products_vectorstore/`.
2. **Train and save `deep_neural_network.pth`** (the residual MLP defined in `agents/deep_neural_network.py`).
3. **Fine-tune Llama-3.2-3B with QLoRA** on a price-prediction dataset and push the adapter to HuggingFace Hub.
4. **Deploy the Modal service**:
   ```bash
   export HF_USER=your-hf-username HF_RUN_NAME=your-run-name
   modal deploy pricer_service.py
   ```
5. **Launch the UI**:
   ```bash
   python price_is_right.py
   ```

The UI auto-runs the Planning Agent every 5 minutes, surfaces deals with > $50 estimated discount, and (if Pushover is configured) sends push notifications written by Claude.

## Project layout

```
price-prediction-pipeline/
├── agents/
│   ├── agent.py              # Base class with colored logging
│   ├── scanner_agent.py      # RSS scrape + GPT structured output
│   ├── frontier_agent.py     # GPT + RAG over ChromaDB
│   ├── specialist_agent.py   # Modal RPC to fine-tuned Llama
│   ├── neural_network_agent.py
│   ├── deep_neural_network.py # 10-layer residual MLP
│   ├── ensemble_agent.py     # weighted combiner
│   ├── planning_agent.py     # autonomous orchestrator
│   ├── messaging_agent.py    # Pushover + Claude
│   ├── deals.py              # pydantic models, RSS scraper
│   ├── items.py              # HF Hub dataset I/O
│   └── preprocessor.py       # litellm-driven text normalizer
├── deal_agent_framework.py   # framework + memory + t-SNE plot data
├── price_is_right.py         # Gradio UI
├── pricer_service.py         # Modal serverless GPU service
├── seed_chroma.py            # tiny demo corpus seed
├── demo.py                   # local demo entry point
├── memory.json               # persisted opportunities (resets to empty)
├── requirements.txt
├── .env.example
└── README.md
```

## Tech stack

Python · OpenAI API (GPT-5.1, structured outputs) · Anthropic Claude · litellm · Llama-3.2-3B · QLoRA / PEFT / bitsandbytes · Modal (serverless GPU) · ChromaDB · sentence-transformers · PyTorch · scikit-learn · Gradio · Plotly · Pushover · feedparser
