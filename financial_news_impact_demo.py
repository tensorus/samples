import streamlit as st
import torch
import pandas as pd
import numpy as np
import re
import json
import uuid
import time
import logging
from typing import List, Dict, Tuple, Optional, Callable, Any
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel, pipeline

# Set page config must be the first Streamlit command
st.set_page_config(page_title="Financial News Impact RAG Demo", layout="wide")

# --- Configure basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Embedded TensorStorage Class (Simplified) ---
class EmbeddedTensorStorage:
    def __init__(self):
        self.datasets: Dict[str, Dict[str, List[Any]]] = {}
        logger.info("EmbeddedTensorStorage initialized (In-Memory).")

    def create_dataset(self, name: str) -> None:
        if name in self.datasets:
            raise ValueError(f"Dataset '{name}' already exists.")
        self.datasets[name] = {"tensors": [], "metadata": []}
        logger.info(f"Dataset '{name}' created successfully.")

    def insert(self, name: str, tensor: torch.Tensor, metadata: Optional[Dict[str, Any]] = None) -> str:
        if name not in self.datasets:
            self.create_dataset(name) # Auto-create if not exists for demo simplicity
            # raise ValueError(f"Dataset '{name}' does not exist. Create it first.")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Data to be inserted must be a torch.Tensor.")
        if metadata is None: metadata = {}
        record_id = metadata.get("record_id", str(uuid.uuid4())) # Allow pre-defined ID for linking
        metadata["record_id"] = record_id # Ensure it's there
        metadata["timestamp_utc"] = metadata.get("timestamp_utc", time.time())
        metadata["shape"] = list(tensor.shape)
        metadata["dtype"] = str(tensor.dtype).replace('torch.', '')

        self.datasets[name]["tensors"].append(tensor.clone())
        self.datasets[name]["metadata"].append(metadata)
        return record_id

    def get_records_by_metadata_filter(self, dataset_name: str, filter_fn: Callable[[Dict], bool]) -> List[Dict[str, Any]]:
        if dataset_name not in self.datasets: return []
        results = []
        for i, meta in enumerate(self.datasets[dataset_name]["metadata"]):
            if filter_fn(meta):
                results.append({"tensor": self.datasets[dataset_name]["tensors"][i], "metadata": meta})
        return results

    def get_all_records(self, dataset_name: str) -> List[Dict[str, Any]]:
        if dataset_name not in self.datasets: return []
        return [{"tensor": t, "metadata": m} for t, m in zip(self.datasets[dataset_name]["tensors"], self.datasets[dataset_name]["metadata"])]

# --- End of Embedded TensorStorage ---

# --- NLP Models & Utilities (Load once) ---
@st.cache_resource
def load_nlp_models():
    # Using a lighter model for faster demo
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return tokenizer, model, sentiment_analyzer

tokenizer, model, sentiment_analyzer = load_nlp_models()

def get_embedding(text: str, max_length=128) -> torch.Tensor:
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze() # Squeeze to make it 1D

def get_token_level_embeddings(text: str, max_length=128) -> torch.Tensor:
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=max_length, return_attention_mask=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Return embeddings for actual tokens (not padding), up to max_length
    # last_hidden_state shape: [batch_size, sequence_length, hidden_size]
    token_embeddings = outputs.last_hidden_state.squeeze(0) # Remove batch_dim
    # attention_mask shape: [batch_size, sequence_length]
    # actual_tokens_mask = inputs['attention_mask'].squeeze(0).bool()
    # return token_embeddings[actual_tokens_mask] # This would return variable length
    return token_embeddings # Return fixed length with padding, model using it should handle mask

def get_simplified_attention_flow(text: str, max_length=32) -> torch.Tensor:
    """Simulates a basic attention flow - not a real transformer attention map."""
    tokens = tokenizer.tokenize(text, padding=True, truncation=True, max_length=max_length)
    num_tokens = min(len(tokens), max_length) # Use actual number of tokens up to max_length
    if num_tokens == 0: return torch.zeros((max_length, max_length))
    
    # Simple heuristic: words closer together get higher attention
    # In a real scenario, this would come from a model's attention heads.
    attention_matrix = torch.zeros((num_tokens, num_tokens))
    for i in range(num_tokens):
        for j in range(num_tokens):
            distance = abs(i - j)
            attention_matrix[i, j] = 1.0 / (1.0 + distance) # Closer words, higher attention
    
    # Pad to max_length x max_length if necessary
    if num_tokens < max_length:
        padded_attention = torch.zeros((max_length, max_length))
        padded_attention[:num_tokens, :num_tokens] = attention_matrix
        return padded_attention
    return attention_matrix


# --- Sample Financial Data ---
# Assets involved in our hypothetical scenarios
ASSETS = ["TechCorp (TC)", "InnovateInc (II)", "GlobalWidgets (GW)"]

NEWS_EVENTS_DATA = [
    {
        "event_id": "EVT001",
        "timestamp": time.mktime(pd.to_datetime("2025-05-19 09:00:00").timetuple()),
        "headline": "TechCorp announces breakthrough in AI chips, stock surges!",
        "full_text": "TechCorp today unveiled its new 'QuantumLeap' AI processor, promising a 10x performance increase. Analysts predict a significant market disruption. The stock for TechCorp (TC) jumped 15% in pre-market trading. Competitors like InnovateInc (II) saw a slight dip.",
        "affected_assets": ["TechCorp (TC)", "InnovateInc (II)"],
        "primary_asset_impacted": "TechCorp (TC)",
        "market_context_features": { # Simplified market features at the time of news
            "TechCorp (TC)": {"price": 150.0, "volatility": 0.8, "volume_spike": 3.5},
            "InnovateInc (II)": {"price": 120.0, "volatility": 0.5, "volume_spike": 1.2},
            "GlobalWidgets (GW)": {"price": 80.0, "volatility": 0.3, "volume_spike": 1.0}
        },
        "future_impact_simulated": { # What we'd ideally predict
            "TechCorp (TC)": {"price_change_pct": 10.5, "volatility_change": 0.2},
            "InnovateInc (II)": {"price_change_pct": -2.0, "volatility_change": 0.1},
            "GlobalWidgets (GW)": {"price_change_pct": 0.1, "volatility_change": 0.0}
        }
    },
    {
        "event_id": "EVT002",
        "timestamp": time.mktime(pd.to_datetime("2025-05-19 11:30:00").timetuple()),
        "headline": "Regulatory concerns hit InnovateInc over data privacy, stock tumbles.",
        "full_text": "Shares of InnovateInc (II) dropped sharply by 8% after a report raised significant data privacy concerns regarding its flagship product. TechCorp (TC) remained largely unaffected, while GlobalWidgets (GW) saw minor gains as investors sought safer havens.",
        "affected_assets": ["InnovateInc (II)", "TechCorp (TC)", "GlobalWidgets (GW)"],
        "primary_asset_impacted": "InnovateInc (II)",
         "market_context_features": {
            "TechCorp (TC)": {"price": 165.0, "volatility": 0.7, "volume_spike": 1.1},
            "InnovateInc (II)": {"price": 110.0, "volatility": 1.2, "volume_spike": 4.0},
            "GlobalWidgets (GW)": {"price": 81.0, "volatility": 0.3, "volume_spike": 1.5}
        },
        "future_impact_simulated": {
            "TechCorp (TC)": {"price_change_pct": 0.5, "volatility_change": 0.0},
            "InnovateInc (II)": {"price_change_pct": -7.5, "volatility_change": 0.5},
            "GlobalWidgets (GW)": {"price_change_pct": 1.0, "volatility_change": -0.05}
        }
    }
]

# --- TensorStorage Initialization & Data Ingestion ---
if 'ts_finance' not in st.session_state:
    st.session_state.ts_finance = EmbeddedTensorStorage()
ts_finance: EmbeddedTensorStorage = st.session_state.ts_finance

# Dataset names
VB_NEWS_EMBEDDINGS_DS = "vb_news_embeddings"
TD_NEWS_TOKEN_EMBEDDINGS_DS = "td_news_token_embeddings"
TD_NEWS_ATTENTION_DS = "td_news_attention_sim" # Simulated attention
TD_NEWS_SENTIMENT_DS = "td_news_sentiment_tensor" # e.g. [overall_sent_score, positive_keywords_embedding_sum, negative_keywords_embedding_sum]
TD_MARKET_CONTEXT_DS = "td_market_context_tensor" # e.g. [asset1_price, asset1_vol, asset2_price, asset2_vol, ...]


def ingest_financial_data():
    # For Vector DB approach (VB): Store one embedding per news event
    for event in NEWS_EVENTS_DATA:
        news_embedding = get_embedding(event["full_text"])
        metadata = {
            "event_id": event["event_id"],
            "headline": event["headline"],
            "timestamp_utc": event["timestamp"],
            "raw_text_snippet": event["full_text"][:200], # Store snippet for VB context
            "_future_impact_simulated": event["future_impact_simulated"] # For demo comparison
        }
        ts_finance.insert(VB_NEWS_EMBEDDINGS_DS, news_embedding, metadata)

    # For Tensor Database approach (TD): Store multiple, richer tensors per event
    for event in NEWS_EVENTS_DATA:
        event_id = event["event_id"]
        common_metadata = {
            "event_id": event_id,
            "headline": event["headline"],
            "timestamp_utc": event["timestamp"],
            "raw_text_snippet_for_context": event["full_text"][:200], # Store full text if needed by model later
             "_future_impact_simulated": event["future_impact_simulated"] # For demo comparison
        }

        # 1. Token-level embeddings for the news text (2D Tensor)
        token_embeds = get_token_level_embeddings(event["full_text"])
        ts_finance.insert(TD_NEWS_TOKEN_EMBEDDINGS_DS, token_embeds, {**common_metadata, "tensor_type": "token_embeddings"})

        # 2. Simulated Attention Flow (2D Tensor)
        attention_tensor = get_simplified_attention_flow(event["full_text"])
        ts_finance.insert(TD_NEWS_ATTENTION_DS, attention_tensor, {**common_metadata, "tensor_type": "attention_flow_simulated"})

        # 3. Sentiment Tensor (1D or 2D Tensor) - simplified here
        # Example: [overall_sentiment_score (from -1 to 1), num_positive_keywords, num_negative_keywords]
        sentiment_result = sentiment_analyzer(event["headline"])[0] # Use headline for simplicity
        overall_score = (1 if sentiment_result['label'] == 'POSITIVE' else -1) * sentiment_result['score']
        # Dummy keyword counts
        num_pos_keywords = len(re.findall(r'good|great|breakthrough|surges|gains', event["full_text"], re.IGNORECASE))
        num_neg_keywords = len(re.findall(r'bad|poor|concerns|tumbles|dip', event["full_text"], re.IGNORECASE))
        sentiment_data = torch.tensor([overall_score, float(num_pos_keywords), float(num_neg_keywords)])
        ts_finance.insert(TD_NEWS_SENTIMENT_DS, sentiment_data, {**common_metadata, "tensor_type": "sentiment_features"})

        # 4. Market Context Tensor (1D or 2D Tensor)
        # Example: Flattened vector of [asset1_price, asset1_vol, asset2_price, asset2_vol, ...] for affected assets
        market_features_list = []
        ordered_assets_for_context = sorted(event["market_context_features"].keys()) # Consistent order
        for asset in ordered_assets_for_context:
            features = event["market_context_features"][asset]
            market_features_list.extend([features["price"], features["volatility"], features["volume_spike"]])
        market_context_tensor = torch.tensor(market_features_list, dtype=torch.float32)
        ts_finance.insert(TD_MARKET_CONTEXT_DS, market_context_tensor, {**common_metadata, "tensor_type": "market_context", "context_asset_order": ordered_assets_for_context})

    st.session_state.finance_data_ingested = True
    st.sidebar.success("Financial event data ingested into Tensorus representations.")

# --- Retrieval Logic for RAG ---
def retrieve_context_for_rag(event_id: str, approach: str = "vector_db"):
    retrieved_context = {"text_chunks": [], "structured_tensors": {}, "metadata": {}}
    
    if approach == "vector_db":
        st.write(f"**Vector DB Approach: Retrieving context for Event ID: {event_id}**")
        # Simulate VB: fetch the single news embedding and its associated raw text snippet
        records = ts_finance.get_records_by_metadata_filter(
            VB_NEWS_EMBEDDINGS_DS,
            lambda meta: meta.get("event_id") == event_id
        )
        if records:
            record = records[0]
            retrieved_context["text_chunks"].append(record["metadata"]["raw_text_snippet"])
            retrieved_context["structured_tensors"]["overall_news_embedding (VB)"] = {
                "shape": list(record["tensor"].shape),
                "preview": record["tensor"][:5].tolist() # Show first 5 dims
            }
            retrieved_context["metadata"] = record["metadata"]
            st.success("Retrieved: Main text snippet and its overall embedding.")
        else:
            st.error("Event not found in VB representation.")

    elif approach == "tensor_db":
        st.write(f"**Tensor DB Approach: Retrieving multi-faceted context for Event ID: {event_id}**")
        # Simulate TD: fetch multiple rich tensors associated with the event
        event_found = False
        
        # 1. Token Embeddings
        token_records = ts_finance.get_records_by_metadata_filter(
            TD_NEWS_TOKEN_EMBEDDINGS_DS, lambda meta: meta.get("event_id") == event_id
        )
        if token_records:
            event_found = True
            retrieved_context["text_chunks"].append(token_records[0]["metadata"]["raw_text_snippet_for_context"]) # Primary text
            retrieved_context["structured_tensors"]["token_embeddings (TD)"] = {
                "shape": list(token_records[0]["tensor"].shape),
                "comment": "Represents sequence & nuance of news text."
            }
            retrieved_context["metadata"] = token_records[0]["metadata"] # Base metadata
            st.success("Retrieved: Token-level embeddings (captures sequence).")


        # 2. Attention Flow
        attention_records = ts_finance.get_records_by_metadata_filter(
            TD_NEWS_ATTENTION_DS, lambda meta: meta.get("event_id") == event_id
        )
        if attention_records:
            event_found = True
            retrieved_context["structured_tensors"]["attention_flow_sim (TD)"] = {
                "shape": list(attention_records[0]["tensor"].shape),
                "comment": "Highlights internal text relationships."
            }
            st.success("Retrieved: Simulated attention flow tensor.")


        # 3. Sentiment Tensor
        sentiment_records = ts_finance.get_records_by_metadata_filter(
            TD_NEWS_SENTIMENT_DS, lambda meta: meta.get("event_id") == event_id
        )
        if sentiment_records:
            event_found = True
            retrieved_context["structured_tensors"]["sentiment_features (TD)"] = {
                "shape": list(sentiment_records[0]["tensor"].shape),
                "data": sentiment_records[0]["tensor"].tolist(),
                "comment": "[OverallScore, PosKeywords, NegKeywords]"
            }
            st.success("Retrieved: Structured sentiment tensor.")

        # 4. Market Context Tensor
        market_records = ts_finance.get_records_by_metadata_filter(
            TD_MARKET_CONTEXT_DS, lambda meta: meta.get("event_id") == event_id
        )
        if market_records:
            event_found = True
            retrieved_context["structured_tensors"]["market_context (TD)"] = {
                "shape": list(market_records[0]["tensor"].shape),
                "data": market_records[0]["tensor"].tolist(),
                "asset_order": market_records[0]["metadata"].get("context_asset_order"),
                "comment": "Price, Vol, VolumeSpike for assets at news time."
            }
            st.success("Retrieved: Market context tensor at time of news.")
        
        if not event_found:
            st.error("Event not found in TD representations.")

    return retrieved_context

# --- Streamlit UI ---
st.title("💸 Multi-Modal News Impact RAG: Tensor DB vs. Vector DB")
st.caption("Illustrating how a Tensor Database can provide richer context for financial prediction RAG.")

if 'finance_data_ingested' not in st.session_state:
    st.session_state.finance_data_ingested = False

st.sidebar.header("Setup")
if not st.session_state.finance_data_ingested:
    if st.sidebar.button("Load & Ingest Sample Financial Events"):
        with st.spinner("Processing and ingesting financial data... This may take a few minutes for NLP models."):
            ingest_financial_data()
        st.rerun()
else:
    st.sidebar.success("Sample financial data loaded.")
    if st.sidebar.button("Re-Ingest Data (Clears Demo Data)"):
        st.session_state.ts_finance = EmbeddedTensorStorage() # Re-init
        ts_finance = st.session_state.ts_finance # update global ref
        with st.spinner("Re-ingesting financial data..."):
            ingest_financial_data()
        st.rerun()

if st.session_state.finance_data_ingested:
    st.header("Simulate RAG Context Retrieval for Prediction")
    
    event_options = {event["event_id"]: event["headline"] for event in NEWS_EVENTS_DATA}
    selected_event_id = st.selectbox(
        "Select a News Event to Analyze:",
        options=list(event_options.keys()),
        format_func=lambda x: f"{x}: {event_options[x]}"
    )

    if selected_event_id:
        st.subheader(f"Event: {event_options[selected_event_id]}")
        original_news_text = next((e["full_text"] for e in NEWS_EVENTS_DATA if e["event_id"] == selected_event_id), "N/A")
        with st.expander("Original News Text"):
            st.markdown(original_news_text)

        # Hypothetical Prediction Task:
        # "Given this news event, predict the price change % and volatility change for TechCorp (TC), InnovateInc (II), and GlobalWidgets (GW)."
        st.markdown("""
        **Hypothetical Predictive Model's Goal:**
        Given the news event, predict the impact on key assets (e.g., price change %, volatility change).
        A more informed context (input) for this model should lead to better predictions.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Vector Database (VB) RAG Approach")
            st.caption("VB typically retrieves a single embedding & associated text chunk for the event.")
            if st.button("Retrieve VB Context", key="vb_retrieve"):
                with st.spinner("Retrieving context (VB style)..."):
                    vb_context = retrieve_context_for_rag(selected_event_id, approach="vector_db")
                st.markdown("**Context Provided to Predictive Model (VB):**")
                if vb_context["text_chunks"]:
                    st.info(f"**Text Snippet:** \"{vb_context['text_chunks'][0]}...\"")
                if "overall_news_embedding (VB)" in vb_context["structured_tensors"]:
                    st.json({"overall_news_embedding": vb_context["structured_tensors"]["overall_news_embedding (VB)"]})
                st.markdown("*This context is primarily semantic similarity of the whole news item.*")


        with col2:
            st.subheader("Tensor Database (TD) RAG Approach (Tensorus)")
            st.caption("TD retrieves multiple, richer tensor representations of the event & its context.")
            if st.button("Retrieve TD Context", key="td_retrieve"):
                with st.spinner("Retrieving context (TD style)..."):
                    td_context = retrieve_context_for_rag(selected_event_id, approach="tensor_db")
                st.markdown("**Context Provided to Predictive Model (TD):**")
                if td_context["text_chunks"]: # Main text snippet
                     st.info(f"**Primary Text Snippet:** \"{td_context['text_chunks'][0]}...\"")
                st.json(td_context["structured_tensors"]) # Show all the different tensors
                st.markdown("""
                *This context provides the predictive model with:*
                * *Sequence information* from token embeddings.
                * *Internal text relationships* from attention flow.
                * *Structured sentiment indicators*.
                * *Market state at the time of news*.
                *Potentially leading to more nuanced and accurate impact predictions across multiple assets.*
                """)
        
        st.divider()
        st.subheader("Visualizing the 'Superiority' for Prediction")
        st.markdown("""
        While we don't train a model here, the **key difference for a predictive AI model** lies in the input context:

        * **Vector DB RAG:** Provides a single general semantic vector and a text chunk. The model has to infer all relationships and specific impacts from this limited view. It might capture the main sentiment but miss how different parts of the news affect different assets or how the market was already positioned.

        * **Tensor DB RAG (Tensorus):** Provides a *multi-faceted tensor profile* of the event:
            * **Token Embeddings Tensor:** The *full sequence and nuance* of the news text, not just a summary.
            * **Attention Tensor:** *Which parts of the news are internally important* or related.
            * **Sentiment Tensor:** *Structured sentiment signals*, perhaps beyond a single score.
            * **Market Context Tensor:** The *state of relevant assets at the moment of the news*.

        **An advanced predictive model receiving the Tensor Database context can:**
        1.  Understand the news text more deeply (from token sequences & attention).
        2.  Correlate specific parts of the news with specific sentiment features.
        3.  Condition its predictions on the pre-existing market state for multiple assets.
        4.  Potentially predict not just a single outcome, but a *vector or tensor of outcomes* (e.g., impact on multiple assets simultaneously, or price + volatility changes).

        This richer, structured, multi-modal input allows the predictive model to learn more complex patterns and make more accurate, specific, and robust financial forecasts compared to relying on a single document-level embedding.
        """)

else:
    st.info("Please load and ingest sample financial event data from the sidebar to enable the demo.")