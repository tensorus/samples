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

# --- Embedded TensorStorage Class (Simplified Simulation) ---
# This class simulates the core functionalities of a Tensor Database like Tensorus.
# It's an in-memory storage for demonstration purposes.
class EmbeddedTensorStorage:
    """
    A simplified in-memory simulation of a Tensor Database.
    It stores tensors and their associated metadata, organized into datasets.
    This is a basic substitute for a real Tensorus instance for this demo.
    """
    def __init__(self):
        """Initializes the in-memory storage for datasets."""
        self.datasets: Dict[str, Dict[str, List[Any]]] = {}
        logger.info("EmbeddedTensorStorage initialized (In-Memory Simulation).")

    def create_dataset(self, name: str) -> None:
        """
        Creates a new dataset (akin to a table or collection) to store tensors and metadata.
        Args:
            name: The unique name for the dataset.
        Raises:
            ValueError: If a dataset with the same name already exists.
        """
        if name in self.datasets:
            raise ValueError(f"Dataset '{name}' already exists.")
        self.datasets[name] = {"tensors": [], "metadata": []}
        logger.info(f"Dataset '{name}' created successfully in the simulated storage.")

    def insert(self, name: str, tensor: torch.Tensor, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Inserts a tensor and its associated metadata into a specified dataset.
        Args:
            name: The name of the dataset to insert into.
            tensor: The torch.Tensor object to be stored.
            metadata: An optional dictionary of metadata associated with the tensor.
                      A unique 'record_id' will be generated if not provided.
        Returns:
            str: The unique record ID for the inserted tensor (either provided or generated).
        Raises:
            TypeError: If the 'tensor' argument is not a torch.Tensor.
        """
        if name not in self.datasets:
            self.create_dataset(name) # Auto-create if not exists for demo simplicity
            logger.warning(f"Dataset '{name}' did not exist and was auto-created for insertion.")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Data to be inserted must be a torch.Tensor.")
        
        metadata = metadata if metadata is not None else {}
        # Use provided record_id or generate a new one; ensure it's part of the metadata.
        record_id = metadata.get("record_id", str(uuid.uuid4())) 
        metadata["record_id"] = record_id 
        # Add timestamp, tensor shape, and dtype to metadata for record-keeping.
        metadata["timestamp_utc"] = metadata.get("timestamp_utc", time.time()) 
        metadata["shape"] = list(tensor.shape) 
        metadata["dtype"] = str(tensor.dtype).replace('torch.', '') 

        # Store a clone of the tensor to prevent external modifications.
        self.datasets[name]["tensors"].append(tensor.clone())
        self.datasets[name]["metadata"].append(metadata)
        # logger.debug(f"Inserted tensor with ID {record_id} into dataset '{name}'.")
        return record_id

    def get_records_by_metadata_filter(self, dataset_name: str, filter_fn: Callable[[Dict], bool]) -> List[Dict[str, Any]]:
        """
        Retrieves records (tensor and its metadata) from a dataset that match a given filter function.
        The filter function is applied to the metadata of each record.
        Args:
            dataset_name: The name of the dataset to query.
            filter_fn: A callable that accepts a metadata dictionary and returns True if the record should be included.
        Returns:
            A list of dictionaries, where each dictionary contains a 'tensor' and its 'metadata' for matching records.
        """
        if dataset_name not in self.datasets:
            logger.warning(f"Attempted to query non-existent dataset: {dataset_name}")
            return []
        
        results = []
        for i, meta in enumerate(self.datasets[dataset_name]["metadata"]):
            if filter_fn(meta):
                results.append({"tensor": self.datasets[dataset_name]["tensors"][i], "metadata": meta})
        # logger.debug(f"Retrieved {len(results)} records from '{dataset_name}' using filter.")
        return results

    def get_all_records(self, dataset_name: str) -> List[Dict[str, Any]]:
        """
        Retrieves all records (tensor and metadata) from a specified dataset.
        Args:
            dataset_name: The name of the dataset.
        Returns:
            A list of all records in the dataset. Each record is a dictionary with 'tensor' and 'metadata'.
        """
        if dataset_name not in self.datasets:
            logger.warning(f"Attempted to get all records from non-existent dataset: {dataset_name}")
            return []
        return [{"tensor": t, "metadata": m} for t, m in zip(self.datasets[dataset_name]["tensors"], self.datasets[dataset_name]["metadata"])]

# --- End of EmbeddedTensorStorage ---

# --- NLP Models & Utilities (Load once using Streamlit's cache for efficiency) ---
@st.cache_resource # Caches the loaded models across Streamlit sessions/reruns, improving performance.
def load_nlp_models():
    """
    Loads and caches the NLP models required for the demo.
    This includes a sentence transformer (for generating embeddings) and a sentiment analysis pipeline.
    Using relatively lightweight models for faster execution in a demo environment.
    Returns:
        A tuple containing:
            - tokenizer: The tokenizer corresponding to the sentence transformer model.
            - model: The sentence transformer model itself.
            - sentiment_analyzer: A Hugging Face pipeline for sentiment analysis.
    """
    # Using a lighter, general-purpose sentence transformer model for speed.
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    # Using a DistilBERT-based model for sentiment analysis, fine-tuned on SST-2.
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    logger.info("NLP models (tokenizer, sentence-transformer, sentiment_analyzer) loaded successfully.")
    return tokenizer, model, sentiment_analyzer

tokenizer, model, sentiment_analyzer = load_nlp_models() # Load models globally for use in functions

def get_embedding(text: str, max_length=128) -> torch.Tensor:
    """
    Generates a single vector embedding (a sentence embedding) for a given text.
    This function uses mean pooling of token embeddings from the last hidden state of the model.
    The output is a 1D tensor representing the semantic meaning of the entire input text.
    Args:
        text: The input string to embed.
        max_length: The maximum token length for the tokenizer; texts longer than this will be truncated.
    Returns:
        A 1D torch.Tensor representing the sentence embedding of the input text.
        Shape: [embedding_dimension] (e.g., [384] for 'all-MiniLM-L6-v2').
    Role:
        Provides a compact, fixed-size representation of text, suitable for similarity comparisons
        or as a basic input feature for some machine learning models. This is typical for Vector DBs.
    """
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
    with torch.no_grad(): # Disable gradient calculations during inference for efficiency.
        outputs = model(**inputs)
    # Mean pooling of the last hidden state to get a sentence embedding.
    # .squeeze() removes dimensions of size 1, resulting in a 1D tensor.
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def get_token_level_embeddings(text: str, max_length=128) -> torch.Tensor:
    """
    Generates token-level embeddings for a given text. Each token in the input text
    is mapped to an embedding vector.
    The output is a 2D tensor where each row corresponds to a token's embedding.
    Args:
        text: The input string.
        max_length: The maximum sequence length. Texts will be padded or truncated to this length.
    Returns:
        A 2D torch.Tensor of shape (sequence_length, hidden_size).
        - sequence_length: Padded/truncated length (equal to max_length).
        - hidden_size: Dimensionality of each token embedding (e.g., 384 for 'all-MiniLM-L6-v2').
    Role:
        Provides a detailed representation of the text, preserving sequence information and
        the individual meaning of tokens in context. This is a richer representation
        often stored in Tensor Databases for more complex modeling.
    """
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=max_length, return_attention_mask=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # outputs.last_hidden_state is [batch_size, sequence_length, hidden_size].
    # .squeeze(0) removes the batch_dim (assuming batch_size is 1).
    token_embeddings = outputs.last_hidden_state.squeeze(0) 
    # Note: This returns embeddings for all tokens up to max_length, including padding tokens.
    # Downstream models should use an attention mask if they need to distinguish actual tokens from padding.
    return token_embeddings

def get_simplified_attention_flow(text: str, max_length=32) -> torch.Tensor:
    """
    Simulates a basic attention flow matrix for a given text.
    **This is a simplified heuristic and NOT a real attention map from a transformer model.**
    It's intended to represent, for demonstration, how different parts of the text might conceptually relate to each other.
    A real attention map would be derived from a model's internal attention mechanisms.
    Args:
        text: The input string.
        max_length: The maximum number of tokens to consider for the attention matrix.
                   The output matrix will be (max_length x max_length).
    Returns:
        A 2D torch.Tensor of shape (max_length, max_length) representing the simulated attention scores.
        Higher values suggest stronger (simulated) attention between token pairs (token i to token j).
    Role:
        Illustrates the concept of an attention mechanism, which can highlight important relationships
        or dependencies between different parts of a text. In a real Tensor DB scenario, actual
        attention tensors from a model could be stored to provide insights into text structure.
    """
    tokens = tokenizer.tokenize(text, padding=True, truncation=True, max_length=max_length)
    num_tokens = min(len(tokens), max_length) # Actual number of tokens considered, up to max_length.
    if num_tokens == 0: return torch.zeros((max_length, max_length)) # Return zero matrix if no tokens.
    
    # Simple heuristic: words closer together get higher "attention".
    # This is a placeholder for the complex calculations in a real transformer's attention.
    attention_matrix = torch.zeros((num_tokens, num_tokens))
    for i in range(num_tokens):
        for j in range(num_tokens):
            distance = abs(i - j)
            # Assign higher scores to pairs of tokens that are closer to each other.
            # This very simple rule simulates that local context is often more attended to.
            attention_matrix[i, j] = 1.0 / (1.0 + distance) 
    
    # Pad the attention matrix to ensure it has dimensions (max_length, max_length) for consistency.
    if num_tokens < max_length:
        padded_attention = torch.zeros((max_length, max_length))
        padded_attention[:num_tokens, :num_tokens] = attention_matrix
        return padded_attention
    return attention_matrix


# --- Sample Financial Data (Illustrative) ---
# Defines the assets involved in the hypothetical news scenarios. These are used for context in the demo.
ASSETS = ["TechCorp (TC)", "InnovateInc (II)", "GlobalWidgets (GW)"]

# A list of dictionaries, each representing a financial news event with associated data.
# This data is used to populate the simulated Tensor Database and Vector Database.
NEWS_EVENTS_DATA = [
    {
        "event_id": "EVT001", # Unique identifier for the event
        "timestamp": time.mktime(pd.to_datetime("2025-05-19 09:00:00").timetuple()),
        "headline": "TechCorp announces breakthrough in AI chips, stock surges!", # News headline
        "full_text": "TechCorp today unveiled its new 'QuantumLeap' AI processor, promising a 10x performance increase. Analysts predict a significant market disruption. The stock for TechCorp (TC) jumped 15% in pre-market trading. Competitors like InnovateInc (II) saw a slight dip.", # Full text of the news
        "affected_assets": ["TechCorp (TC)", "InnovateInc (II)"], # List of assets mentioned or impacted
        "primary_asset_impacted": "TechCorp (TC)", # The main asset discussed in the news
        "market_context_features": { # Simplified market features for relevant assets at the time of the news
            "TechCorp (TC)": {"price": 150.0, "volatility": 0.8, "volume_spike": 3.5}, # Example features: price, volatility, trading volume spike
            "InnovateInc (II)": {"price": 120.0, "volatility": 0.5, "volume_spike": 1.2},
            "GlobalWidgets (GW)": {"price": 80.0, "volatility": 0.3, "volume_spike": 1.0}
        },
        "future_impact_simulated": { # Hypothetical future impact data; for demo purposes to show what a model might predict
            "TechCorp (TC)": {"price_change_pct": 10.5, "volatility_change": 0.2},
            "InnovateInc (II)": {"price_change_pct": -2.0, "volatility_change": 0.1},
            "GlobalWidgets (GW)": {"price_change_pct": 0.1, "volatility_change": 0.0}
        }
    },
    # Second news event example
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
# Initialize the (simulated) TensorStorage instance. Store in session_state to persist across Streamlit reruns.
if 'ts_finance' not in st.session_state:
    st.session_state.ts_finance = EmbeddedTensorStorage()
ts_finance: EmbeddedTensorStorage = st.session_state.ts_finance # Get a reference for easier use in the script

# Define names for the datasets within the simulated TensorStorage. These names categorize the types of tensors stored.
VB_NEWS_EMBEDDINGS_DS = "vb_news_embeddings"            # For standard vector database (VB) style single embeddings per news event.
TD_NEWS_TOKEN_EMBEDDINGS_DS = "td_news_token_embeddings" # For token-level embeddings (Tensor Database - TD style).
TD_NEWS_ATTENTION_DS = "td_news_attention_sim"          # For simulated attention flow tensors (TD style).
TD_NEWS_SENTIMENT_DS = "td_news_sentiment_tensor"       # For structured sentiment feature tensors (TD style).
TD_MARKET_CONTEXT_DS = "td_market_context_tensor"       # For market context tensors (TD style).


def ingest_financial_data():
    """
    Processes the sample NEWS_EVENTS_DATA and ingests it into the simulated TensorStorage (ts_finance).
    Data is ingested in two ways to contrast Vector DB and Tensor DB approaches:
    1.  **Vector DB (VB) style:** Stores one global embedding per news event.
    2.  **Tensor DB (TD) style (Simulating Tensorus):** Stores multiple, diverse tensors per news event,
        capturing different facets like token sequences, attention, sentiment, and market context.
    """
    logger.info("Starting data ingestion process for financial news events into simulated storage.")

    # --- Ingestion for Vector DB (VB) Approach Simulation ---
    # For each news event, create and store a single embedding representing the entire news text.
    # This simulates how a traditional vector database might store and represent this data.
    for event in NEWS_EVENTS_DATA:
        # --- VB_NEWS_EMBEDDINGS_DS ---
        # - Represents: A single vector summarizing the entire semantic content of a news article.
        # - Dimensionality: 1D tensor (vector) of shape [embedding_dim] (e.g., [384] for all-MiniLM-L6-v2).
        #   - Each element is a component of the embedding vector.
        # - Usefulness for financial prediction:
        #   - Good for general semantic similarity search (finding related news).
        #   - Can capture overall positive/negative sentiment if the embedding model is tuned for it.
        #   - Limited for nuanced predictions as it averages out the entire text, losing specific details,
        #     sequential information, and relationships between different parts of the news or market context.
        news_embedding = get_embedding(event["full_text"]) 
        metadata = {
            "event_id": event["event_id"],
            "headline": event["headline"],
            "timestamp_utc": event["timestamp"],
            "raw_text_snippet": event["full_text"][:200], # Store a snippet for context during VB retrieval
            "_future_impact_simulated": event["future_impact_simulated"] # For demo comparison purposes
        }
        ts_finance.insert(VB_NEWS_EMBEDDINGS_DS, news_embedding, metadata)
    logger.info(f"Ingested {len(NEWS_EVENTS_DATA)} events into '{VB_NEWS_EMBEDDINGS_DS}' for VB simulation.")

    # --- Ingestion for Tensor Database (TD) Approach Simulation (Simulating Tensorus) ---
    # For each news event, create and store multiple, diverse tensors.
    # This demonstrates the richer data representation possible with a Tensor Database like Tensorus.
    for event in NEWS_EVENTS_DATA:
        event_id = event["event_id"]
        # Common metadata to be associated with all tensors derived from this specific news event.
        common_metadata = {
            "event_id": event_id,
            "headline": event["headline"],
            "timestamp_utc": event["timestamp"],
            "raw_text_snippet_for_context": event["full_text"][:200], # Store a snippet for quick reference
            "_future_impact_simulated": event["future_impact_simulated"] # Store ground truth for demo comparison
        }

        # --- TD_NEWS_TOKEN_EMBEDDINGS_DS ---
        # - Represents: Semantic meaning of each token (word/sub-word) in the news text, preserving sequence.
        # - Dimensionality: 2D tensor of shape [sequence_length, embedding_dim] (e.g., [128, 384]).
        #   - sequence_length: Number of tokens (padded/truncated to a fixed length like 128).
        #   - embedding_dim: Dimensionality of each token's embedding vector (e.g., 384).
        # - Usefulness for financial prediction:
        #   - Allows the model to understand the news text in detail, capturing nuances, specific phrases,
        #     and word order that are lost in a single document embedding.
        #   - Crucial for tasks requiring fine-grained understanding of text structure and how specific
        #     pieces of information within the news might affect different assets or market aspects.
        token_embeds = get_token_level_embeddings(event["full_text"])
        ts_finance.insert(TD_NEWS_TOKEN_EMBEDDINGS_DS, token_embeds, {**common_metadata, "tensor_type": "token_embeddings"})

        # --- TD_NEWS_ATTENTION_DS (Simulated) ---
        # - Represents: A simplified, heuristic-based matrix suggesting relationships or importance among tokens
        #   (e.g., which words "attend" more strongly to others within the text).
        #   This is a SIMULATION; a real system would use actual attention maps from a transformer model.
        # - Dimensionality: 2D tensor of shape [max_length, max_length] (e.g., [32, 32] for this demo).
        #   - Each dimension corresponds to token positions. Value indicates simulated attention strength.
        # - Usefulness for financial prediction (with real attention):
        #   - Could highlight key phrases, entities, or relationships within the news that are critical for impact.
        #   - Helps the model focus on the most relevant parts of the text, potentially filtering out noise.
        attention_tensor = get_simplified_attention_flow(event["full_text"], max_length=32) # Using a smaller max_length for this demo tensor
        ts_finance.insert(TD_NEWS_ATTENTION_DS, attention_tensor, {**common_metadata, "tensor_type": "attention_flow_simulated"})

        # --- TD_NEWS_SENTIMENT_DS ---
        # - Represents: Quantified sentiment aspects of the news, potentially multi-dimensional.
        # - Dimensionality: 1D tensor (vector) of shape [num_sentiment_features] (e.g., [3] for this demo:
        #   [overall_sentiment_score, num_positive_keywords, num_negative_keywords]).
        #   - Each element is a specific sentiment metric. Could be expanded to include aspect-based sentiment.
        # - Usefulness for financial prediction:
        #   - Provides explicit sentiment signals, which are often highly correlated with market reactions.
        #   - More structured than a single sentiment label (e.g., positive/negative/neutral), offering
        #     quantitative measures that a model can learn from.
        sentiment_result = sentiment_analyzer(event["headline"])[0] # Analyze headline for simplicity in this demo
        overall_score = (1 if sentiment_result['label'] == 'POSITIVE' else -1) * sentiment_result['score']
        # Simple keyword counts as additional sentiment features (could be replaced by more sophisticated methods)
        num_pos_keywords = len(re.findall(r'good|great|breakthrough|surges|gains|advances', event["full_text"], re.IGNORECASE))
        num_neg_keywords = len(re.findall(r'bad|poor|concerns|tumbles|dip|falls|regulatory', event["full_text"], re.IGNORECASE))
        sentiment_data = torch.tensor([overall_score, float(num_pos_keywords), float(num_neg_keywords)])
        ts_finance.insert(TD_NEWS_SENTIMENT_DS, sentiment_data, {**common_metadata, "tensor_type": "sentiment_features"})

        # --- TD_MARKET_CONTEXT_DS ---
        # - Represents: A snapshot of relevant market conditions (for specific assets) at the time the news event occurred.
        # - Dimensionality: 1D tensor (vector) for this demo, shape [num_assets * num_features_per_asset].
        #   (e.g., [asset1_price, asset1_vol, asset1_spike, asset2_price, asset2_vol, asset2_spike, ...]).
        #   Could also be a 2D tensor: [num_assets, num_features_per_asset].
        #   - Features include price, volatility, volume spikes for assets mentioned in `ordered_assets_for_context`.
        # - Usefulness for financial prediction:
        #   - Absolutely crucial. The impact of news often depends heavily on the prevailing market conditions.
        #   - Allows the model to learn how the same news might have different effects under different market scenarios
        #     (e.g., high vs. low volatility, or bullish vs. bearish trends for an asset).
        market_features_list = []
        # Sort asset names to ensure consistent order of features in the tensor
        ordered_assets_for_context = sorted(event["market_context_features"].keys()) 
        for asset in ordered_assets_for_context:
            features = event["market_context_features"][asset]
            market_features_list.extend([features["price"], features["volatility"], features["volume_spike"]])
        market_context_tensor = torch.tensor(market_features_list, dtype=torch.float32)
        ts_finance.insert(TD_MARKET_CONTEXT_DS, market_context_tensor, {**common_metadata, "tensor_type": "market_context", "context_asset_order": ordered_assets_for_context})
    
    logger.info(f"Ingested multi-faceted tensor representations for {len(NEWS_EVENTS_DATA)} events for TD (Tensorus) simulation.")
    st.session_state.finance_data_ingested = True
    # Update success message to reflect simulation
    st.sidebar.success("Financial event data ingested into (Simulated) Tensorus & VB representations.")

# --- Retrieval Logic for RAG (Retrieval Augmented Generation) ---
def retrieve_context_for_rag(event_id: str, approach: str = "vector_db"):
    """
    Simulates the retrieval of context for a given news event ID, using either a
    Vector DB approach or a Tensor DB (simulating Tensorus) approach.
    This function demonstrates what kind of data would be fed into a downstream prediction model.
    Args:
        event_id: The ID of the news event to retrieve context for.
        approach: "vector_db" or "tensor_db", specifying the retrieval strategy.
    Returns:
        A dictionary containing the retrieved context. This context typically includes:
        - "text_chunks": List of relevant text snippets.
        - "structured_tensors": Dictionary of retrieved tensors and their descriptions/data.
        - "metadata": Associated metadata for the event.
    """
    retrieved_context = {"text_chunks": [], "structured_tensors": {}, "metadata": {}}
    
    if approach == "vector_db":
        # --- Vector DB Retrieval Simulation ---
        # This approach typically retrieves one primary piece of information: the overall news embedding (a single vector)
        # and some basic associated metadata or the raw text snippet itself.
        # It's good for semantic similarity but lacks depth for complex, multi-faceted analysis.
        st.write(f"**Simulated Vector DB Approach: Retrieving context for Event ID: {event_id}**")
        records = ts_finance.get_records_by_metadata_filter(
            VB_NEWS_EMBEDDINGS_DS, # Query the dataset containing single, global news embeddings
            lambda meta: meta.get("event_id") == event_id
        )
        if records:
            record = records[0] # Expecting one record per event_id for VB approach
            retrieved_context["text_chunks"].append(record["metadata"]["raw_text_snippet"])
            retrieved_context["structured_tensors"]["overall_news_embedding (VB_sim)"] = { # Added _sim for clarity
                "shape": list(record["tensor"].shape),
                "preview": record["tensor"][:5].tolist() # Show first 5 dimensions as a preview
            }
            retrieved_context["metadata"] = record["metadata"]
            st.success("Retrieved: Main text snippet and its overall semantic embedding.")
        else:
            st.error(f"Event ID {event_id} not found in the simulated Vector DB representation.")

    elif approach == "tensor_db":
        # --- Tensor DB (Tensorus) Retrieval Simulation ---
        # This approach aims to gather a comprehensive set of diverse tensors associated with the event.
        # The combination of these tensors (e.g., token embeddings, attention maps, sentiment features, market state)
        # provides a much richer, multi-modal context for a downstream predictive model.
        # This allows the model to consider various aspects of the news and its environment simultaneously.
        st.write(f"**Simulated Tensor DB (Tensorus) Approach: Retrieving multi-faceted context for Event ID: {event_id}**")
        event_found_in_any_td_dataset = False # Flag to track if any data is found for the event
        
        # 1. Retrieve Token-Level Embeddings for the news text
        token_records = ts_finance.get_records_by_metadata_filter(
            TD_NEWS_TOKEN_EMBEDDINGS_DS, lambda meta: meta.get("event_id") == event_id
        )
        if token_records:
            event_found_in_any_td_dataset = True
            # Use the metadata from the first found tensor as the base for the event
            retrieved_context["metadata"] = token_records[0]["metadata"] 
            retrieved_context["text_chunks"].append(token_records[0]["metadata"]["raw_text_snippet_for_context"]) # Primary text reference
            retrieved_context["structured_tensors"]["token_embeddings (TD_sim)"] = { # Added _sim
                "shape": list(token_records[0]["tensor"].shape),
                "comment": "Captures the sequence and nuance of the news text."
            }
            st.success("Retrieved: Token-level embeddings tensor (captures text sequence).")

        # 2. Retrieve Simulated Attention Flow tensor
        attention_records = ts_finance.get_records_by_metadata_filter(
            TD_NEWS_ATTENTION_DS, lambda meta: meta.get("event_id") == event_id
        )
        if attention_records:
            event_found_in_any_td_dataset = True
            if not retrieved_context["metadata"]: retrieved_context["metadata"] = attention_records[0]["metadata"] # Fallback
            retrieved_context["structured_tensors"]["attention_flow_sim (TD_sim)"] = { # Added _sim
                "shape": list(attention_records[0]["tensor"].shape),
                "comment": "Simulated tensor highlighting internal text relationships."
            }
            st.success("Retrieved: Simulated attention flow tensor.")

        # 3. Retrieve Structured Sentiment Tensor
        sentiment_records = ts_finance.get_records_by_metadata_filter(
            TD_NEWS_SENTIMENT_DS, lambda meta: meta.get("event_id") == event_id
        )
        if sentiment_records:
            event_found_in_any_td_dataset = True
            if not retrieved_context["metadata"]: retrieved_context["metadata"] = sentiment_records[0]["metadata"] # Fallback
            retrieved_context["structured_tensors"]["sentiment_features (TD_sim)"] = { # Added _sim
                "shape": list(sentiment_records[0]["tensor"].shape),
                "data": sentiment_records[0]["tensor"].tolist(), # Show actual sentiment data
                "comment": "Structured sentiment: [OverallScore, PositiveKeywordsCount, NegativeKeywordsCount]"
            }
            st.success("Retrieved: Structured sentiment features tensor.")

        # 4. Retrieve Market Context Tensor
        market_records = ts_finance.get_records_by_metadata_filter(
            TD_MARKET_CONTEXT_DS, lambda meta: meta.get("event_id") == event_id
        )
        if market_records:
            event_found_in_any_td_dataset = True
            if not retrieved_context["metadata"]: retrieved_context["metadata"] = market_records[0]["metadata"] # Fallback
            retrieved_context["structured_tensors"]["market_context (TD_sim)"] = { # Added _sim
                "shape": list(market_records[0]["tensor"].shape),
                "data": market_records[0]["tensor"].tolist(), # Show actual market data
                "asset_order": market_records[0]["metadata"].get("context_asset_order"),
                "comment": "Market state (price, volatility, volume spike) for relevant assets at news time."
            }
            st.success("Retrieved: Market context tensor at the time of the news.")
        
        if not event_found_in_any_td_dataset:
            st.error(f"Event ID {event_id} not found in any of the simulated Tensor DB (Tensorus) representations.")
        elif event_found_in_any_td_dataset:
            # This message highlights the benefit of the Tensor DB (Tensorus) approach: combining multiple rich tensors.
            st.info("The combination of these retrieved tensors (token embeddings, attention, sentiment, market context) provides a richer, multi-modal input for a predictive financial model compared to a single vector embedding from a traditional Vector DB.")

    return retrieved_context

# --- Streamlit UI ---
# Updated title and caption to emphasize simulation
st.title("💸 Comparing RAG Context: (Simulated) Tensor DB vs. Vector DB for Financial News Impact")
st.caption("Illustrating how a (Simulated) Tensor Database like Tensorus can provide richer, multi-modal context for financial prediction RAG.")

# Initialize session state for tracking data ingestion
if 'finance_data_ingested' not in st.session_state:
    st.session_state.finance_data_ingested = False

st.sidebar.header("Demo Setup")
if not st.session_state.finance_data_ingested:
    st.sidebar.markdown("Click below to load and process the sample financial news data. This will populate the *simulated* Tensor Database (Tensorus-like) and Vector Database representations with various tensors derived from the news.")
    if st.sidebar.button("Load & Ingest Sample Financial Events into Simulated DBs"): # Clarified button text
        # Show a spinner during data ingestion
        with st.spinner("Processing news, generating tensors, and ingesting into simulated DBs... This may take a moment for NLP models."):
            ingest_financial_data()
        st.rerun() # Rerun to update UI based on new state (e.g., show success message)
else:
    st.sidebar.success("Sample financial data loaded into simulated DBs.")
    st.sidebar.markdown("You can re-ingest the data if needed. This will clear and re-populate the demo data in the simulated Tensor and Vector Databases.")
    if st.sidebar.button("Re-Ingest Data (Clears Simulated Demo Data)"): # Clarified button text
        st.session_state.ts_finance = EmbeddedTensorStorage() # Re-initialize the storage, clearing old data
        ts_finance = st.session_state.ts_finance # Update global reference
        with st.spinner("Re-processing and re-ingesting financial data into simulated DBs..."):
            ingest_financial_data()
        st.rerun()

# Main application area: only proceed if data has been ingested
if st.session_state.finance_data_ingested:
    st.header("Simulate RAG Context Retrieval for a Predictive Financial Model")
    st.markdown("Select a news event below. Then, retrieve context using two different simulated RAG approaches: one emulating a standard **Vector DB** and another emulating a **Tensor DB (like Tensorus)**. Observe and compare the richness of the context each approach would provide to a hypothetical financial prediction model.")
    
    # Create a mapping of event IDs to headlines for user-friendly selection in the selectbox
    event_options = {event["event_id"]: event["headline"] for event in NEWS_EVENTS_DATA}
    selected_event_id = st.selectbox(
        "Select a News Event to Analyze:",
        options=list(event_options.keys()), # Present event IDs as options for selection
        format_func=lambda x: f"{x}: {event_options[x]}" # Display as "ID: Headline" for readability
    )

    if selected_event_id:
        st.subheader(f"Analyzing Event: {event_options[selected_event_id]}")
        # Display the original news text for the selected event, for user reference
        original_news_text = next((e["full_text"] for e in NEWS_EVENTS_DATA if e["event_id"] == selected_event_id), "News text not found.")
        with st.expander("View Original News Text for Selected Event"):
            st.markdown(original_news_text)

        st.markdown("""
        ---
        **Hypothetical Predictive Model's Goal:**
        Imagine a sophisticated AI model tasked with predicting the financial impact of this selected news event (e.g., estimating percentage price change and volatility shifts for various affected assets).
        The quality, depth, and structure of the input context (retrieved via a RAG system) directly influence this model's potential for accurate and insightful predictions.
        """)

        # Create two columns for side-by-side comparison of VB and TD RAG approaches
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Simulated Vector DB (VB) RAG")
            st.caption("Traditional VB RAG typically retrieves a single embedding vector and an associated text chunk for the given event.")
            if st.button("Retrieve VB Context (Simulated)", key="vb_retrieve", help="Click to simulate retrieving context from a standard Vector Database."):
                with st.spinner("Retrieving context (Simulated VB style)..."):
                    vb_context = retrieve_context_for_rag(selected_event_id, approach="vector_db")
                
                st.markdown("**Context Provided to Predictive Model (Simulated VB):**")
                if vb_context["text_chunks"]:
                    st.info(f"**Retrieved Text Snippet:** \"{vb_context['text_chunks'][0]}...\"")
                if "overall_news_embedding (VB_sim)" in vb_context["structured_tensors"]: # Key updated for clarity
                    st.json({"overall_news_embedding (VB_sim)": vb_context["structured_tensors"]["overall_news_embedding (VB_sim)"]})
                st.markdown("*This context is primarily based on the **overall semantic similarity** of the news item, represented by a single vector. It offers a general summary but may lack specific details needed for nuanced financial predictions.*")

        with col2:
            st.subheader("Simulated Tensor DB (TD) RAG (Tensorus-like)")
            st.caption("A Tensor DB RAG (like the simulated Tensorus) retrieves multiple, richer tensor representations of the event and its surrounding context.")
            if st.button("Retrieve TD Context (Simulated)", key="td_retrieve", help="Click to simulate retrieving a multi-tensor context from a Tensor Database like Tensorus."):
                with st.spinner("Retrieving context (Simulated Tensorus style)..."):
                    td_context = retrieve_context_for_rag(selected_event_id, approach="tensor_db")

                st.markdown("**Context Provided to Predictive Model (Simulated Tensorus):**")
                if td_context["text_chunks"]: # Display the primary text snippet for reference
                     st.info(f"**Retrieved Primary Text Snippet:** \"{td_context['text_chunks'][0]}...\"")
                # Display all the different structured tensors retrieved from the simulated Tensor DB
                st.json(td_context["structured_tensors"]) 
                st.markdown("""
                *This richer, multi-faceted context (from the **simulated Tensor DB**) provides the predictive model with:*
                *   ***Detailed Sequential Information:*** From token-level embeddings (understanding how words are arranged and their specific meanings in context).
                *   ***Internal Text Relationships (Simulated):*** From the (simulated) attention flow tensor (indicating which parts of the text might relate to others).
                *   ***Structured Sentiment Indicators:*** Quantified sentiment beyond a simple positive/negative label (e.g., intensity, keyword counts).
                *   ***Prevailing Market State:*** Critical market conditions for relevant assets *at the precise time of the news event*.
                *This comprehensive, multi-modal input can empower a model to make more nuanced and accurate financial impact predictions.*
                """)
        
        st.divider()
        st.subheader("Why a (Simulated) Tensor DB like Tensorus Offers Superior Context for Prediction")
        st.markdown("""
        While this demo doesn't involve training an actual predictive model, the **critical difference for an advanced AI model** lies in the *nature, depth, and structure* of the input context it receives:

        *   **Simulated Vector DB RAG:**
            *   Provides a single, general semantic vector (essentially an "average" representation of the content) and perhaps a basic text chunk.
            *   The model is left to infer complex relationships, specific impacts on different assets, and the crucial influence of concurrent market conditions from this relatively limited, high-level view. It might capture the main sentiment but could easily miss vital details regarding *how* different parts of the news affect various assets or *how* the market was already positioned for those assets.

        *   **Simulated Tensor DB RAG (Tensorus-like approach):**
            *   Delivers a *multi-faceted tensor profile* of the event. This is a collection of distinct tensors, each representing a different aspect or modality of information, offering a much richer and more structured input:
                *   **Token Embeddings Tensor:** The *full sequence and nuanced meaning* of individual words or sub-words within the news text, not just a summary. This preserves local context and word order.
                *   **Attention Tensor (Simulated):** Insights into *which parts of the news are internally important* or interconnected (a real attention tensor from a sophisticated model would be even more powerful).
                *   **Sentiment Tensor:** *Structured, potentially multi-dimensional sentiment signals* (e.g., overall score, count of positive/negative keywords, aspect-specific sentiments), far more informative than a single categorical label.
                *   **Market Context Tensor:** The *actual, quantitative state of relevant financial assets (prices, volatility, volume trends, etc.) at the moment the news broke*.

        **An advanced predictive model, when supplied with the richer, more structured context from a (Simulated) Tensor DB like Tensorus, can potentially:**
        1.  **Understand News More Deeply:** Leverage detailed token sequences and attention mechanisms (even simulated ones) for a more granular and accurate comprehension of the textual information.
        2.  **Correlate News Nuances with Sentiment:** Link specific phrases or statements within the news (via token embeddings) to detailed, multi-dimensional sentiment features.
        3.  **Factor in Real-time Market Conditions:** Condition its predictions on the pre-existing market state for multiple assets, leading to more realistic and context-aware impact assessments. For example, the same news might have a very different impact if an asset is already highly volatile versus stable.
        4.  **Predict Complex, Interrelated Outcomes:** Potentially predict not just a single, isolated outcome (e.g., one stock price change), but a *vector or tensor of outcomes* (e.g., simultaneous impact on multiple assets, or changes in both price and future volatility, or even correlations between asset movements).

        This richer, structured, multi-modal input allows the predictive model to learn more complex patterns and make more accurate, specific, and robust financial forecasts compared to relying on a single, condensed document-level embedding.
        """)

else:
    # User guidance if data hasn't been loaded yet
    st.info("👈 Please use the sidebar to load and ingest the sample financial event data. This will enable the demo functionalities by populating the simulated databases.")