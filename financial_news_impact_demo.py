import streamlit as st
import torch
import pandas as pd
import numpy as np
import re
import json
# import uuid # No longer directly used here, moved to tensor_storage_utils
# import time # No longer directly used here for EmbeddedTensorStorage logic
import logging
from typing import List, Dict, Tuple, Optional, Callable, Any
# from collections import defaultdict # No longer directly used here
from transformers import AutoTokenizer, AutoModel, pipeline

from tensor_storage_utils import EmbeddedTensorStorage # Import the class

# Set page config must be the first Streamlit command
st.set_page_config(page_title="Financial News Impact RAG Demo", layout="wide")

# --- Configure basic logging ---
# Sets up basic logging for the application.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- NLP Models & Utilities (Load once using Streamlit's cache for efficiency) ---
@st.cache_resource # Caches the loaded models across Streamlit sessions/reruns, improving performance.
def load_nlp_models() -> Tuple[AutoTokenizer, AutoModel, pipeline]:
    """
    Loads and caches the NLP models required for the demo.

    This function initializes and returns a sentence transformer tokenizer and model,
    and a sentiment analysis pipeline. It uses Hugging Face's `transformers` library.
    Models are cached using Streamlit's `st.cache_resource` to prevent reloading
    on each script rerun, significantly improving performance.

    Returns:
        A tuple containing:
            - tokenizer (AutoTokenizer): The tokenizer for the sentence transformer model.
            - model (AutoModel): The sentence transformer model itself (e.g., 'all-MiniLM-L6-v2').
            - sentiment_analyzer (pipeline): A Hugging Face pipeline for sentiment analysis
                                             (e.g., 'distilbert-base-uncased-finetuned-sst-2-english').
    Raises:
        Exception: If any model fails to load, an error is logged, and the exception
                   is re-raised, which Streamlit will typically display.
                   Consider more specific exception handling for production.
    """
    try:
        # Using a lighter, general-purpose sentence transformer model for speed.
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        # Using a DistilBERT-based model for sentiment analysis, fine-tuned on SST-2.
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        logger.info("NLP models (tokenizer, sentence-transformer, sentiment_analyzer) loaded successfully.")
        return tokenizer, model, sentiment_analyzer
    except Exception as e:
        logger.error(f"Error loading NLP models: {e}", exc_info=True)
        st.error(f"Fatal Error: Could not load critical NLP models. Please check model names and internet connection. Details: {e}")
        # Re-raise the exception to halt execution if models are critical,
        # or handle more gracefully depending on application requirements.
        raise

# Attempt to load models globally; if this fails, the app should not proceed.
try:
    tokenizer, model, sentiment_analyzer = load_nlp_models()
except Exception:
    # Stop the Streamlit app if models can't be loaded.
    # Add a more user-friendly message or recovery mechanism if appropriate.
    st.stop()


def get_embedding(text: str, max_length: int = 128) -> Optional[torch.Tensor]:
    """
    Generates a single vector embedding (a sentence embedding) for a given text.

    This function uses mean pooling of token embeddings from the last hidden state
    of the pre-loaded sentence transformer model. The output is a 1D tensor
    representing the semantic meaning of the entire input text.

    Args:
        text: The input string to embed.
        max_length: The maximum token length for the tokenizer; texts longer
                    than this will be truncated.

    Returns:
        Optional[torch.Tensor]: A 1D torch.Tensor representing the sentence embedding
        (shape: [embedding_dimension], e.g., [384] for 'all-MiniLM-L6-v2').
        Returns None if an error occurs during embedding generation or if input is invalid.

    Side Effects:
        Logs an error via `logger` and displays an error via `st.error` if
        embedding generation fails or input is invalid.

    Role:
        Provides a compact, fixed-size representation of text, suitable for
        similarity comparisons or as a basic input feature for some machine
        learning models. This is typical for Vector DBs.
    """
    if not isinstance(text, str):
        logger.error("Invalid input: 'text' must be a string for get_embedding.")
        st.error("Embedding Error: Input text must be a string.")
        return None
    if not text.strip(): # Handle empty or whitespace-only strings
        logger.warning("Input text for get_embedding is empty or whitespace. Returning None.")
        # Depending on desired behavior, could return torch.zeros(model.config.hidden_size)
        # For this demo, returning None to indicate no meaningful embedding.
        return None

    try:
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length
        )
        with torch.no_grad(): # Disable gradient calculations during inference for efficiency.
            outputs = model(**inputs)
        # Mean pooling of the last hidden state to get a sentence embedding.
        # .squeeze() removes dimensions of size 1, resulting in a 1D tensor.
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding for text snippet '{text[:50]}...': {e}", exc_info=True)
        st.error(f"Could not generate text embedding. Please check the input or model. Error: {e}")
        return None

def get_token_level_embeddings(text: str, max_length: int = 128) -> Optional[torch.Tensor]:
    """
    Generates token-level embeddings for a given text.

    Each token in the input text is mapped to an embedding vector. The output is
    a 2D tensor where each row corresponds to a token's embedding.

    Args:
        text: The input string.
        max_length: The maximum sequence length. Texts will be padded or
                    truncated to this length.

    Returns:
        Optional[torch.Tensor]: A 2D torch.Tensor of shape (sequence_length, hidden_size)
        (e.g., (128, 384) for 'all-MiniLM-L6-v2' with max_length=128).
        'sequence_length' is the padded/truncated length.
        'hidden_size' is the dimensionality of each token embedding.
        Returns None if an error occurs or input is invalid.

    Side Effects:
        Logs an error via `logger` and displays an error via `st.error` if
        embedding generation fails or input is invalid.

    Role:
        Provides a detailed representation of the text, preserving sequence
        information and the individual meaning of tokens in context. This is a
        richer representation often stored in Tensor Databases for more complex modeling.
    """
    if not isinstance(text, str):
        logger.error("Invalid input: 'text' must be a string for get_token_level_embeddings.")
        st.error("Token Embedding Error: Input text must be a string.")
        return None
    if not text.strip():
        logger.warning("Input text for get_token_level_embeddings is empty or whitespace. Returning None.")
        return None

    try:
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
            return_attention_mask=True # Good practice for models that use it
        )
        with torch.no_grad():
            outputs = model(**inputs)
        # outputs.last_hidden_state is [batch_size, sequence_length, hidden_size].
        # .squeeze(0) removes the batch_dim (assuming batch_size is 1 for single text input).
        token_embeddings = outputs.last_hidden_state.squeeze(0)
        # Note: This returns embeddings for all tokens up to max_length, including padding tokens.
        # Downstream models should use an attention mask if they need to distinguish actual tokens from padding.
        return token_embeddings
    except Exception as e:
        logger.error(f"Error generating token-level embeddings for text snippet '{text[:50]}...': {e}", exc_info=True)
        st.error(f"Could not generate token-level embeddings. Error: {e}")
        return None

def get_simplified_attention_flow(text: str, max_length: int = 32) -> Optional[torch.Tensor]:
    """
    Simulates a basic attention flow matrix for a given text.

    **This is a simplified heuristic and NOT a real attention map from a transformer model.**
    It's intended to represent, for demonstration, how different parts of the text
    might conceptually relate to each other. A real attention map would be derived
    from a model's internal attention mechanisms.

    Args:
        text: The input string.
        max_length: The maximum number of tokens to consider for the attention matrix.
                   The output matrix will be (max_length x max_length).

    Returns:
        Optional[torch.Tensor]: A 2D torch.Tensor of shape (max_length, max_length)
        representing the simulated attention scores. Higher values suggest stronger
        (simulated) attention between token pairs (token i to token j).
        Returns a zero tensor of the same shape if the text is empty or tokenization fails,
        or None if a more significant error occurs or input is invalid.

    Side Effects:
        Logs an error via `logger` and displays an error via `st.error` if
        tensor generation fails or input is invalid.

    Role:
        Illustrates the concept of an attention mechanism, which can highlight
        important relationships or dependencies between different parts of a text.
        In a real Tensor DB scenario, actual attention tensors from a model could
        be stored to provide insights into text structure.
    """
    if not isinstance(text, str):
        logger.error("Invalid input: 'text' must be a string for get_simplified_attention_flow.")
        st.error("Attention Flow Error: Input text must be a string.")
        return None
    
    try:
        # Tokenize the text using the global tokenizer.
        # `tokenizer.tokenize` returns a list of token strings.
        tokens = tokenizer.tokenize(text)

        # Truncate tokens if they exceed max_length before further processing.
        if len(tokens) > max_length:
            tokens = tokens[:max_length]

        num_tokens = len(tokens)

        if num_tokens == 0:
            logger.warning("No tokens found in text for simplified attention flow. Returning zero tensor.")
            return torch.zeros((max_length, max_length)) # Return zero matrix if no tokens.

        # Simple heuristic: words closer together get higher "attention".
        # This is a placeholder for the complex calculations in a real transformer's attention.
        attention_matrix = torch.zeros((num_tokens, num_tokens))
        for i in range(num_tokens):
            for j in range(num_tokens):
                distance = abs(i - j)
                # Assign higher scores to pairs of tokens that are closer to each other.
                # Add 1.0 to distance to prevent division by zero if i == j.
                attention_matrix[i, j] = 1.0 / (1.0 + distance)
    
        # Pad the attention matrix to ensure it has dimensions (max_length, max_length) for consistency.
        if num_tokens < max_length:
            padded_attention = torch.zeros((max_length, max_length))
            padded_attention[:num_tokens, :num_tokens] = attention_matrix
            return padded_attention
        return attention_matrix
    except Exception as e:
        logger.error(f"Error generating simplified attention flow for text snippet '{text[:50]}...': {e}", exc_info=True)
        st.error(f"Could not generate simplified attention flow. Error: {e}")
        return None


# --- Sample Financial Data (Illustrative) ---
# Defines the assets involved in the hypothetical news scenarios.
# Used for context in the demo.
# Defines the assets involved in the hypothetical news scenarios. These are used for context in the demo.
ASSETS = ["TechCorp (TC)", "InnovateInc (II)", "GlobalWidgets (GW)"]

# A list of dictionaries, each representing a financial news event with associated data.
# This data is used to populate the simulated Tensor Database and Vector Database.
NEWS_EVENTS_DATA = [
    {
        "event_id": "EVT001", # Unique identifier for the event
        "timestamp": pd.to_datetime("2025-05-19 09:00:00").timestamp(), # Use .timestamp() for POSIX timestamp
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
        "timestamp": pd.to_datetime("2025-05-19 11:30:00").timestamp(), # Use .timestamp()
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
    Processes the sample `NEWS_EVENTS_DATA` and ingests it into the
    simulated TensorStorage (`ts_finance`).

    Data is ingested in two ways to contrast Vector DB and Tensor DB approaches:
    1.  **Vector DB (VB) style:** Stores one global embedding per news event.
        This is stored in the `VB_NEWS_EMBEDDINGS_DS` dataset.
    2.  **Tensor DB (TD) style (Simulating Tensorus):** Stores multiple, diverse
        tensors per news event, capturing different facets like token sequences,
        attention, sentiment, and market context. These are stored in various
        `TD_..._DS` datasets.

    This function iterates through `NEWS_EVENTS_DATA`, processes each event to
    generate the necessary tensors using helper functions like `get_embedding`,
    `get_token_level_embeddings`, etc., and then inserts them into the
    appropriate datasets in `ts_finance`.

    Side Effects:
        - Modifies `ts_finance` by inserting data into its datasets.
        - Logs information about the ingestion process using `logger`.
        - Updates `st.session_state.finance_data_ingested` to `True` on successful completion.
        - Displays a success message in the Streamlit sidebar (`st.sidebar.success`).
        - Can display error messages (`st.error`) if tensor generation or insertion fails.
    """
    try:
        logger.info("Starting data ingestion process for financial news events into simulated storage.")
        if not NEWS_EVENTS_DATA:
            st.warning("NEWS_EVENTS_DATA is empty. No data to ingest.")
            logger.warning("NEWS_EVENTS_DATA is empty. Ingestion skipped.")
            return

        # --- Ingestion for Vector DB (VB) Approach Simulation ---
        # For each news event, create and store a single embedding representing the entire news text.
        for event_idx, event in enumerate(NEWS_EVENTS_DATA):
            try:
                # Validate essential keys in event data
                required_keys = ["full_text", "event_id", "headline", "timestamp", "future_impact_simulated"]
                if not all(key in event for key in required_keys):
                    logger.error(f"Event at index {event_idx} is missing one or more required keys: {required_keys}. Skipping.")
                    st.error(f"Skipping event (index {event_idx}) due to missing data: {', '.join(k for k in required_keys if k not in event)}")
                    continue # Skip this event if critical data is missing

                news_embedding = get_embedding(event["full_text"])
                if news_embedding is None: # Check if embedding generation failed
                    logger.error(f"Failed to generate embedding for event {event.get('event_id', 'Unknown_ID_VB')}. Skipping VB insertion.")
                    st.error(f"Embedding generation failed for event: {event.get('event_id', 'Unknown_ID_VB')}. VB data may be incomplete.")
                    continue # Skip VB insertion for this event

                metadata = {
                    "event_id": event["event_id"],
                    "headline": event["headline"],
                    "timestamp_utc": event["timestamp"],
                    "raw_text_snippet": event["full_text"][:200], # Store a snippet for context
                    "_future_impact_simulated": event["future_impact_simulated"] # For demo comparison
                }
                ts_finance.insert(VB_NEWS_EMBEDDINGS_DS, news_embedding, metadata)
            except KeyError as ke:
                logger.error(f"KeyError accessing data for event {event.get('event_id', 'Unknown_ID_VB_KeyErr')} during VB ingestion: {ke}", exc_info=True)
                st.error(f"Data error for event {event.get('event_id', 'Unknown_ID_VB_KeyErr')} (VB): {ke}. Check data structure.")
                continue # Skip to next event
            except Exception as e: # Catch any other unexpected error during VB processing for one event
                logger.error(f"Error processing event {event.get('event_id', 'Unknown_ID_VB_Err')} for VB: {e}", exc_info=True)
                st.error(f"Unexpected error for event {event.get('event_id', 'Unknown_ID_VB_Err')} (VB): {e}")
                continue # Skip to next event

        logger.info(f"Successfully ingested {len(NEWS_EVENTS_DATA)} events into '{VB_NEWS_EMBEDDINGS_DS}' for VB simulation.")

        # --- Ingestion for Tensor Database (TD) Approach Simulation ---
        # For each news event, create and store multiple, diverse tensors.
        for event_idx, event in enumerate(NEWS_EVENTS_DATA):
            try:
                # Validate essential keys for TD processing
                required_keys_td = ["event_id", "headline", "timestamp", "full_text",
                                    "market_context_features", "_future_impact_simulated"]
                if not all(key in event for key in required_keys_td):
                    logger.error(f"Event at index {event_idx} for TD is missing keys: {required_keys_td}. Skipping TD part.")
                    st.error(f"Skipping TD processing for event (index {event_idx}) due to missing data: {', '.join(k for k in required_keys_td if k not in event)}")
                    continue

                event_id = event["event_id"]
                # Common metadata to be associated with all tensors derived from this specific news event.
                common_metadata = {
                    "event_id": event_id,
                    "headline": event["headline"],
                    "timestamp_utc": event["timestamp"],
                    "raw_text_snippet_for_context": event["full_text"][:200], # Snippet for quick reference
                    "_future_impact_simulated": event["_future_impact_simulated"] # Ground truth for demo
                }

                # Token Embeddings
                token_embeds = get_token_level_embeddings(event["full_text"])
                if token_embeds is not None:
                    ts_finance.insert(TD_NEWS_TOKEN_EMBEDDINGS_DS, token_embeds, {**common_metadata, "tensor_type": "token_embeddings"})
                else:
                    logger.warning(f"Token embeddings not generated for event {event_id}. Skipping TD_NEWS_TOKEN_EMBEDDINGS_DS insertion.")
                    st.warning(f"Token embeddings failed for {event_id}. TD data may be incomplete.")

                # Attention Flow
                attention_tensor = get_simplified_attention_flow(event["full_text"], max_length=32)
                if attention_tensor is not None:
                    ts_finance.insert(TD_NEWS_ATTENTION_DS, attention_tensor, {**common_metadata, "tensor_type": "attention_flow_simulated"})
                else:
                    logger.warning(f"Attention flow not generated for event {event_id}. Skipping TD_NEWS_ATTENTION_DS insertion.")
                    st.warning(f"Attention flow failed for {event_id}. TD data may be incomplete.")

                # Sentiment Features
                try:
                    # Ensure 'headline' and 'full_text' exist before processing sentiment
                    headline_text = event.get("headline")
                    full_text_content = event.get("full_text")
                    if not headline_text or not full_text_content:
                        raise ValueError("Missing headline or full_text for sentiment analysis.")

                    sentiment_result = sentiment_analyzer(headline_text)[0] # Analyze headline
                    overall_score = (1 if sentiment_result['label'] == 'POSITIVE' else -1) * sentiment_result['score']
                    # Simple keyword counts as additional sentiment features
                    num_pos_keywords = len(re.findall(r'good|great|breakthrough|surges|gains|advances', full_text_content, re.IGNORECASE))
                    num_neg_keywords = len(re.findall(r'bad|poor|concerns|tumbles|dip|falls|regulatory', full_text_content, re.IGNORECASE))
                    sentiment_data = torch.tensor([overall_score, float(num_pos_keywords), float(num_neg_keywords)])
                    ts_finance.insert(TD_NEWS_SENTIMENT_DS, sentiment_data, {**common_metadata, "tensor_type": "sentiment_features"})
                except (KeyError, TypeError, ValueError) as e_sent_prep: # Catch errors from data prep for sentiment
                    logger.error(f"Data preparation error for sentiment analysis of event {event_id}: {e_sent_prep}", exc_info=True)
                    st.error(f"Sentiment data prep failed for {event_id}: {e_sent_prep}")
                except Exception as e_sent: # Catch errors during sentiment_analyzer call or tensor creation
                    logger.error(f"Error processing sentiment for event {event_id}: {e_sent}", exc_info=True)
                    st.error(f"Sentiment analysis failed for {event_id}: {e_sent}")

                # Market Context
                try:
                    market_features = event.get("market_context_features")
                    if market_features is None or not isinstance(market_features, dict): # Check if key exists and is a dict
                        raise ValueError("market_context_features is missing or not a dictionary.")

                    market_features_list = []
                    # Sort asset names to ensure consistent order of features in the tensor
                    ordered_assets_for_context = sorted(market_features.keys())
                    if not ordered_assets_for_context: # Handle case where market_context_features is empty dict
                        logger.warning(f"No assets found in market_context_features for event {event_id}. Market context tensor will be empty/skipped.")

                    for asset in ordered_assets_for_context:
                        features = market_features.get(asset, {}) # Use .get for safety
                        # Ensure all expected sub-keys are present with defaults
                        price = features.get("price", 0.0)
                        volatility = features.get("volatility", 0.0)
                        volume_spike = features.get("volume_spike", 0.0)
                        market_features_list.extend([price, volatility, volume_spike])

                    if market_features_list: # Only insert if there's actual data
                        market_context_tensor = torch.tensor(market_features_list, dtype=torch.float32)
                        ts_finance.insert(TD_MARKET_CONTEXT_DS, market_context_tensor, {**common_metadata, "tensor_type": "market_context", "context_asset_order": ordered_assets_for_context})
                    else:
                        # This can be normal if an event has no market context features defined.
                        logger.info(f"Market features list is empty for {event_id}; market context tensor not created.")

                except KeyError as ke_mkt: # Should be caught by .get now, but as a safeguard
                    logger.error(f"KeyError accessing market_context_features for event {event_id}: {ke_mkt}", exc_info=True)
                    st.error(f"Market context data error for {event_id}: Missing key {ke_mkt}.")
                except ValueError as ve_mkt: # For custom error like missing market_context_features
                    logger.error(f"ValueError in market_context_features for event {event_id}: {ve_mkt}", exc_info=True)
                    st.error(f"Market context data issue for {event_id}: {ve_mkt}.")
                except Exception as e_mkt: # Catch any other errors during market context processing
                    logger.error(f"Error processing market context for event {event_id}: {e_mkt}", exc_info=True)
                    st.error(f"Market context processing failed for {event_id}: {e_mkt}")

            except KeyError as ke_outer: # Catch KeyErrors from main event structure (e.g. event["event_id"])
                 logger.error(f"KeyError accessing data for event (index {event_idx}) during TD ingestion: {ke_outer}", exc_info=True)
                 st.error(f"Data error for event (index {event_idx}) (TD): {ke_outer}. Check data structure.")
                 continue # Skip to next event
            except Exception as e_outer: # Catch other errors for the event
                logger.error(f"Error processing event (index {event_idx}) for TD: {e_outer}", exc_info=True)
                st.error(f"Unexpected error for event (index {event_idx}) (TD): {e_outer}")
                continue # Skip to next event

        logger.info(f"Successfully ingested multi-faceted tensor representations for {len(NEWS_EVENTS_DATA)} events for TD simulation.")

    except Exception as e: # Catch-all for unexpected errors during the entire ingestion function
        st.error(f"A critical error occurred during the data ingestion process: {e}")
        logger.critical(f"Data ingestion process failed: {e}", exc_info=True)
        st.session_state.finance_data_ingested = False # Mark as not ingested on critical failure
        return # Stop ingestion

    st.session_state.finance_data_ingested = True
    st.sidebar.success("Financial event data ingestion complete. (Simulated Tensorus & VB representations).")


# --- Retrieval Logic for RAG (Retrieval Augmented Generation) ---
def retrieve_context_for_rag(event_id: str, approach: str = "vector_db") -> Dict[str, Any]:
    """
    Simulates the retrieval of context for a given news event ID, using either a
    Vector DB approach or a Tensor DB (simulating Tensorus) approach.

    This function demonstrates what kind of data would be fed into a downstream
    prediction model based on the chosen retrieval strategy.

    Args:
        event_id: The ID of the news event to retrieve context for.
        approach: "vector_db" or "tensor_db", specifying the retrieval strategy.
                        Defaults to "vector_db".

    Returns:
        A dictionary containing the retrieved context.
        This context typically includes:
        - "text_chunks" (List[str]): List of relevant text snippets.
        - "structured_tensors" (Dict[str, Any]): Dictionary of retrieved tensors,
          their shapes, and previews or comments.
        - "metadata" (Dict[str, Any]): Associated metadata for the event.

    Side Effects:
        - Logs warnings or errors via `logger` and `st.error` if issues occur
          (e.g., event_id not found, dataset issues).
        - Displays success messages via `st.success` upon successful retrieval
          of different tensor types in the "tensor_db" approach.
        - Uses `st.write` and `st.info` for informational messages in the UI.
    """
    retrieved_context: Dict[str, Any] = {
        "text_chunks": [],
        "structured_tensors": {},
        "metadata": {}
    }
    
    if not event_id: # Basic validation
        st.error("Event ID cannot be empty for retrieval.")
        logger.warning("retrieve_context_for_rag called with empty event_id.")
        return retrieved_context

    try:
        if approach == "vector_db":
            # --- Vector DB Retrieval Simulation ---
            st.write(f"**Simulated Vector DB Approach: Retrieving context for Event ID: {event_id}**")

            # Define filter function for metadata
            def vb_filter_fn(meta: Dict[str, Any]) -> bool:
                return meta.get("event_id") == event_id

            records = ts_finance.get_records_by_metadata_filter(
                VB_NEWS_EMBEDDINGS_DS, vb_filter_fn
            )

            if records:
                record = records[0] # Expecting one record per event_id for VB approach

                # Safely access metadata and tensor data
                record_metadata = record.get("metadata", {})
                record_tensor = record.get("tensor")

                retrieved_context["text_chunks"].append(record_metadata.get("raw_text_snippet", "Snippet not available."))

                if record_tensor is not None:
                    retrieved_context["structured_tensors"]["overall_news_embedding (VB_sim)"] = {
                        "shape": list(record_tensor.shape),
                        "preview": record_tensor[:5].tolist() # Show first 5 dimensions as a preview
                    }
                else:
                    logger.warning(f"No tensor found for event {event_id} in VB records, though metadata exists.")
                    retrieved_context["structured_tensors"]["overall_news_embedding (VB_sim)"] = {
                        "shape": "N/A", "preview": "Tensor data missing"
                    }

                retrieved_context["metadata"] = record_metadata
                st.success("Retrieved: Main text snippet and its overall semantic embedding.")
            else:
                st.error(f"Event ID {event_id} not found in the simulated Vector DB representation ('{VB_NEWS_EMBEDDINGS_DS}').")
                logger.warning(f"No records found for event_id '{event_id}' in dataset '{VB_NEWS_EMBEDDINGS_DS}'.")

        elif approach == "tensor_db":
            # --- Tensor DB (Tensorus) Retrieval Simulation ---
            st.write(f"**Simulated Tensor DB (Tensorus) Approach: Retrieving multi-faceted context for Event ID: {event_id}**")
            event_found_in_any_td_dataset = False # Flag to track if any data is found for the event

            # Define a reusable filter function for metadata
            def td_filter_fn(meta: Dict[str, Any]) -> bool:
                return meta.get("event_id") == event_id

            # Helper function to process and add a tensor to the context
            def _add_tensor_to_context(
                dataset_name: str,
                tensor_key_name: str,
                comment: str,
                include_data: bool = False
            ) -> bool:
                nonlocal event_found_in_any_td_dataset, retrieved_context
                try:
                    records = ts_finance.get_records_by_metadata_filter(dataset_name, td_filter_fn)
                    if records:
                        event_found_in_any_td_dataset = True
                        record = records[0] # Assuming one matching record for this event_id in the dataset
                        record_meta = record.get("metadata", {})
                        record_tensor = record.get("tensor")

                        # Use metadata from the first successfully retrieved tensor as the base
                        if not retrieved_context["metadata"]:
                            retrieved_context["metadata"] = record_meta
                        # Add text snippet if not already present (prefer token embeddings' context)
                        if not retrieved_context["text_chunks"] and "raw_text_snippet_for_context" in record_meta:
                            retrieved_context["text_chunks"].append(record_meta["raw_text_snippet_for_context"])

                        tensor_info_payload: Dict[str, Any] = {"comment": comment}
                        if record_tensor is not None:
                            tensor_info_payload["shape"] = list(record_tensor.shape)
                            if include_data:
                                tensor_info_payload["data"] = record_tensor.tolist()
                            # Add specific metadata if relevant (e.g., asset_order for market context)
                            if "context_asset_order" in record_meta: # Check if key exists
                                 tensor_info_payload["asset_order"] = record_meta["context_asset_order"]
                            retrieved_context["structured_tensors"][tensor_key_name] = tensor_info_payload
                            st.success(f"Retrieved: {tensor_key_name.split('(')[0].strip()} tensor.") # Clean key name for message
                            return True
                        else:
                            logger.warning(f"Tensor data missing for {tensor_key_name} in event {event_id} from {dataset_name}.")
                            retrieved_context["structured_tensors"][tensor_key_name] = {
                                "shape": "N/A", "comment": f"{comment} - Data Missing"
                            }
                    else: # No records found for this event_id in this dataset
                        # This is not necessarily an error, could be expected for some events/datasets
                        logger.info(f"No records for event {event_id} in dataset {dataset_name} during TD retrieval.")
                except Exception as e_ctx: # Catch errors during a specific tensor retrieval
                    logger.error(f"Error retrieving from {dataset_name} for event {event_id}: {e_ctx}", exc_info=True)
                    st.error(f"Failed to retrieve {tensor_key_name.split('(')[0].strip()} for {event_id}: {e_ctx}")
                return False

            # Call helper for each tensor type in Tensor DB approach
            _add_tensor_to_context(
                TD_NEWS_TOKEN_EMBEDDINGS_DS,
                "token_embeddings (TD_sim)",
                "Captures the sequence and nuance of the news text."
            )
            _add_tensor_to_context(
                TD_NEWS_ATTENTION_DS,
                "attention_flow_sim (TD_sim)",
                "Simulated tensor highlighting internal text relationships."
            )
            _add_tensor_to_context(
                TD_NEWS_SENTIMENT_DS,
                "sentiment_features (TD_sim)",
                "Structured sentiment: [OverallScore, PositiveKeywordsCount, NegativeKeywordsCount]",
                include_data=True
            )
            _add_tensor_to_context(
                TD_MARKET_CONTEXT_DS,
                "market_context (TD_sim)",
                "Market state (price, volatility, volume spike) for relevant assets at news time.",
                include_data=True
            )

            if not event_found_in_any_td_dataset:
                st.error(f"Event ID {event_id} not found in any of the simulated Tensor DB (Tensorus) representations.")
                logger.warning(f"No TD records found for event_id '{event_id}' across relevant datasets.")
            elif event_found_in_any_td_dataset:
                # This message highlights the benefit of the Tensor DB (Tensorus) approach.
                st.info("The combination of these retrieved tensors provides a richer, multi-modal input for a predictive model.")
        else: # Should not happen if UI restricts choices, but good for robustness
            st.error(f"Invalid retrieval approach specified: {approach}. Must be 'vector_db' or 'tensor_db'.")
            logger.error(f"Invalid retrieval approach: {approach} for event_id '{event_id}'.")

    except Exception as e: # Catch-all for any other unexpected error during retrieval
        st.error(f"An unexpected error occurred during context retrieval for event ID '{event_id}': {e}")
        logger.error(f"Context retrieval failed for event_id '{event_id}', approach '{approach}': {e}", exc_info=True)
        # Ensure partial context is still returned or cleared based on desired behavior
        # retrieved_context = {"text_chunks": [], "structured_tensors": {}, "metadata": {}} # Optionally clear

    return retrieved_context

# --- Streamlit UI ---
# Updated title and caption to emphasize simulation
# Updated title and caption to emphasize simulation
st.title("💸 Comparing RAG Context: (Simulated) Tensor DB vs. Vector DB for Financial News Impact")
st.caption("Illustrating how a (Simulated) Tensor Database like Tensorus can provide richer, multi-modal context for financial prediction RAG.")

# Initialize session state for tracking data ingestion status.
if 'finance_data_ingested' not in st.session_state:
    st.session_state.finance_data_ingested = False

st.sidebar.header("Demo Setup")

# Conditional UI for data ingestion based on session state.
if not st.session_state.get('finance_data_ingested', False): # Use .get for safety if key might be missing
    st.sidebar.markdown(
        "Click below to load and process the sample financial news data. "
        "This will populate the *simulated* Tensor Database (Tensorus-like) "
        "and Vector Database representations with various tensors derived from the news."
    )
    if st.sidebar.button("Load & Ingest Sample Financial Events into Simulated DBs"):
        # Show a spinner during data ingestion for better UX.
        with st.spinner("Processing news, generating tensors, and ingesting... This may take a moment."):
            ingest_financial_data()
            # If ingest_financial_data updates session_state.finance_data_ingested,
            # Streamlit will automatically rerun and reflect the change.
        # Explicitly rerun if ingest_financial_data doesn't always trigger it (e.g., if it fails early)
        # or to ensure UI updates immediately after the spinner completes.
        st.rerun()
else:
    st.sidebar.success("Sample financial data loaded into simulated DBs.")
    st.sidebar.markdown(
        "You can re-ingest the data if needed. This will clear and "
        "re-populate the demo data in the simulated Tensor and Vector Databases."
    )
    if st.sidebar.button("Re-Ingest Data (Clears Simulated Demo Data)"):
        try:
            # Re-initialize the storage, clearing old data.
            st.session_state.ts_finance = EmbeddedTensorStorage()
            # Update the global reference if it's used elsewhere, ensuring consistency.
            # This line might be redundant if ts_finance is always accessed via st.session_state.ts_finance
            ts_finance = st.session_state.ts_finance
            st.session_state.finance_data_ingested = False # Mark as not ingested before starting
            with st.spinner("Re-processing and re-ingesting financial data..."):
                ingest_financial_data()
            # Rerun to refresh the UI state after ingestion.
            st.rerun()
        except Exception as e_reingest: # Catch broad exceptions during re-ingestion
            st.error(f"Error during re-ingestion: {e_reingest}")
            logger.error(f"Re-ingestion failed: {e_reingest}", exc_info=True)


# Main application area: only proceed if data has been ingested.
if st.session_state.get('finance_data_ingested', False): # Use .get for safety
    st.header("Simulate RAG Context Retrieval for a Predictive Financial Model")
    st.markdown(
        "Select a news event below. Then, retrieve context using two different "
        "simulated RAG approaches: one emulating a standard **Vector DB** and "
        "another emulating a **Tensor DB (like Tensorus)**. Observe and compare "
        "the richness of the context each approach would provide to a "
        "hypothetical financial prediction model."
    )

    # Create a mapping of event IDs to headlines for user-friendly selection.
    # Handle potential errors if NEWS_EVENTS_DATA is not as expected.
    try:
        event_options = {
            event["event_id"]: event.get("headline", "No headline available") # Use .get for headline
            for event in NEWS_EVENTS_DATA if "event_id" in event # Ensure event_id exists
        }
        if not event_options:
            st.warning("No news events available for selection. Check NEWS_EVENTS_DATA structure or content.")
            # Stop further execution in this block if no options.
            st.stop() # Avoids errors if event_options is empty

        selected_event_id = st.selectbox(
            "Select a News Event to Analyze:",
            options=list(event_options.keys()), # Present event IDs
            format_func=lambda x: f"{x}: {event_options.get(x, 'Unknown Event')}" # Display "ID: Headline"
        )
    except (TypeError, AttributeError, KeyError) as e_event_options: # Catch errors during event_options creation
        st.error(f"Error preparing event selection: {e_event_options}. NEWS_EVENTS_DATA might be malformed.")
        logger.error(f"Failed to create event_options from NEWS_EVENTS_DATA: {e_event_options}", exc_info=True)
        st.stop() # Halt if event selection cannot be prepared.


    if selected_event_id: # Proceed only if an event is selected
        # Display the selected event's headline and original text.
        selected_event_headline = event_options.get(selected_event_id, "Headline not found")
        st.subheader(f"Analyzing Event: {selected_event_headline}")

        try:
            # Find the full text of the selected event, with robust error handling.
            original_news_text = next(
                (e.get("full_text", "Full text not available.") # Use .get for full_text
                 for e in NEWS_EVENTS_DATA if e.get("event_id") == selected_event_id), # Use .get for event_id
                "News text not found for this event." # Default if event_id not found
            )
        except (TypeError, AttributeError, KeyError) as e_news_text: # Catch errors if NEWS_EVENTS_DATA is not a list of dicts
            original_news_text = "Error retrieving news text due to data structure issue."
            st.error(f"Could not retrieve news text: {e_news_text}")
            logger.error(f"Error getting original_news_text for {selected_event_id}: {e_news_text}", exc_info=True)

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
                # Safely access context elements and provide defaults or messages if missing
                vb_text_chunks = vb_context.get("text_chunks", [])
                if vb_text_chunks:
                    # Display a snippet, ensuring it's a string and handling potential None
                    snippet = str(vb_text_chunks[0]) if vb_text_chunks[0] is not None else "Empty snippet"
                    st.info(f"**Retrieved Text Snippet:** \"{snippet[:300]}...\"") # Truncate for display
                else:
                    st.info("No text snippets retrieved for VB context.")

                vb_tensors = vb_context.get("structured_tensors", {})
                if "overall_news_embedding (VB_sim)" in vb_tensors:
                    st.json({"overall_news_embedding (VB_sim)": vb_tensors["overall_news_embedding (VB_sim)"]})
                else:
                    st.warning("Overall news embedding not found in VB context.")

                st.markdown(
                    "*This context is primarily based on the **overall semantic similarity** "
                    "of the news item, represented by a single vector. It offers a general "
                    "summary but may lack specific details needed for nuanced financial predictions.*"
                )

        with col2:
            st.subheader("Simulated Tensor DB (TD) RAG (Tensorus-like)")
            st.caption(
                "A Tensor DB RAG (like the simulated Tensorus) retrieves multiple, "
                "richer tensor representations of the event and its surrounding context."
            )
            if st.button(
                "Retrieve TD Context (Simulated)",
                key="td_retrieve", # Unique key for this button
                help="Click to simulate retrieving a multi-tensor context from a Tensor Database like Tensorus."
            ):
                with st.spinner("Retrieving context (Simulated Tensorus style)..."):
                    td_context = retrieve_context_for_rag(selected_event_id, approach="tensor_db")

                st.markdown("**Context Provided to Predictive Model (Simulated Tensorus):**")
                # Safely access context elements
                td_text_chunks = td_context.get("text_chunks", [])
                if td_text_chunks:
                    snippet = str(td_text_chunks[0]) if td_text_chunks[0] is not None else "Empty snippet"
                    st.info(f"**Retrieved Primary Text Snippet:** \"{snippet[:300]}...\"") # Truncate
                else:
                    st.info("No text snippets retrieved for TD context.")

                td_tensors = td_context.get("structured_tensors", {})
                if td_tensors: # Check if dictionary is not empty
                    st.json(td_tensors)
                else:
                    st.warning("No structured tensors found in TD context.") # Message if empty

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