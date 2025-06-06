import streamlit as st
import torch
import pandas as pd
import numpy as np
import re
import json # Added for metadata parsing in UI
# import uuid # No longer directly used here, moved to tensor_storage_utils
# import time # No longer directly used here for EmbeddedTensorStorage logic
import logging # For TensorStorage logging
from typing import List, Dict, Tuple, Optional, Callable, Any # Expanded for TensorStorage

from tensor_storage_utils import EmbeddedTensorStorage # Import the class

# --- Streamlit page config must be set first ---
st.set_page_config(page_title="Story Analyzer (Tensorus Concept)", layout="wide") # Adjusted title for clarity

# --- Configure basic logging (optional, but good for the embedded TensorStorage) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- NLTK Data Download ---
# Ensures necessary NLTK resources are available.
try:
    import nltk
    import urllib.error # For catching network errors during NLTK download

    required_nltk_data = ['punkt', 'stopwords']
    nltk_data_doc_url = "https://www.nltk.org/data.html"
    manual_download_cmd = "python -m nltk.downloader punkt stopwords"

    for data_package in required_nltk_data:
        try:
            # Check if the resource is already available
            # NLTK's find checks specific paths; for tokenizers, it's 'tokenizers/<package>'
            resource_path = f'tokenizers/{data_package}' if data_package == 'punkt' else f'corpora/{data_package}'
            if data_package == 'stopwords': resource_path = f'corpora/{data_package}' # stopwords is in corpora

            nltk.data.find(resource_path)
            logger.info(f"NLTK resource '{data_package}' found at '{resource_path}'.")
        except LookupError:
            logger.info(f"NLTK resource '{data_package}' not found. Attempting download...")
            st.info(f"Downloading NLTK resource '{data_package}' (this may take a moment)...")
            try:
                nltk.download(data_package, quiet=True)
                logger.info(f"NLTK resource '{data_package}' downloaded successfully.")
                # Verify after download
                nltk.data.find(resource_path)
                logger.info(f"NLTK resource '{data_package}' verified after download.")
            except (OSError, urllib.error.URLError) as e_download: # Catch network/filesystem errors during download
                error_message = (
                    f"Failed to download NLTK data package '{data_package}'. This might be due to network issues or file permission problems. "
                    f"Please check your internet connection and ensure NLTK can write to its data directory. "
                    f"You can also try downloading manually: `{manual_download_cmd}`. "
                    f"For more information, see: {nltk_data_doc_url}"
                )
                logger.error(f"{error_message} (Error: {e_download})", exc_info=True)
                st.error(error_message)
                st.stop() # Critical for app functionality
            except Exception as e_unknown_download: # Catch any other unexpected error during download
                logger.error(f"An unexpected error occurred while downloading NLTK data '{data_package}': {e_unknown_download}", exc_info=True)
                st.error(f"Unexpected error downloading NLTK resource '{data_package}': {e_unknown_download}")
                st.stop()
        except Exception as e_find: # Catch errors from nltk.data.find itself (other than LookupError)
             logger.error(f"Error finding NLTK resource '{data_package}': {e_find}", exc_info=True)
             st.error(f"Could not verify NLTK resource '{data_package}': {e_find}")
             st.stop()


except ImportError:
    # NLTK itself is not installed
    st.error("NLTK package not found. Please install it by running: pip install nltk")
    logger.error("NLTK package not found critical error.")
    st.stop() # Stop the app if NLTK is missing
except Exception as e_nltk_general:
    # Catch any other top-level errors during NLTK setup phase
    st.error(f"A critical error occurred during NLTK setup: {e_nltk_general}. The application cannot proceed.")
    logger.error(f"NLTK general setup error: {e_nltk_general}", exc_info=True)
    st.stop() # Stop the app if NLTK data can't be set up

# --- NLP Model Loading ---
# Encapsulated and cached for efficiency.
@st.cache_resource
def load_nlp_models() -> Tuple[pipeline, AutoTokenizer, AutoModel]:
    """
    Loads and caches the NLP models required for the demo.

    This includes a sentiment analysis pipeline and a sentence transformer
    (tokenizer and model) for generating embeddings. Models are cached to
    prevent reloading on each Streamlit script rerun.

    Returns:
        A tuple containing:
            - sentiment_analyzer (pipeline): Hugging Face pipeline for sentiment analysis.
            - embedding_tokenizer (AutoTokenizer): Tokenizer for sentence embeddings.
            - embedding_model (AutoModel): Model for sentence embeddings.

    Raises:
        RuntimeError: If any of the models fail to load, indicating a critical issue.
    """
    try:
        logger.info("Loading NLP models...")
        # Using DistilBERT for sentiment analysis (fine-tuned on SST-2)
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        # Using a common sentence transformer model for generating embeddings
        embedding_tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        embedding_model = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info("NLP models (sentiment_analyzer, embedding_tokenizer, embedding_model) loaded successfully.")
        return sentiment_analyzer, embedding_tokenizer, embedding_model
    except OSError as e:
        offline_docs_url = "https://huggingface.co/docs/transformers/installation#offline-mode"
        error_message = (
            "Failed to download NLP models from Hugging Face. This is often due to network issues. "
            "Please check your internet connection. \n\nFor offline usage, "
            f"refer to Hugging Face's documentation on offline mode: {offline_docs_url}"
        )
        logger.error(f"OSError during NLP model loading for Story Analyzer: {e}. {error_message}", exc_info=True)
        st.error(error_message)
        raise RuntimeError(f"Story Analyzer model download failed (OSError): {e}") from e
    except Exception as e: # Catch any other exception
        logger.error(f"Fatal error loading NLP models for Story Analyzer: {e}", exc_info=True)
        st.error(f"Could not load essential NLP models for Story Analyzer: {e}. The application cannot continue. Please check model names, network connection, or Hugging Face Hub status.")
        raise RuntimeError(f"Story Analyzer NLP model loading failed: {e}") from e

# Load models globally. If this fails, the app will stop due to the error handling in load_nlp_models.
try:
    sentiment_analyzer, embedding_tokenizer, embedding_model = load_nlp_models()
except RuntimeError: # Catch the re-raised RuntimeError
    logger.info("Application halted due to failure in loading NLP models for Story Analyzer.")
    st.stop() # Ensure app stops if models didn't load.

# --- Sample Story Data (Alice's Adventures in Wonderland - Snippets) ---
STORY_DATA = {
    "book_1": {
        "title": "Alice's Adventures in Wonderland",
        "chapters": [
            {
                "chapter_id": "ch1_down_the_rabbit_hole",
                "title": "Chapter 1: Down the Rabbit-Hole",
                "sections": [
                    {"section_id": "s1", "text": "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do. Suddenly a White Rabbit with pink eyes ran close by her. There was nothing so VERY remarkable in that; nor did Alice think it so VERY much out of the way to hear the Rabbit say to itself, 'Oh dear! Oh dear! I shall be late!'"},
                    {"section_id": "s2", "text": "Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it. Burning with curiosity, she ran across the field after it."}
                ]
            },
            {
                "chapter_id": "ch8_croquet_ground",
                "title": "Chapter 8: The Queen's Croquet-Ground",
                "sections": [
                    {"section_id": "s3", "text": "A large rose-tree stood near the entrance of the garden: the roses growing on it were white, but there were three gardeners at it, busily painting them red. Alice thought this a very curious thing. The Queen of Hearts shouted, 'Off with their heads!' Alice felt a little nervous."},
                    {"section_id": "s4", "text": "Alice was soon called upon to play croquet with the Queen. The Queen was in a furious passion, and went stamping about, and shouting 'Off with his head!' or 'Off with her head!' about once in a minute. Alice began to feel very uneasy: 'they're dreadfully fond of beheading people here,' she thought."},
                    {"section_id": "s5", "text": "The Queen left off, quite out of breath, and said to Alice, 'Have you seen the Mock Turtle yet?' 'No,' said Alice. 'I don't even know what a Mock Turtle is.' 'It's the thing Mock Turtle Soup is made from,' said the Queen."}
                ]
            },
            {
                "chapter_id": "ch9_mock_turtles_story",
                "title": "Chapter 9: The Mock Turtle's Story",
                "sections": [
                    {"section_id": "s6", "text": "They had not gone far before they saw the Mock Turtle in the distance, sitting sad and lonely on a little ledge of rock, and, as they came nearer, Alice could hear him sighing as if his heart would break. Alice pitied him deeply. 'What is his sorrow?' she asked the Gryphon."},
                    {"section_id": "s7", "text": "The Mock Turtle looked at Alice with large eyes full of tears, but said nothing. 'Tell her your history, you know,' the Gryphon said to the Mock Turtle in an undertone."}
                ]
            }
        ]
    }
}

CHARACTERS_OF_INTEREST = ["Alice", "Queen", "Rabbit", "Mock Turtle"]

# --- TensorStorage Initialization using the Embedded Class ---
if 'ts' not in st.session_state:
    st.session_state.ts = EmbeddedTensorStorage() # Use the embedded class
ts: EmbeddedTensorStorage = st.session_state.ts

# Dataset names in TensorStorage
SENTENCE_EMBEDDINGS_DS = "sentence_embeddings_store"
CHARACTER_SENTIMENT_DS = "character_sentiment_store"
CHARACTER_INTERACTION_DS = "character_interaction_store"


# --- NLP Processing Functions ---
def get_sentence_embeddings(text: str) -> Optional[torch.Tensor]:
    """
    Generates sentence embeddings for each sentence in the input text.
    Generates sentence embeddings for each sentence in the input text.

    Args:
        text: The input string containing one or more sentences.

    Returns:
        Optional[torch.Tensor]: A 2D torch.Tensor of shape
        (num_sentences, embedding_dim), where each row is the embedding
        for a sentence. Returns `None` if the input text is invalid,
        no sentences are found, or an error occurs during embedding.

    Side Effects:
        Logs warnings or errors via `logger`. Displays errors via `st.error`.
    """
    if not isinstance(text, str) or not text.strip():
        logger.warning("get_sentence_embeddings called with empty or invalid text.")
        # Optionally: st.warning("Input text is empty, cannot generate sentence embeddings.")
        return None

    try:
        sentences = nltk.sent_tokenize(text) # Split text into sentences
        if not sentences:
            logger.warning("No sentences found in text for embedding.")
            return None

        # Tokenize sentences and prepare input for the embedding model.
        inputs = embedding_tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128 # Max length for sentence embeddings
        )
        with torch.no_grad(): # Disable gradient calculations for inference.
            outputs = embedding_model(**inputs)

        # Use mean pooling of the last hidden state to get sentence embeddings.
        sentence_embeds = outputs.last_hidden_state.mean(dim=1)
        return sentence_embeds
    except Exception as e:
        logger.error(f"Error generating sentence embeddings for text snippet '{text[:50]}...': {e}", exc_info=True)
        st.error(f"Could not generate sentence embeddings: {e}")
        return None


def analyze_character_sentiment_and_interaction(
    text: str,
    characters: List[str]
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Analyzes text to determine sentiment associated with characters in each
    sentence and builds a character interaction matrix for the entire text.

    Args:
        text: The input string (e.g., a story section).
        characters: A list of character names to track.

    Returns:
        A tuple containing:
        - char_sentiment_data (Optional[torch.Tensor]): A 3D tensor of shape
          [num_sentences, num_chars, 3 (neg, neut, pos)], representing
          one-hot encoded sentiment of each character in each sentence.
          Sentiment categories are [Negative, Neutral, Positive].
        - interaction_matrix (Optional[torch.Tensor]): A 2D tensor of shape
          [num_chars, num_chars], counting co-occurrences of characters
          within the same sentences.
        Returns `(None, None)` if input is invalid, no sentences are found,
        or a critical error occurs.

    Side Effects:
        Logs warnings or errors. May display errors via `st.error`.
    """
    if not isinstance(text, str) or not text.strip():
        logger.warning("analyze_character_sentiment_and_interaction called with empty or invalid text.")
        return None, None
    if not characters:
        logger.warning("Character list is empty. Cannot analyze character sentiment/interaction.")
        # Depending on desired behavior, could return zero tensors of appropriate shapes.
        return None, None

    try:
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            logger.warning("No sentences found in text for character analysis.")
            return None, None

        num_sentences = len(sentences)
        num_chars = len(characters)

        # Initialize tensors
        # char_sentiment_data: [num_sentences, num_chars, 3 (neg, neut, pos)]
        char_sentiment_data = torch.zeros(num_sentences, num_chars, 3)
        # interaction_matrix: [num_chars, num_chars]
        interaction_matrix = torch.zeros(num_chars, num_chars)

        # Pre-compile regex patterns for efficient character searching (case-insensitive)
        char_patterns = {
            char: re.compile(r'\b' + re.escape(char) + r'\b', re.IGNORECASE)
            for char in characters
        }

        for i, sentence in enumerate(sentences):
            try:
                # Analyze overall sentiment of the current sentence
                # sentiment_analyzer returns a list of dicts; we take the first.
                sent_sentiment_result = sentiment_analyzer(sentence)[0]
                label = sent_sentiment_result.get('label', 'NEUTRAL') # Default to NEUTRAL if label missing

                # Map sentiment label to an index: 0 for Negative, 1 for Neutral, 2 for Positive
                if label == 'POSITIVE':
                    sent_sentiment_category_idx = 2
                elif label == 'NEGATIVE':
                    sent_sentiment_category_idx = 0
                else: # Includes NEUTRAL or any other label
                    sent_sentiment_category_idx = 1

                # Identify which characters are present in the current sentence
                present_chars_indices_in_sentence = []
                for char_idx, char_name in enumerate(characters):
                    if char_patterns[char_name].search(sentence):
                        present_chars_indices_in_sentence.append(char_idx)
                        # Attribute the sentence's overall sentiment to each character present.
                        # This is a simplification; more advanced methods could find character-specific sentiment.
                        char_sentiment_data[i, char_idx, sent_sentiment_category_idx] = 1

                # Update interaction matrix for characters co-occurring in this sentence
                # If characters A, B, C are in a sentence, this increments counts for (A,B), (A,C), (B,C) and self-interactions (A,A).
                for list_idx1 in range(len(present_chars_indices_in_sentence)):
                    for list_idx2 in range(list_idx1, len(present_chars_indices_in_sentence)):
                        actual_char_idx1 = present_chars_indices_in_sentence[list_idx1]
                        actual_char_idx2 = present_chars_indices_in_sentence[list_idx2]

                        interaction_matrix[actual_char_idx1, actual_char_idx2] += 1
                        # If it's not a self-interaction (diagonal), increment the symmetric position too.
                        if actual_char_idx1 != actual_char_idx2:
                            interaction_matrix[actual_char_idx2, actual_char_idx1] += 1
            except Exception as e_sentence: # Error processing a single sentence
                logger.error(f"Error processing sentence '{sentence[:30]}...' for character analysis: {e_sentence}", exc_info=True)
                # Continue to the next sentence, but log the error.
                # char_sentiment_data and interaction_matrix for this sentence might be incomplete.
                continue

        return char_sentiment_data, interaction_matrix
    except Exception as e_main: # Catch-all for errors in the main function logic
        logger.error(f"Fatal error in analyze_character_sentiment_and_interaction for text '{text[:50]}...': {e_main}", exc_info=True)
        st.error(f"Could not analyze character sentiment and interactions: {e_main}")
        return None, None

# --- Data Ingestion Logic ---
def ingest_story_data(story_data: Dict, characters: List[str]):
    """
    Processes raw story data, generates various tensor representations for each
    section, and stores them in the (simulated) TensorStorage (`ts`).

    Args:
        story_data (Dict): A dictionary containing the story data, structured by
                           books, chapters, and sections. Expected format is like
                           `STORY_DATA` global variable.
        characters (List[str]): A list of character names to focus on during
                                analysis (e.g., for sentiment and interaction).

    Side Effects:
        - Modifies the global `ts` (TensorStorage) instance by creating datasets
          (if they don't exist) and inserting tensor records.
        - Updates Streamlit UI elements:
            - `st.sidebar.info` for dataset status.
            - `st.sidebar.progress` to show ingestion progress.
            - `st.sidebar.success` on completion.
            - `st.error` if critical errors occur (e.g., malformed input data).
        - Logs information and errors using `logger`.
        - Sets `st.session_state.data_ingested = True` upon successful completion.
    """
    try:
        # Ensure datasets are created in TensorStorage before ingestion.
        # This is idempotent; create_dataset in EmbeddedTensorStorage handles existing datasets.
        for ds_name in [SENTENCE_EMBEDDINGS_DS, CHARACTER_SENTIMENT_DS, CHARACTER_INTERACTION_DS]:
            try:
                ts.create_dataset(ds_name)
                logger.info(f"Dataset '{ds_name}' ensured in simulated TensorStorage.")
            except ValueError: # Should be handled by EmbeddedTensorStorage, but good for defense
                logger.warning(f"Dataset '{ds_name}' might already exist or failed creation (should be handled by storage class).")
                # Not showing st.sidebar.info here as create_dataset in storage class already logs.
            except Exception as e_ds: # Catch other unexpected errors from create_dataset
                logger.error(f"Unexpected error creating dataset {ds_name}: {e_ds}", exc_info=True)
                st.error(f"Critical error setting up dataset {ds_name}: {e_ds}. Ingestion halted.")
                return # Halt ingestion if a dataset cannot be ensured.

        # Calculate total sections for progress bar
        total_sections = 0
        if not isinstance(story_data, dict): # Validate story_data type
            st.error("Invalid story_data format: Expected a dictionary. Cannot start ingestion.")
            logger.error("ingest_story_data: story_data is not a dictionary.")
            return

        for book_id, book_data in story_data.items():
            if not isinstance(book_data, dict) or "chapters" not in book_data or not isinstance(book_data.get("chapters"), list):
                logger.warning(f"Malformed book data for book_id '{book_id}'. Skipping this book for section count.")
                st.warning(f"Skipping potentially malformed book data during count: {book_id}")
                continue
            for chapter_data in book_data["chapters"]:
                if not isinstance(chapter_data, dict) or "sections" not in chapter_data or not isinstance(chapter_data.get("sections"), list):
                    logger.warning(f"Malformed chapter data in book '{book_id}'. Skipping this chapter for section count.")
                    st.warning(f"Skipping potentially malformed chapter during count in book: {book_id}")
                    continue
                total_sections += len(chapter_data["sections"])

        if total_sections == 0:
            st.info("No sections found in the story data to ingest.")
            logger.info("No sections to ingest from story_data because total_sections is 0.")
            st.session_state.data_ingested = False # No data was actually ingested
            return

        ingestion_progress = st.sidebar.progress(0.0, "Ingesting story sections...")
        processed_sections = 0
        # Use a more descriptive variable name for full_section_id in case of early skip
        current_full_section_id_for_progress = "Starting..."

        # Iterate through the structured story data
        for book_id, book_data in story_data.items():
            # Safe access to book_data and its 'chapters' list
            if not isinstance(book_data, dict): continue # Already warned by section counter

            for chapter_data in book_data.get("chapters", []): # Use .get for safety
                if not isinstance(chapter_data, dict): continue # Already warned

                for section_data in chapter_data.get("sections", []): # Use .get for safety
                    current_full_section_id_for_progress = "Processing section..." # Default for progress
                    try:
                        # Validate section_data structure using .get for safer access
                        section_id_short = section_data.get("section_id")
                        text_content = section_data.get("text")
                        chapter_id_val = chapter_data.get("chapter_id")
                        chapter_title_val = chapter_data.get("title")

                        if not all([section_id_short, text_content, chapter_id_val, chapter_title_val]):
                            logger.warning(f"Skipping section in {book_id} - {chapter_id_val or 'UnknownChapter'} due to missing critical data (id, text, chapter_id, title). Section data: {section_data}")
                            st.warning(f"Skipping section with missing data in {chapter_title_val or 'Unknown Chapter'}.")
                            # full_section_id might not be available here if chapter_id_val or section_id_short is None
                            current_full_section_id_for_progress = f"{book_id}_{chapter_id_val or '?'}_{section_id_short or '?'}_skipped"
                            # Still increment processed_sections for progress bar to complete
                            processed_sections += 1
                            if total_sections > 0:
                                ingestion_progress.progress(processed_sections / total_sections, f"Processed: {current_full_section_id_for_progress}")
                            continue

                        full_section_id = f"{book_id}_{chapter_id_val}_{section_id_short}"
                        current_full_section_id_for_progress = full_section_id


                        base_metadata = {
                            "book_id": book_id, "chapter_id": chapter_id_val,
                            "chapter_title": chapter_title_val, "section_id": section_id_short,
                            "full_id": full_section_id,
                            "_text_snippet_for_demo": text_content[:100]
                        }

                        # --- Tensor 1: Sentence Embeddings ---
                        sentence_embeds_tensor = get_sentence_embeddings(text_content)
                        if sentence_embeds_tensor is not None:
                            ts.insert(SENTENCE_EMBEDDINGS_DS, sentence_embeds_tensor, {**base_metadata, "tensor_type": "sentence_embeddings"})
                        else:
                            logger.warning(f"Sentence embeddings not generated for section {full_section_id}. Skipping insertion.")
                            st.warning(f"Could not generate sentence embeddings for section: {full_section_id}")

                        # --- Tensors 2 & 3: Character Sentiment and Interaction ---
                        char_sentiment_tensor, interaction_tensor = analyze_character_sentiment_and_interaction(text_content, characters)

                        if char_sentiment_tensor is not None:
                            ts.insert(CHARACTER_SENTIMENT_DS, char_sentiment_tensor, {**base_metadata, "tensor_type": "character_sentiment_flow", "characters_definition": characters})
                        else: # analyze_character_sentiment_and_interaction already logs/warns if text was valid
                            if text_content and characters: # Only log if inputs were valid but still no tensor
                                logger.info(f"Character sentiment tensor not generated for section {full_section_id} (e.g. no sentences or no characters found).")

                        if interaction_tensor is not None:
                            ts.insert(CHARACTER_INTERACTION_DS, interaction_tensor, {**base_metadata, "tensor_type": "character_interaction_matrix", "characters_definition": characters})
                        else:
                            if text_content and characters:
                                logger.info(f"Interaction tensor not generated for section {full_section_id} (e.g. no sentences or no characters found).")

                    except KeyError as e_key:
                        logger.error(f"Missing key {e_key} in section data for book '{book_id}', chapter '{chapter_data.get('title', 'UnknownChap')}'. Section: {section_data.get('section_id', 'UnknownSec')}", exc_info=True)
                        st.error(f"Data error in section of '{chapter_data.get('title', 'UnknownChap')}': Missing {e_key}.")
                    except Exception as e_section:
                        logger.error(f"Error processing section '{section_data.get('section_id', 'UnknownSec')}' in chapter '{chapter_data.get('title', 'UnknownChap')}': {e_section}", exc_info=True)
                        st.error(f"Failed to process section {section_data.get('section_id', 'UnknownSec')} in {chapter_data.get('title', 'UnknownChap')}: {e_section}")
                    finally:
                        processed_sections += 1
                        if total_sections > 0:
                            ingestion_progress.progress(processed_sections / total_sections, f"Ingested: {current_full_section_id_for_progress}")

        ingestion_progress.empty()
        st.sidebar.success(f"Story data ingestion attempt complete ({processed_sections} sections processed out of {total_sections}).")
        st.session_state.data_ingested = True # Mark as ingested even if some sections failed, as some data might be usable.

    except Exception as e_ingest_main:
        st.error(f"A critical error occurred during the main data ingestion routine: {e_ingest_main}")
        logger.critical(f"Main data ingestion process failed critically: {e_ingest_main}", exc_info=True)
        if 'ingestion_progress' in locals() and ingestion_progress is not None : ingestion_progress.empty()
        st.session_state.data_ingested = False


# --- "Story Analyst" Agent Logic (Conceptual) ---
def analyze_character_evolution(
    target_char: str,
    related_char: Optional[str], # Can be None if analyzing single character sentiment
    book_id_filter: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyzes the evolution of interaction strength (if `related_char` is provided)
    and sentiment for a `target_char` throughout a specified book.

    Retrieves data from the (simulated) TensorStorage.

    Args:
        target_char: The primary character for analysis.
        related_char: The secondary character for interaction analysis.
                      If None, only sentiment evolution for target_char is analyzed.
        book_id_filter: The ID of the book to filter data from.

    Returns:
        A tuple of two pandas DataFrames:
        - df_interactions_plot: DataFrame for plotting interaction strength.
          Columns: ['display_order', 'chapter', 'interaction_strength'].
          Empty if `related_char` is None or no interaction data.
        - df_sentiments_plot: DataFrame for plotting sentiment evolution of `target_char`.
          Columns: ['display_order', 'chapter', f'{target_char}_sentiment_score'].
          Empty if no sentiment data.
    """
    # Validate character inputs
    if not target_char or target_char not in CHARACTERS_OF_INTEREST:
        st.error(f"Target character '{target_char}' is invalid or not in the predefined list: {CHARACTERS_OF_INTEREST}.")
        logger.warning(f"analyze_character_evolution: Invalid target_char '{target_char}'.")
        return pd.DataFrame(), pd.DataFrame() # Return empty DataFrames for consistency
    if related_char and related_char not in CHARACTERS_OF_INTEREST:
        st.error(f"Related character '{related_char}' is invalid or not in the predefined list: {CHARACTERS_OF_INTEREST}.")
        logger.warning(f"analyze_character_evolution: Invalid related_char '{related_char}'.")
        return pd.DataFrame(), pd.DataFrame()
    if target_char == related_char and related_char is not None:
        st.warning("Target and related characters should be different for interaction analysis. Analyzing sentiment for target character only.")
        # Proceed with sentiment analysis for target_char, but interaction will be empty.
        related_char = None # Disable interaction part

    all_interaction_metadata: List[Dict] = []
    all_sentiment_metadata: List[Dict] = []
    try:
        # Retrieve all metadata first, then filter.
        if related_char and CHARACTER_INTERACTION_DS in ts.datasets:
            all_interaction_metadata = ts.get_all_metadata_for_query(CHARACTER_INTERACTION_DS)

        if CHARACTER_SENTIMENT_DS in ts.datasets:
            all_sentiment_metadata = ts.get_all_metadata_for_query(CHARACTER_SENTIMENT_DS)

        # Check if essential data is missing
        if not all_sentiment_metadata and (not related_char or not all_interaction_metadata):
             st.info(f"No relevant story data (sentiment or interaction for '{target_char}') found in storage for book '{book_id_filter}'. Was data fully ingested?")
             logger.info(f"No sentiment or interaction datasets found for analysis of '{target_char}' in book '{book_id_filter}'.")
             return pd.DataFrame(), pd.DataFrame()

    except ValueError as e_val:
        st.error(f"Dataset not found during analysis: {e_val}. Was data ingested correctly and are dataset names correct?")
        logger.error(f"analyze_character_evolution: Dataset not found: {e_val}", exc_info=True)
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e_retrieve:
        st.error(f"An error occurred while retrieving metadata for analysis: {e_retrieve}")
        logger.error(f"analyze_character_evolution: Error retrieving metadata: {e_retrieve}", exc_info=True)
        return pd.DataFrame(), pd.DataFrame()

    # Filter metadata for the selected book
    interaction_metadata_book = [m for m in all_interaction_metadata if m.get("book_id") == book_id_filter and m.get("full_id")]
    sentiment_metadata_book = [m for m in all_sentiment_metadata if m.get("book_id") == book_id_filter and m.get("full_id")]

    if not sentiment_metadata_book and (not related_char or not interaction_metadata_book):
        st.info(f"No data found for book '{book_id_filter}' in the relevant datasets after filtering.")
        logger.info(f"No data for book '{book_id_filter}' after filtering metadata.")
        return pd.DataFrame(), pd.DataFrame()

    try:
        char_map = {name: i for i, name in enumerate(CHARACTERS_OF_INTEREST)}
        target_char_idx = char_map[target_char]
        # Handle related_char being None or not in map (already checked, but defensive)
        related_char_idx = char_map.get(related_char) if related_char else -1
    except KeyError as e_map: # Should not happen if initial validation is correct
        st.error(f"Internal character mapping error: {e_map}. This should not happen.")
        logger.critical(f"analyze_character_evolution: Character mapping failed unexpectedly: {e_map}", exc_info=True)
        return pd.DataFrame(), pd.DataFrame()

    evolution_data_points = []
    # Use set of full_ids from both lists to ensure each section is processed once
    section_ids_interaction = {m['full_id'] for m in interaction_metadata_book}
    section_ids_sentiment = {m['full_id'] for m in sentiment_metadata_book}
    all_section_full_ids_in_book = sorted(list(section_ids_interaction.union(section_ids_sentiment)))

    if not all_section_full_ids_in_book:
        st.info(f"No sections with relevant data identified for book '{book_id_filter}' after processing metadata.")
        return pd.DataFrame(), pd.DataFrame()

    for full_id in all_section_full_ids_in_book:
        interaction_strength = 0.0
        avg_sentiment_score = 0.0 # Default to neutral

        # Get metadata for the current section, preferring sentiment if available, else interaction
        current_section_meta = next((m for m in sentiment_metadata_book if m.get("full_id") == full_id), None)
        if not current_section_meta and related_char: # Fallback if only interaction data exists for this section
            current_section_meta = next((m for m in interaction_metadata_book if m.get("full_id") == full_id), None)

        if not current_section_meta: # Should ideally not happen if all_section_full_ids_in_book is built correctly
            logger.warning(f"Metadata for section_id '{full_id}' unexpectedly not found. Skipping.")
            continue

        # --- Calculate Interaction Strength ---
        if related_char and related_char_idx != -1: # Ensure related_char is valid
            interaction_meta_entry = next((m for m in interaction_metadata_book if m.get("full_id") == full_id), None)
            if interaction_meta_entry and interaction_meta_entry.get('record_id'):
                interaction_tensor = ts.get_tensor_by_record_id_from_list(
                    CHARACTER_INTERACTION_DS, interaction_meta_entry['record_id']
                )
                if interaction_tensor is not None and interaction_tensor.ndim == 2 and \
                   target_char_idx < interaction_tensor.shape[0] and related_char_idx < interaction_tensor.shape[1]:
                    interaction_strength = interaction_tensor[target_char_idx, related_char_idx].item()
                elif interaction_tensor is not None:
                    logger.warning(f"Interaction tensor for section {full_id} (record: {interaction_meta_entry['record_id']}) has unexpected shape {interaction_tensor.shape} or char indices out of bounds ({target_char_idx}, {related_char_idx}).")
            # else: No interaction record or tensor for this section. interaction_strength remains 0.0.

        # --- Calculate Average Sentiment Score for target_char ---
        sentiment_meta_entry = next((m for m in sentiment_metadata_book if m.get("full_id") == full_id), None)
        if sentiment_meta_entry and sentiment_meta_entry.get('record_id'):
            sentiment_tensor = ts.get_tensor_by_record_id_from_list(
                CHARACTER_SENTIMENT_DS, sentiment_meta_entry['record_id']
            )
            if sentiment_tensor is not None and sentiment_tensor.ndim == 3 and \
               target_char_idx < sentiment_tensor.shape[1] and sentiment_tensor.shape[2] == 3: # [sentences, chars, 3_scores]

                sentiments_for_target = sentiment_tensor[:, target_char_idx, :]
                valid_sentences_mask = torch.sum(sentiments_for_target, dim=1) > 0 # Check if any sentiment is non-zero
                if torch.any(valid_sentences_mask):
                    relevant_sentiments_one_hot = sentiments_for_target[valid_sentences_mask]
                    scores = relevant_sentiments_one_hot[:, 2] * 1.0 + relevant_sentiments_one_hot[:, 0] * (-1.0) # Pos=1, Neg=-1
                    avg_sentiment_score = scores.float().mean().item()
            elif sentiment_tensor is not None:
                logger.warning(f"Sentiment tensor for section {full_id} (record: {sentiment_meta_entry['record_id']}) has unexpected shape {sentiment_tensor.shape} or char index out of bounds ({target_char_idx}).")

        evolution_data_points.append({
            "full_id": full_id,
            "chapter": current_section_meta.get("chapter_title", "N/A"), # Safely get chapter title
            "section_id_val": current_section_meta.get("section_id", "N/A"),
            "interaction_strength": interaction_strength,
            f"{target_char}_sentiment_score": avg_sentiment_score
        })

    if not evolution_data_points:
        st.info(f"No data points generated for analysis of '{target_char}' (and '{related_char or 'N/A'}') in book '{book_id_filter}'. This might be due to no interactions or no sentiment data for the character in this book.")
        return pd.DataFrame(), pd.DataFrame()

    try:
        df_evolution = pd.DataFrame(evolution_data_points)
        # Sort by 'full_id' to ensure chronological (narrative) order.
        df_evolution = df_evolution.sort_values(by='full_id').reset_index(drop=True)
        # Use 'full_id' or a processed version of it for display on x-axis if needed.
        # For simplicity in plotting, we might just use the sorted order.
        df_evolution['display_order'] = df_evolution['full_id']
    except Exception as e_df:
        st.error(f"Failed to create DataFrame for analysis results: {e_df}")
        logger.error(f"DataFrame creation/sorting failed in analyze_character_evolution: {e_df}", exc_info=True)
        return pd.DataFrame(), pd.DataFrame() # Return empty DFs

    # Prepare DataFrames for plotting
    # Interaction DataFrame is only relevant if related_char was analyzed.
    df_interactions_plot = df_evolution[['display_order', 'chapter', 'interaction_strength']] if related_char else pd.DataFrame()
    df_sentiments_plot = df_evolution[['display_order', 'chapter', f"{target_char}_sentiment_score"]]
    
    return df_interactions_plot, df_sentiments_plot


# --- Streamlit UI ---
st.title("📚 Smart Story Analyzer Demo (Tensorus Concept)")
st.write(
    "Analyzes character relationships and sentiment evolution "
    "using a simulated, embedded TensorStorage."
)

# Initialize session state for data ingestion tracking if not already present.
if 'data_ingested' not in st.session_state:
    st.session_state.data_ingested = False

# More robust check for data ingestion status by inspecting TensorStorage.
# This helps if the script reran and `st.session_state.data_ingested` is True
# but `ts` (the TensorStorage instance) is empty.
try:
    # Ensure ts is available (it's initialized globally but might fail if EmbeddedTensorStorage init has issues)
    if 'ts' in st.session_state and hasattr(st.session_state.ts, 'list_datasets'):
        if not st.session_state.ts.list_datasets(): # Check if any datasets exist
            st.session_state.data_ingested = False
            logger.info("TensorStorage has no datasets; resetting data_ingested state to False.")
    elif 'ts' not in st.session_state: # If ts was never even assigned to session_state
        st.session_state.data_ingested = False
        logger.error("TensorStorage 'ts' not found in session_state. Critical initialization error likely.")
        st.error("TensorStorage 'ts' instance is missing. Please restart the application.")
        st.stop()

except AttributeError: # ts might not be initialized if EmbeddedTensorStorage() failed or ts object is malformed
    st.session_state.data_ingested = False
    logger.error("TensorStorage 'ts' object is not correctly initialized or missing 'list_datasets' method.", exc_info=True)
    st.error("TensorStorage initialization error. Please refresh the page or check logs.")
    st.stop() # Stop if TensorStorage is not functional


st.sidebar.header("Demo Setup & Control")

# UI for data ingestion
if not st.session_state.get('data_ingested', False): # Use .get for safer access
    st.sidebar.markdown(
        "Load the sample story data to enable analysis. This will populate "
        "the **simulated TensorStorage**."
    )
    if st.sidebar.button("Load and Ingest Sample Story Data", key="load_ingest_button"):
        with st.spinner("Processing text, generating tensors, and ingesting... This may take a moment."):
            ingest_story_data(STORY_DATA, CHARACTERS_OF_INTEREST)
        # Rerun to reflect the new state (e.g., hide button, show success message).
        st.rerun()
else:
    st.sidebar.success("Sample story data loaded into the simulated TensorStorage.")
    st.sidebar.markdown("Re-ingest if you want to clear and reload all demo data.")
    if st.sidebar.button("Re-Ingest Data (Clears Simulated Data)", key="reingest_button"):
        try:
            # Re-initialize the TensorStorage to clear previous data.
            st.session_state.ts = EmbeddedTensorStorage()
            ts = st.session_state.ts # Update global reference.
            st.session_state.data_ingested = False # Mark as not ingested before starting.
            logger.info("TensorStorage cleared for data re-ingestion.")
            with st.spinner("Re-processing and re-ingesting story data..."):
                ingest_story_data(STORY_DATA, CHARACTERS_OF_INTEREST)
            st.rerun() # Refresh UI.
        except Exception as e_reingest:
            st.error(f"An error occurred during data re-ingestion: {e_reingest}")
            logger.error(f"Data re-ingestion failed: {e_reingest}", exc_info=True)

# Main application area: only proceed if data has been ingested.
if st.session_state.get('data_ingested', False):
    st.header("Character Evolution Analysis (Conceptual)")
    st.markdown(
        "This section demonstrates how an AI agent might query the "
        "**simulated TensorStorage** to analyze character dynamics over the narrative."
    )
    
    # Assuming only one book for this demo's STORY_DATA for simplicity.
    # Robustly get the first book_id and its title.
    try:
        # Check if STORY_DATA is not empty and is a dictionary
        if not STORY_DATA or not isinstance(STORY_DATA, dict):
            raise ValueError("STORY_DATA is empty or not a dictionary.")
        first_book_id = list(STORY_DATA.keys())[0]

        # Check if the first book_id maps to a dictionary with a "title"
        if not isinstance(STORY_DATA.get(first_book_id), dict) or "title" not in STORY_DATA.get(first_book_id, {}):
             raise ValueError(f"Data for book '{first_book_id}' is malformed or missing a title.")
        story_title = STORY_DATA[first_book_id].get("title", "Unknown Title")
        st.subheader(f"Story: *{story_title}*")

    except (IndexError, AttributeError, KeyError, ValueError) as e_story_title:
        st.error(f"Could not display story title due to data structure error: {e_story_title}")
        logger.error(f"Error accessing story title from STORY_DATA: {e_story_title}", exc_info=True)
        first_book_id = None # Ensure it's None if not found, preventing further errors
        st.stop() # Stop if basic story data cannot be loaded/parsed

    if first_book_id: # Proceed only if a book_id was successfully obtained
        col1, col2 = st.columns(2)
        with col1:
            main_char_options = CHARACTERS_OF_INTEREST
            main_character = st.selectbox(
                "Select Main Character:",
                options=main_char_options,
                index=0 if main_char_options else -1, # Handle empty options
                key="main_char_select"
            )
        with col2:
            related_character_options = [c for c in CHARACTERS_OF_INTEREST if c != main_character]
            # Default to "None" option selected if no other options, or if it's the desired default
            default_related_index = 0

            related_character = st.selectbox(
                "Select Related Character (for interaction analysis, optional):",
                options=[None] + related_character_options,
                index=default_related_index,
                format_func=lambda x: x if x is not None else "None (Sentiment for Main Character Only)",
                key="related_char_select"
            )

        # Construct button label dynamically and provide user guidance
        analysis_button_label = f"Analyze {main_character}"
        actual_related_char_for_analysis = None # Initialize
        if related_character and main_character != related_character:
            analysis_button_label += f"'s Attitude & Interaction with {related_character}"
            actual_related_char_for_analysis = related_character
        elif related_character and main_character == related_character:
            st.warning("Main and Related characters are the same. Analyzing sentiment for the main character only.")
            analysis_button_label += "'s Sentiment Evolution"
            # actual_related_char_for_analysis remains None
        else: # No related character selected
            analysis_button_label += "'s Sentiment Evolution"
            # actual_related_char_for_analysis remains None

        # Ensure there's a main character to analyze
        if main_character and st.button(analysis_button_label, key="analyze_story_button"):
            spinner_message = f"Analyzing data for {main_character}"
            if actual_related_char_for_analysis:
                spinner_message += f" and {actual_related_char_for_analysis}"
            spinner_message += "..."

            with st.spinner(spinner_message):
                df_interactions_plot, df_sentiments_plot = analyze_character_evolution(
                    main_character, actual_related_char_for_analysis, first_book_id
                )

            # Plotting interaction strength
            if actual_related_char_for_analysis: # Only show if interaction was analyzed
                if df_interactions_plot is not None and not df_interactions_plot.empty:
                    st.subheader(f"Interaction Strength: {main_character} & {actual_related_char_for_analysis} Over Narrative")
                    st.caption("Interaction strength is based on co-occurrence count in sentences. Higher values mean more frequent co-mentions.")
                    df_interactions_plot_renamed = df_interactions_plot.rename(
                        columns={'display_order': 'Narrative Section (Chronological)'}
                    )
                    st.line_chart(df_interactions_plot_renamed.set_index('Narrative Section (Chronological)')['interaction_strength'])
                    with st.expander("View Interaction Data Table"):
                        st.dataframe(df_interactions_plot_renamed)
                else:
                    st.info(f"No significant interaction data found between {main_character} and {actual_related_char_for_analysis} to plot.")

            # Plotting sentiment evolution for the main character
            if df_sentiments_plot is not None and not df_sentiments_plot.empty:
                sentiment_subheader = f"Sentiment Score of {main_character}"
                if actual_related_char_for_analysis:
                    sentiment_subheader += f" (when interacting with {actual_related_char_for_analysis})"
                st.subheader(sentiment_subheader)
                st.caption(
                    f"Sentiment score for {main_character} (from -1 Negative to +1 Positive). "
                    f"Based on average sentiment of sentences where {main_character} appears "
                    f"{f'and interacts with {actual_related_char_for_analysis}' if actual_related_char_for_analysis else ''}."
                )

                sentiment_col_name = f"{main_character}_sentiment_score"
                df_sentiments_plot_renamed = df_sentiments_plot.rename(
                    columns={'display_order': 'Narrative Section (Chronological)', sentiment_col_name: 'Sentiment Score'}
                )
                
                if 'Sentiment Score' in df_sentiments_plot_renamed.columns:
                    st.line_chart(df_sentiments_plot_renamed.set_index('Narrative Section (Chronological)')['Sentiment Score'])
                    with st.expander("View Sentiment Data Table"):
                        st.dataframe(df_sentiments_plot_renamed)
                else:
                     st.warning(f"Sentiment data column '{sentiment_col_name}' not found. This might indicate an issue with the analysis or data.")
            else:
                sentiment_context = f" when interacting with {actual_related_char_for_analysis}" if actual_related_char_for_analysis else ""
                st.info(f"No significant sentiment data found for {main_character}{sentiment_context} to plot.")
        elif not main_character: # If no main character is available/selected (e.g. CHARACTERS_OF_INTEREST is empty)
             st.error("No characters available for analysis. Please check character definitions.")

    st.divider()
    st.header("Explore Stored Tensors (Simulated TensorStorage Peek)")
    st.markdown(
        "This section allows you to peek into the raw tensor data stored in the "
        "**simulated TensorStorage** for the ingested story sections."
    )

    try:
        available_datasets = ts.list_datasets() if 'ts' in st.session_state and hasattr(st.session_state.ts, 'list_datasets') else []
        if available_datasets:
            dataset_to_explore = st.selectbox(
                "Select Dataset to Peek Into:",
                options=available_datasets,
                key="explore_ds_select" # Unique key
            )
            if st.button("Show First 3 Records from Selected Dataset", key="show_records_button"): # Unique key
                try:
                    records = ts.get_dataset_with_metadata(dataset_to_explore)
                    if records:
                        st.write(f"Showing first {min(3, len(records))} of {len(records)} records from **'{dataset_to_explore}'**:")
                        for r_idx, record_data in enumerate(records[:3]):
                            st.markdown(f"--- \n **Record {r_idx+1}**")
                            st.json({"metadata": record_data.get('metadata', {})})
                            tensor_data = record_data.get('tensor')
                            if tensor_data is not None:
                                st.markdown(f"*Tensor Shape:* `{list(tensor_data.shape)}`")
                                st.markdown(f"*Tensor Dtype:* `{str(tensor_data.dtype)}`")
                                preview_elements = tensor_data.flatten()[:5].tolist()
                                st.markdown(f"*Tensor Preview (first 5 elements):* `{preview_elements}`")
                            else:
                                st.markdown("Tensor data not available for this record.")
                    else:
                        st.info(f"No records found in dataset '{dataset_to_explore}'.")
                except ValueError as ve:
                    st.error(f"Error exploring dataset '{dataset_to_explore}': {ve}")
                    logger.warning(f"ValueError while exploring dataset {dataset_to_explore}: {ve}", exc_info=True)
                except Exception as e_gen:
                    st.error(f"An unexpected error occurred while exploring dataset '{dataset_to_explore}': {e_gen}")
                    logger.error(f"Error exploring dataset {dataset_to_explore}: {e_gen}", exc_info=True)
        else:
            st.info("No datasets have been created or loaded in the (simulated) TensorStorage yet. Please ingest data first.")
    except Exception as e_list_ds:
        st.error(f"Could not list datasets from TensorStorage: {e_list_ds}")
        logger.error(f"Failed to list datasets from TensorStorage: {e_list_ds}", exc_info=True)

else: # If data_ingested is False
    st.info(
        "👈 Please use the sidebar to load and ingest the sample story data. "
        "This will enable the analysis features by populating the simulated TensorStorage."
    )