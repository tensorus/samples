import streamlit as st
import torch
import pandas as pd
import numpy as np
import re
import json # Added for metadata parsing in UI
import uuid # For TensorStorage record IDs
import time # For TensorStorage timestamps
import logging # For TensorStorage logging
from typing import List, Dict, Tuple, Optional, Callable, Any # Expanded for TensorStorage

# --- Streamlit page config must be set first ---
st.set_page_config(page_title="Story Analyzer (Tensorus Concept)", layout="wide") # Adjusted title for clarity

# --- Configure basic logging (optional, but good for the embedded TensorStorage) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Embedded TensorStorage Class (Simulating Tensorus Capabilities) ---
# This class provides a simplified in-memory simulation of a Tensor Database like Tensorus.
# It's designed for this demo to show how structured tensors can be stored and retrieved.
class EmbeddedTensorStorage:
    """
    Manages datasets stored as collections of tensors in memory.
    This is a simplified, self-contained simulation for the demo,
    mimicking some core functionalities of a Tensor Database.
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
            logger.warning(f"Attempted to create dataset '{name}' which already exists.")
            # For this demo, we might allow re-creation silently or just log,
            # but raising an error is stricter and good for showing intent.
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
                      A unique 'record_id' will be generated.
        Returns:
            str: The unique record ID generated for the inserted tensor.
        Raises:
            ValueError: If the dataset 'name' does not exist.
            TypeError: If the 'tensor' argument is not a torch.Tensor.
        """
        if name not in self.datasets:
            # In a real system, one might auto-create or require explicit creation.
            # For this demo, explicit creation via create_dataset is preferred.
            raise ValueError(f"Dataset '{name}' does not exist. Create it first.")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Data to be inserted must be a torch.Tensor.")

        metadata = metadata if metadata is not None else {}

        record_id = str(uuid.uuid4()) # Generate a unique ID for this tensor record.
        # Standardized metadata fields that the storage itself manages.
        default_metadata = {
            "record_id": record_id,
            "timestamp_utc": time.time(), # Timestamp of insertion.
            "shape": list(tensor.shape),  # Store tensor shape for quick reference.
            "dtype": str(tensor.dtype).replace('torch.', ''), # Store tensor data type.
            "version": len(self.datasets[name]["tensors"]) + 1, # Simple versioning.
        }
        # User-provided metadata can override some defaults if necessary, or add new fields.
        final_metadata = {**default_metadata, **metadata} 

        self.datasets[name]["tensors"].append(tensor.clone()) # Store a clone to prevent external modifications.
        self.datasets[name]["metadata"].append(final_metadata)
        logger.debug(f"Tensor with ID {record_id} inserted into dataset '{name}'.")
        return record_id

    def get_dataset_with_metadata(self, name: str) -> List[Dict[str, Any]]:
        """
        Retrieves all tensors and their corresponding metadata from a specified dataset.
        Args:
            name: The name of the dataset.
        Returns:
            A list of dictionaries, where each dictionary contains a 'tensor' and its 'metadata'.
        Raises:
            ValueError: If the dataset 'name' does not exist.
        """
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' does not exist.")
        results = []
        for i in range(len(self.datasets[name]["tensors"])):
            results.append({
                "tensor": self.datasets[name]["tensors"][i], 
                "metadata": self.datasets[name]["metadata"][i]
            })
        return results

    def query(self, name: str, query_fn: Callable[[torch.Tensor, Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
        """
        Queries a dataset using a custom function that operates on each tensor and its metadata.
        Args:
            name: The name of the dataset to query.
            query_fn: A callable that takes a tensor and its metadata dict, returns True if the record matches.
        Returns:
            A list of matching records (dictionaries with 'tensor' and 'metadata').
        Raises:
            ValueError: If the dataset 'name' does not exist.
            TypeError: If query_fn is not callable.
        """
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' does not exist.")
        if not callable(query_fn):
             raise TypeError("query_fn must be a callable function.")
        
        results = []
        for i in range(len(self.datasets[name]["tensors"])):
            tensor = self.datasets[name]["tensors"][i]
            meta = self.datasets[name]["metadata"][i]
            try:
                if query_fn(tensor, meta): # query_fn expects tensor as first arg, metadata as second
                    results.append({"tensor": tensor, "metadata": meta})
            except Exception as e:
                # Log error during query execution but continue with other records.
                logger.warning(f"Error executing query_fn on record {meta.get('record_id', 'N/A')}: {e}")
                continue
        return results
    
    def get_all_metadata_for_query(self, name: str) -> List[Dict[str, Any]]:
        """
        Helper function to retrieve all metadata records from a dataset.
        Useful for performing initial filtering based on metadata before loading potentially large tensors.
        Args:
            name: The name of the dataset.
        Returns:
            A list of all metadata dictionaries in the dataset.
        Raises:
            ValueError: If the dataset 'name' does not exist.
        """
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' does not exist.")
        return list(self.datasets[name]["metadata"]) # Return copies to prevent external modification.

    def get_tensor_by_record_id_from_list(self, name: str, record_id: str) -> Optional[torch.Tensor]:
        """
        Efficiently retrieves a specific tensor by its record_id after its metadata has been identified.
        Args:
            name: The name of the dataset.
            record_id: The unique ID of the record whose tensor is to be retrieved.
        Returns:
            The torch.Tensor if found, otherwise None.
        """
        if name not in self.datasets:
            logger.warning(f"Dataset '{name}' not found when trying to get tensor by record_id '{record_id}'.")
            return None
        for i, meta in enumerate(self.datasets[name]["metadata"]):
            if meta.get("record_id") == record_id:
                return self.datasets[name]["tensors"][i]
        logger.debug(f"Tensor with record_id '{record_id}' not found in dataset '{name}'.")
        return None

# --- End of EmbeddedTensorStorage ---


# For NLP - ensure you have these installed: pip install transformers nltk
try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    import nltk
    
    # Download all required NLTK data
    required_nltk_data = ['punkt', 'punkt_tab', 'stopwords']
    
    for data in required_nltk_data:
        try:
            nltk.data.find(f'tokenizers/{data}')
        except LookupError:
            st.info(f"Downloading NLTK '{data}' data (this will only happen once)...")
            nltk.download(data, quiet=True)
            
    # Also download the english.pickle for sentence tokenization
    try:
        nltk.data.find('tokenizers/punkt/english.pickle')
    except LookupError:
        nltk.download('punkt')
    
    # Verify all required data is available
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
        
except ImportError as e:
    st.error(f"Required packages not found. Please install them with: pip install transformers torch nltk")
    st.stop()
except Exception as e:
    st.error(f"Error initializing NLTK data: {str(e)}")
    st.stop()


# --- Global Variables & Models (Load once using Streamlit's cache for efficiency) ---
@st.cache_resource # Caches the loaded models across Streamlit sessions/reruns.
def load_nlp_models():
    """
    Loads and caches the NLP models required for the demo:
    - Sentiment analyzer.
    - Tokenizer and model for sentence embeddings.
    """
    logger.info("Loading NLP models...")
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    # Using a common sentence transformer model for generating embeddings.
    embedding_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    logger.info("NLP models loaded successfully.")
    return sentiment_analyzer, embedding_tokenizer, embedding_model

# Load models globally for use in functions.
sentiment_analyzer, embedding_tokenizer, embedding_model = load_nlp_models()

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
    Args:
        text: The input string containing one or more sentences.
    Returns:
        A 2D torch.Tensor of shape (num_sentences, embedding_dim), where each row
        is the embedding for a sentence. Returns None if no sentences are found.
    Purpose:
        These embeddings capture the semantic meaning of individual sentences,
        allowing for similarity comparisons or as features for downstream tasks.
    """
    sentences = nltk.sent_tokenize(text) # Split text into sentences
    if not sentences:
        logger.warning("No sentences found in text for embedding.")
        return None
    
    # Tokenize sentences and prepare input for the embedding model.
    inputs = embedding_tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt", max_length=128
    )
    with torch.no_grad(): # Disable gradient calculations for inference.
        outputs = embedding_model(**inputs)
    
    # Use mean pooling of the last hidden state to get sentence embeddings.
    sentence_embeds = outputs.last_hidden_state.mean(dim=1)
    return sentence_embeds

def analyze_character_sentiment_and_interaction(text: str, characters: List[str]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Analyzes text to determine sentiment associated with characters in each sentence
    and builds a character interaction matrix for the entire text.
    Args:
        text: The input string (e.g., a story section).
        characters: A list of character names to track.
    Returns:
        A tuple containing:
        - char_sentiment_data (Optional[torch.Tensor]): A 3D tensor
          [num_sentences, num_chars, 3 (neg, neut, pos)] representing one-hot encoded
          sentiment of each character in each sentence.
        - interaction_matrix (Optional[torch.Tensor]): A 2D tensor [num_chars, num_chars]
          counting co-occurrences of characters within the same sentences.
          Returns None for both if no sentences are found.
    """
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        logger.warning("No sentences found in text for character analysis.")
        return None, None

    num_chars = len(characters)
    # char_sentiment_data:
    # Dimension 1: Each sentence in the text.
    # Dimension 2: Each character in the 'characters' list.
    # Dimension 3: One-hot encoded sentiment [negative, neutral, positive].
    # e.g., char_sentiment_data[sent_idx, char_idx, 0] = 1 means char_idx had negative sentiment in sent_idx.
    char_sentiment_data = torch.zeros(len(sentences), num_chars, 3) 
    
    # interaction_matrix:
    # A square matrix where rows and columns correspond to character indices.
    # interaction_matrix[i, j] stores the number of sentences where character i and character j co-occur.
    # It's symmetric: interaction_matrix[i, j] == interaction_matrix[j, i].
    interaction_matrix = torch.zeros(num_chars, num_chars)
    
    # Pre-compile regex patterns for efficient character searching.
    char_patterns = {char: re.compile(r'\b' + re.escape(char) + r'\b', re.IGNORECASE) for char in characters}

    for i, sentence in enumerate(sentences):
        # Analyze overall sentiment of the current sentence.
        sent_sentiment_result = sentiment_analyzer(sentence)[0]
        sent_sentiment_category_idx = 1 # Default to neutral (index 1)
        if sent_sentiment_result['label'] == 'POSITIVE':
            sent_sentiment_category_idx = 2 # Positive sentiment (index 2)
        elif sent_sentiment_result['label'] == 'NEGATIVE':
            sent_sentiment_category_idx = 0 # Negative sentiment (index 0)

        # Identify which characters are present in the current sentence.
        present_chars_indices_in_sentence = []
        for char_idx, char_name in enumerate(characters):
            if char_patterns[char_name].search(sentence):
                present_chars_indices_in_sentence.append(char_idx)
                # Attribute the sentence's overall sentiment to each character present in it.
                # This is a simplification; more advanced methods could try to find character-specific sentiment.
                char_sentiment_data[i, char_idx, sent_sentiment_category_idx] = 1 

        # Update interaction matrix for characters co-occurring in this sentence.
        # If characters A, B, C are in a sentence, this increments counts for (A,B), (A,C), (B,C).
        for list_idx1 in range(len(present_chars_indices_in_sentence)):
            for list_idx2 in range(list_idx1, len(present_chars_indices_in_sentence)): # Use list_idx1 to avoid redundant pairs and self-interaction count twice
                actual_char_idx1 = present_chars_indices_in_sentence[list_idx1]
                actual_char_idx2 = present_chars_indices_in_sentence[list_idx2]
                
                interaction_matrix[actual_char_idx1, actual_char_idx2] += 1
                # If it's not a self-interaction, increment the symmetric position too.
                if actual_char_idx1 != actual_char_idx2:
                    interaction_matrix[actual_char_idx2, actual_char_idx1] += 1
                    
    return char_sentiment_data, interaction_matrix

# --- Data Ingestion Logic ---
def ingest_story_data(story_data: Dict, characters: List[str]):
    """
    Processes raw story data, generates various tensor representations for each section,
    and stores them in the (simulated) TensorStorage.
    """
    # Ensure datasets are created in TensorStorage before ingestion.
    for ds_name in [SENTENCE_EMBEDDINGS_DS, CHARACTER_SENTIMENT_DS, CHARACTER_INTERACTION_DS]:
        try:
            ts.create_dataset(ds_name)
            st.sidebar.info(f"Dataset '{ds_name}' created in simulated TensorStorage.")
        except ValueError: # Dataset already exists
            st.sidebar.info(f"Dataset '{ds_name}' already exists in simulated TensorStorage.")
            pass # Continue if dataset already exists, assuming it's from a previous partial run or similar.

    ingestion_progress = st.sidebar.progress(0.0, "Ingesting story sections...")
    total_sections = sum(len(ch_data["sections"]) for book_data in story_data.values() for ch_data in book_data["chapters"])
    processed_sections = 0

    for book_id, book_data in story_data.items():
        for chapter_data in book_data["chapters"]:
            for section_data in chapter_data["sections"]:
                section_id_short = section_data["section_id"]
                text_content = section_data["text"]
                # Create a unique full ID for each section to link different tensor types.
                full_section_id = f"{book_id}_{chapter_data['chapter_id']}_{section_id_short}"

                # Common metadata to be associated with all tensors derived from this section.
                base_metadata = {
                    "book_id": book_id,
                    "chapter_id": chapter_data['chapter_id'],
                    "chapter_title": chapter_data['title'],
                    "section_id": section_id_short, # Short section ID (s1, s2)
                    "full_id": full_section_id,    # Globally unique ID for the section
                    "_text_snippet_for_demo": text_content[:100] # For easy preview in UI
                }

                # --- Tensor 1: Sentence Embeddings (SENTENCE_EMBEDDINGS_DS) ---
                # - Information: Captures the semantic meaning of each sentence within the section.
                # - Dimensionality: 2D tensor [num_sentences_in_section, embedding_dimension].
                #   - Each row is an embedding vector for a sentence.
                # - Contribution: Allows for semantic search, finding similar sentences/passages,
                #   or tracking thematic shifts across sentences.
                sentence_embeds_tensor = get_sentence_embeddings(text_content)
                if sentence_embeds_tensor is not None:
                    ts.insert(SENTENCE_EMBEDDINGS_DS, sentence_embeds_tensor, {**base_metadata, "tensor_type": "sentence_embeddings"})

                # --- Tensors 2 & 3: Character Sentiment and Interaction ---
                char_sentiment_tensor, interaction_tensor = analyze_character_sentiment_and_interaction(text_content, characters)
                
                # --- Tensor 2: Character Sentiment Flow (CHARACTER_SENTIMENT_DS) ---
                # - Information: Tracks the sentiment associated with each character in each sentence of the section.
                # - Dimensionality: 3D tensor [num_sentences_in_section, num_characters, 3 (neg, neut, pos)].
                #   - For each sentence and character, a 3-element one-hot vector indicates sentiment.
                # - Contribution: Key for analyzing character sentiment evolution. Allows tracking how a character's
                #   perceived sentiment changes sentence by sentence, or aggregating it over sections/chapters.
                if char_sentiment_tensor is not None:
                    # Store character list with tensor for later interpretation of the character dimension.
                    ts.insert(CHARACTER_SENTIMENT_DS, char_sentiment_tensor, {**base_metadata, "tensor_type": "character_sentiment_flow", "characters_definition": characters})
                
                # --- Tensor 3: Character Interaction Matrix (CHARACTER_INTERACTION_DS) ---
                # - Information: Quantifies the co-occurrence of characters within sentences for the entire section.
                # - Dimensionality: 2D tensor [num_characters, num_characters].
                #   - Element (i, j) is a count of sentences where character i and character j appear together.
                # - Contribution: Helps identify which characters frequently interact, forming the basis for
                #   analyzing relationship strength and its evolution over the narrative.
                if interaction_tensor is not None:
                    ts.insert(CHARACTER_INTERACTION_DS, interaction_tensor, {**base_metadata, "tensor_type": "character_interaction_matrix", "characters_definition": characters})

                processed_sections += 1
                ingestion_progress.progress(processed_sections / total_sections if total_sections > 0 else 1.0, f"Ingested: {full_section_id}")
    
    ingestion_progress.empty() # Clear progress bar
    st.sidebar.success(f"Story data ingested ({processed_sections} sections processed).")
    st.session_state.data_ingested = True


# --- "Story Analyst" Agent Logic (Conceptual) ---
def analyze_character_evolution(target_char: str, related_char: str, book_id_filter: str):
    """
    Analyzes the evolution of interaction strength and sentiment for a target character
    in relation to another character throughout a specified book.
    Retrieves data from the (simulated) TensorStorage.
    """
    if target_char not in CHARACTERS_OF_INTEREST or related_char not in CHARACTERS_OF_INTEREST:
        st.error("Selected characters are not in the predefined list for analysis. Please re-ingest if new characters were added.")
        return pd.DataFrame(), pd.DataFrame()

    # --- Data Retrieval from EmbeddedTensorStorage ---
    # This simulates how an agent might query Tensorus.
    # For this demo, we retrieve all metadata first, then filter, then fetch specific tensors by ID.
    # This is a practical approach for in-memory simulation with potentially many small tensors.
    all_interaction_metadata = []
    all_sentiment_metadata = []
    try:
        # Check if datasets actually exist before querying.
        if CHARACTER_INTERACTION_DS in ts.datasets:
            all_interaction_metadata = ts.get_all_metadata_for_query(CHARACTER_INTERACTION_DS)
        if CHARACTER_SENTIMENT_DS in ts.datasets:
            all_sentiment_metadata = ts.get_all_metadata_for_query(CHARACTER_SENTIMENT_DS)
    except ValueError as e: # Raised by get_all_metadata_for_query if dataset doesn't exist
        st.error(f"Dataset not found during analysis: {e}. Was data ingested correctly?")
        return pd.DataFrame(), pd.DataFrame() # Return empty DataFrames

    # Filter metadata to include only records from the selected book.
    interaction_metadata_book = [m for m in all_interaction_metadata if m.get("book_id") == book_id_filter]
    sentiment_metadata_book = [m for m in all_sentiment_metadata if m.get("book_id") == book_id_filter]

    # Map character names to their indices used in the tensors (based on CHARACTERS_OF_INTEREST list).
    # This mapping must be consistent with how data was ingested.
    try:
        # Assuming 'characters_definition' stored in metadata during ingestion matches CHARACTERS_OF_INTEREST.
        # A more robust approach might be to fetch this definition from the first metadata record if available.
        char_map = {name: i for i, name in enumerate(CHARACTERS_OF_INTEREST)}
        target_char_idx = char_map[target_char]
        related_char_idx = char_map[related_char]
    except KeyError as e:
        st.error(f"Character mapping error: {e}. Ensure selected characters were in CHARACTERS_OF_INTEREST during ingestion.")
        return pd.DataFrame(), pd.DataFrame()

    evolution_data_points = [] # List to store data for DataFrame construction.

    # Get a unique, sorted list of all section IDs present in the filtered metadata for the book.
    # This ensures we process sections in narrative order and cover all relevant data points.
    section_full_ids_in_book = sorted(list(set(
        [m['full_id'] for m in interaction_metadata_book if 'full_id' in m] + 
        [m['full_id'] for m in sentiment_metadata_book if 'full_id' in m]
    )))

    # Iterate through each section to calculate interaction and sentiment.
    for full_id in section_full_ids_in_book:
        interaction_strength_in_section = 0.0 # Default to no interaction.
        avg_sentiment_score_for_target_char = 0.0 # Default to neutral sentiment.
        current_section_metadata = None # To store metadata for the current section.

        # --- Calculate Interaction Strength ---
        # Find the metadata for the interaction tensor of the current section.
        interaction_meta_entry = next((m for m in interaction_metadata_book if m.get("full_id") == full_id), None)
        if interaction_meta_entry:
            current_section_metadata = interaction_meta_entry # Use this section's metadata for reporting.
            # Retrieve the actual interaction tensor using its record_id.
            interaction_tensor = ts.get_tensor_by_record_id_from_list(CHARACTER_INTERACTION_DS, interaction_meta_entry['record_id'])
            if interaction_tensor is not None:
                # The interaction_tensor stores co-occurrence counts.
                # interaction_tensor[target_char_idx, related_char_idx] gives the number of sentences
                # in this section where both target_char and related_char appear.
                interaction_strength_in_section = interaction_tensor[target_char_idx, related_char_idx].item()

                # --- Calculate Average Sentiment Score (only if interaction occurs) ---
                # This logic ties sentiment analysis to interactions: we're interested in the target character's
                # sentiment in sections where they interact with the related character.
                if interaction_strength_in_section > 0:
                    # Find the metadata for the sentiment tensor of the current section.
                    sentiment_meta_entry = next((m for m in sentiment_metadata_book if m.get("full_id") == full_id), None)
                    if sentiment_meta_entry:
                        # Retrieve the actual sentiment tensor.
                        sentiment_tensor = ts.get_tensor_by_record_id_from_list(CHARACTER_SENTIMENT_DS, sentiment_meta_entry['record_id'])
                        if sentiment_tensor is not None: # Shape: [num_sentences, num_chars, 3 (neg, neut, pos)]
                            # Extract sentiment data for the target_char across all sentences in this section.
                            sentiments_for_target_char_in_section = sentiment_tensor[:, target_char_idx, :] # Shape: [num_sentences, 3]
                            
                            # Filter for sentences where the target character actually has a sentiment recorded
                            # (i.e., where the character was mentioned and sentiment was assigned).
                            valid_sentences_mask = torch.sum(sentiments_for_target_char_in_section, dim=1) > 0
                            if torch.any(valid_sentences_mask):
                                relevant_sentiments_one_hot = sentiments_for_target_char_in_section[valid_sentences_mask]
                                # Convert one-hot encoded sentiment to a numerical score:
                                # Positive (index 2) = +1, Neutral (index 1) = 0, Negative (index 0) = -1.
                                scores = relevant_sentiments_one_hot[:, 2] * 1.0 + \
                                         relevant_sentiments_one_hot[:, 1] * 0.0 + \
                                         relevant_sentiments_one_hot[:, 0] * (-1.0)
                                avg_sentiment_score_for_target_char = scores.float().mean().item()
                            # else: Character not mentioned, or no sentiment; avg_sentiment_score_for_target_char remains 0.0
                    # else: No sentiment metadata for this section; avg_sentiment_score_for_target_char remains 0.0 or could be np.nan
            # else: No interaction tensor for this section; interaction_strength_in_section remains 0.0
        
        # Add data point for this section if we have its metadata (ensures section exists).
        if current_section_metadata:
            evolution_data_points.append({
                "full_id": full_id, # Used for unique identification and sorting.
                "chapter": current_section_metadata.get("chapter_title", "N/A"),
                "section_id_val": current_section_metadata.get("section_id", "N/A"), 
                "interaction_strength": interaction_strength_in_section,
                f"{target_char}_sentiment_score": avg_sentiment_score_for_target_char
            })

    if not evolution_data_points:
        st.info(f"No relevant sections found for analyzing the relationship between {target_char} and {related_char} in this book.")
        return pd.DataFrame(), pd.DataFrame()

    df_evolution = pd.DataFrame(evolution_data_points)
    # Use 'full_id' for sorting to maintain narrative order.
    df_evolution['display_order'] = df_evolution['full_id'] 
    df_evolution = df_evolution.sort_values(by='display_order').reset_index(drop=True)

    # Prepare separate DataFrames for plotting, as required by the UI.
    df_interactions_plot = df_evolution[['display_order', 'chapter', 'interaction_strength']]
    df_sentiments_plot = df_evolution[['display_order', 'chapter', f"{target_char}_sentiment_score"]]
    
    return df_interactions_plot, df_sentiments_plot


# --- Streamlit UI ---
st.title("📚 Smart Story Analyzer Demo (Tensorus Concept)") # Title updated
st.write("Analyzes character relationships and sentiment evolution using a simulated, embedded TensorStorage.") # Caption updated

if 'data_ingested' not in st.session_state:
    st.session_state.data_ingested = False

# Initial check to see if data might have been ingested in a previous session run
# (Streamlit preserves session_state but not global variables like 'ts' if the script reruns fully)
# A more robust check would involve inspecting if 'ts' has the required datasets and they contain data.
try:
    if SENTENCE_EMBEDDINGS_DS in st.session_state.ts.datasets and \
       CHARACTER_SENTIMENT_DS in st.session_state.ts.datasets and \
       CHARACTER_INTERACTION_DS in st.session_state.ts.datasets:
        # This is a basic check. A stronger one would verify if these datasets have content.
        st.session_state.data_ingested = True 
except AttributeError: # ts not in session_state or ts.datasets doesn't exist
    st.session_state.data_ingested = False # Ensure it's false if ts isn't properly initialized


st.sidebar.header("Demo Setup & Control")
if not st.session_state.data_ingested :
    st.sidebar.markdown("Load the sample story data to enable analysis. This will populate the **simulated TensorStorage**.")
    if st.sidebar.button("Load and Ingest Sample Story Data"):
        with st.spinner("Processing text, generating tensors, and ingesting into simulated TensorStorage... This may take a moment for NLP models."):
            ingest_story_data(STORY_DATA, CHARACTERS_OF_INTEREST)
        st.rerun() 
else:
    st.sidebar.success("Sample story data loaded into the simulated TensorStorage.")
    st.sidebar.markdown("Re-ingest if you want to clear and reload all demo data.")
    if st.sidebar.button("Re-Ingest Data (Clears Simulated Data)"):
        # Re-initialize the TensorStorage to clear previous data for this demo.
        st.session_state.ts = EmbeddedTensorStorage() 
        ts = st.session_state.ts # Update the global reference.
        with st.spinner("Re-processing and re-ingesting story data into simulated TensorStorage..."):
            ingest_story_data(STORY_DATA, CHARACTERS_OF_INTEREST)
        st.rerun() 


if st.session_state.data_ingested:
    st.header("Character Evolution Analysis (Conceptual)")
    st.markdown("This section demonstrates how an AI agent might query the **simulated TensorStorage** to analyze character dynamics over the narrative.")
    
    # Assuming only one book for this demo's STORY_DATA for simplicity.
    selected_book_id = list(STORY_DATA.keys())[0] 
    st.subheader(f"Story: *{STORY_DATA[selected_book_id]['title']}*")
    
    col1, col2 = st.columns(2)
    with col1:
        main_char_options = CHARACTERS_OF_INTEREST
        main_character = st.selectbox("Select Main Character:", options=main_char_options, index=0, key="main_char_select")
    with col2:
        # Ensure related character options don't include the main character.
        related_character_options = [c for c in CHARACTERS_OF_INTEREST if c != main_character]
        if not related_character_options: # Should not happen with current CHARACTERS_OF_INTEREST
            st.warning("Not enough distinct characters available for relationship analysis.")
            related_character = None
        else:
            related_character = st.selectbox("Select Related Character:", options=related_character_options, index=0, key="related_char_select")

    if main_character and related_character and main_character != related_character:
        analyze_button_label = f"Analyze {main_character}'s Attitude & Interaction with {related_character}"
        if st.button(analyze_button_label):
            with st.spinner(f"Analyzing relationship between {main_character} and {related_character}... Fetching and processing tensors..."):
                # Call the analysis function which interacts with the simulated TensorStorage.
                df_interactions_plot, df_sentiments_plot = analyze_character_evolution(
                    main_character, related_character, selected_book_id
                )

            # Plotting interaction strength
            if df_interactions_plot is not None and not df_interactions_plot.empty:
                st.subheader(f"Interaction Strength: {main_character} & {related_character} Over Narrative")
                st.caption("Interaction strength is based on co-occurrence count in sentences within each story section. Higher values indicate more frequent co-mentions.")
                # Rename columns for better plot labels.
                df_interactions_plot = df_interactions_plot.rename(columns={'display_order': 'Narrative Section (Chronological)'})
                st.line_chart(df_interactions_plot.set_index('Narrative Section (Chronological)')['interaction_strength'])
                with st.expander("View Interaction Data Table"):
                    st.dataframe(df_interactions_plot)
            else:
                st.info(f"No significant interaction data found between {main_character} and {related_character} to plot.")

            # Plotting sentiment evolution
            if df_sentiments_plot is not None and not df_sentiments_plot.empty:
                st.subheader(f"Sentiment Score of {main_character} (in sections featuring {related_character})")
                st.caption(f"Sentiment score for {main_character} (from -1 Negative to +1 Positive) in sections where they interact with {related_character}. Based on average sentiment of sentences where {main_character} appears within those interactive sections.")
                sentiment_col_name = f"{main_character}_sentiment_score" # Dynamic column name from analysis function
                # Rename columns for better plot labels.
                df_sentiments_plot = df_sentiments_plot.rename(columns={'display_order': 'Narrative Section (Chronological)', sentiment_col_name: 'Sentiment Score'})
                
                if 'Sentiment Score' in df_sentiments_plot.columns:
                    st.line_chart(df_sentiments_plot.set_index('Narrative Section (Chronological)')['Sentiment Score'])
                    with st.expander("View Sentiment Data Table"):
                        st.dataframe(df_sentiments_plot)
                else:
                     st.warning(f"Sentiment data column '{sentiment_col_name}' not found in the processed data. Check analysis logic.")
            else:
                st.info(f"No significant sentiment data found for {main_character} in the context of {related_character} to plot.")
                
    elif main_character == related_character and main_character is not None:
        st.warning("Please select two different characters to analyze their relationship.")


    st.divider()
    st.header("Explore Stored Tensors (Simulated TensorStorage Peek)")
    st.markdown("This section allows you to peek into the raw tensor data stored in the **simulated TensorStorage** for the ingested story sections.")
    if ts.datasets: # Check if any datasets have been created in our EmbeddedTensorStorage.
        dataset_to_explore = st.selectbox("Select Dataset to Peek Into:", options=list(ts.datasets.keys()), key="explore_ds_select")
        if st.button("Show First 3 Records from Selected Dataset"):
            try:
                # Retrieve all records (tensor + metadata) for the chosen dataset.
                records = ts.get_dataset_with_metadata(dataset_to_explore) 
                if records:
                    st.write(f"Showing first {min(3, len(records))} of {len(records)} records from dataset **'{dataset_to_explore}'**:")
                    for r_idx, record_data in enumerate(records[:3]):
                        st.markdown(f"--- \n **Record {r_idx+1}**")
                        # Display metadata associated with the tensor.
                        st.json({"metadata": record_data.get('metadata', {})})
                        # Display tensor information (shape, dtype) and a small preview.
                        tensor_data = record_data.get('tensor')
                        if tensor_data is not None:
                            st.markdown(f"*Tensor Shape:* `{list(tensor_data.shape)}`")
                            st.markdown(f"*Tensor Dtype:* `{str(tensor_data.dtype)}`")
                            # Flatten and take first few elements for preview to handle various tensor shapes.
                            preview_elements = tensor_data.flatten()[:5].tolist()
                            st.markdown(f"*Tensor Preview (first 5 elements):* `{preview_elements}`")
                        else:
                            st.markdown("Tensor data not available for this record.")
                else:
                    st.info(f"No records found in dataset '{dataset_to_explore}'.")
            except ValueError as ve: # Catch errors like dataset not existing from get_dataset_with_metadata
                st.error(f"Error exploring dataset '{dataset_to_explore}': {ve}")
            except Exception as e_gen: # Catch any other unexpected errors
                st.error(f"An unexpected error occurred while exploring dataset: {e_gen}")
    else:
        st.info("No datasets have been created in the (simulated) TensorStorage yet. Please ingest data first.")

else:
    st.info("👈 Please use the sidebar to load and ingest the sample story data. This will enable the analysis features by populating the simulated TensorStorage.")