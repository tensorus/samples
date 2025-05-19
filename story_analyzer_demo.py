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
st.set_page_config(page_title="Tensorus Story Analyzer", layout="wide")

# --- Configure basic logging (optional, but good for the embedded TensorStorage) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Embedded TensorStorage Class (adapted from your tensorus/tensor_storage.py) ---
class EmbeddedTensorStorage:
    """
    Manages datasets stored as collections of tensors in memory.
    (Adapted for self-contained demo)
    """
    def __init__(self):
        self.datasets: Dict[str, Dict[str, List[Any]]] = {}
        logger.info("EmbeddedTensorStorage initialized (In-Memory).")

    def create_dataset(self, name: str) -> None:
        if name in self.datasets:
            logger.warning(f"Attempted to create dataset '{name}' which already exists.")
            raise ValueError(f"Dataset '{name}' already exists.")
        self.datasets[name] = {"tensors": [], "metadata": []}
        logger.info(f"Dataset '{name}' created successfully.")

    def insert(self, name: str, tensor: torch.Tensor, metadata: Optional[Dict[str, Any]] = None) -> str:
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' does not exist. Create it first.")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Data to be inserted must be a torch.Tensor.")

        if metadata is None:
            metadata = {}

        record_id = str(uuid.uuid4())
        default_metadata = {
            "record_id": record_id,
            "timestamp_utc": time.time(),
            "shape": list(tensor.shape), # Ensure it's a list for JSON later
            "dtype": str(tensor.dtype).replace('torch.', ''), # Clean dtype string
            "version": len(self.datasets[name]["tensors"]) + 1,
        }
        final_metadata = {**default_metadata, **metadata} # User metadata can override non-essential defaults

        self.datasets[name]["tensors"].append(tensor.clone())
        self.datasets[name]["metadata"].append(final_metadata)
        logger.debug(f"Tensor with ID {record_id} inserted into dataset '{name}'.")
        return record_id

    def get_dataset_with_metadata(self, name: str) -> List[Dict[str, Any]]:
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' does not exist.")
        results = []
        for tensor, meta in zip(self.datasets[name]["tensors"], self.datasets[name]["metadata"]):
            results.append({"tensor": tensor, "metadata": meta})
        return results

    def query(self, name: str, query_fn: Callable[[torch.Tensor, Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' does not exist.")
        if not callable(query_fn):
             raise TypeError("query_fn must be a callable function.")
        
        results = []
        for tensor, meta in zip(self.datasets[name]["tensors"], self.datasets[name]["metadata"]):
            try:
                if query_fn(tensor, meta): # query_fn here expects tensor as first arg
                    results.append({"tensor": tensor, "metadata": meta})
            except Exception as e:
                logger.warning(f"Error executing query_fn on record {meta.get('record_id', 'N/A')}: {e}")
                continue
        return results
    
    def get_all_metadata_for_query(self, name: str) -> List[Dict[str, Any]]:
        """Helper to get all metadata, useful for initial filtering before loading tensors"""
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' does not exist.")
        return list(self.datasets[name]["metadata"]) # Return copies

    def get_tensor_by_record_id_from_list(self, name: str, record_id: str) -> Optional[torch.Tensor]:
        """Helper to get a tensor if metadata query already identified the record"""
        if name not in self.datasets:
            return None
        for i, meta in enumerate(self.datasets[name]["metadata"]):
            if meta.get("record_id") == record_id:
                return self.datasets[name]["tensors"][i]
        return None


# --- End of Embedded TensorStorage ---


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


# --- Global Variables & Models (Load once) ---
@st.cache_resource
def load_nlp_models():
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    embedding_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return sentiment_analyzer, embedding_tokenizer, embedding_model

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


# --- NLP Processing Functions (modified slightly for clarity if needed) ---
def get_sentence_embeddings(text: str) -> Optional[torch.Tensor]:
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return None
    inputs = embedding_tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=128)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    sentence_embeds = outputs.last_hidden_state.mean(dim=1)
    return sentence_embeds

def analyze_character_sentiment_and_interaction(text: str, characters: List[str]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return None, None

    num_chars = len(characters)
    char_sentiment_data = torch.zeros(len(sentences), num_chars, 3) # neg, neut, pos
    interaction_matrix = torch.zeros(num_chars, num_chars)
    char_patterns = {char: re.compile(r'\b' + re.escape(char) + r'\b', re.IGNORECASE) for char in characters}

    for i, sentence in enumerate(sentences):
        sent_sentiment_result = sentiment_analyzer(sentence)[0]
        sent_sentiment_score_cat = 1 # neutral index
        if sent_sentiment_result['label'] == 'POSITIVE':
            sent_sentiment_score_cat = 2 # positive index
        elif sent_sentiment_result['label'] == 'NEGATIVE':
            sent_sentiment_score_cat = 0 # negative index

        present_chars_indices_in_sentence = []
        for char_idx, char_name in enumerate(characters):
            if char_patterns[char_name].search(sentence):
                present_chars_indices_in_sentence.append(char_idx)
                char_sentiment_data[i, char_idx, sent_sentiment_score_cat] = 1 # One-hot encode sentiment for this char in this sentence

        for char_list_idx1 in range(len(present_chars_indices_in_sentence)):
            for char_list_idx2 in range(char_list_idx1, len(present_chars_indices_in_sentence)):
                actual_char_idx1 = present_chars_indices_in_sentence[char_list_idx1]
                actual_char_idx2 = present_chars_indices_in_sentence[char_list_idx2]
                interaction_matrix[actual_char_idx1, actual_char_idx2] += 1
                if actual_char_idx1 != actual_char_idx2:
                    interaction_matrix[actual_char_idx2, actual_char_idx1] += 1
    return char_sentiment_data, interaction_matrix

# --- Data Ingestion Logic ---
def ingest_story_data(story_data: Dict, characters: List[str]):
    for ds_name in [SENTENCE_EMBEDDINGS_DS, CHARACTER_SENTIMENT_DS, CHARACTER_INTERACTION_DS]:
        try:
            ts.create_dataset(ds_name)
            st.sidebar.info(f"Dataset '{ds_name}' created.")
        except ValueError:
            st.sidebar.info(f"Dataset '{ds_name}' already exists.")
            pass

    ingestion_progress = st.sidebar.progress(0.0)
    total_sections = sum(len(ch_data["sections"]) for book_data in story_data.values() for ch_data in book_data["chapters"])
    processed_sections = 0

    for book_id, book_data in story_data.items():
        for chapter_data in book_data["chapters"]:
            for section_data in chapter_data["sections"]:
                section_id = section_data["section_id"]
                text = section_data["text"]
                full_id = f"{book_id}_{chapter_data['chapter_id']}_{section_id}"

                metadata = {
                    "book_id": book_id,
                    "chapter_id": chapter_data['chapter_id'],
                    "chapter_title": chapter_data['title'],
                    "section_id": section_id,
                    "full_id": full_id,
                    "_text_snippet_for_demo": text[:100] # Store a snippet for easier manual review in demo
                }

                sentence_embeds_tensor = get_sentence_embeddings(text)
                if sentence_embeds_tensor is not None:
                    ts.insert(SENTENCE_EMBEDDINGS_DS, sentence_embeds_tensor, {**metadata, "tensor_type": "sentence_embeddings"})

                char_sentiment_tensor, interaction_tensor = analyze_character_sentiment_and_interaction(text, characters)
                if char_sentiment_tensor is not None:
                    ts.insert(CHARACTER_SENTIMENT_DS, char_sentiment_tensor, {**metadata, "tensor_type": "character_sentiment_flow", "characters_definition": characters}) # Store how characters map to indices
                if interaction_tensor is not None:
                    ts.insert(CHARACTER_INTERACTION_DS, interaction_tensor, {**metadata, "tensor_type": "character_interaction_matrix", "characters_definition": characters})

                processed_sections += 1
                ingestion_progress.progress(processed_sections / total_sections if total_sections > 0 else 1.0)

    ingestion_progress.empty()
    st.sidebar.success(f"Story data ingested ({processed_sections} sections).")
    st.session_state.data_ingested = True


# --- "Story Analyst" Agent Logic ---
def analyze_character_evolution(target_char: str, related_char: str, book_id_filter: str):
    if target_char not in CHARACTERS_OF_INTEREST or related_char not in CHARACTERS_OF_INTEREST:
        st.error("Selected characters are not in the predefined list for analysis.")
        return pd.DataFrame(), pd.DataFrame()

    # --- Retrieve data using the embedded TensorStorage ---
    # Option 1: Use the existing query mechanism if it can filter metadata efficiently
    # Option 2: Get all metadata, filter, then get tensors by ID (might be simpler for this demo structure)

    # Let's use a metadata-first approach for this demo given our simpler embedded storage
    all_interaction_metadata = []
    all_sentiment_metadata = []
    try:
        if CHARACTER_INTERACTION_DS in ts.datasets:
            all_interaction_metadata = ts.get_all_metadata_for_query(CHARACTER_INTERACTION_DS)
        if CHARACTER_SENTIMENT_DS in ts.datasets:
            all_sentiment_metadata = ts.get_all_metadata_for_query(CHARACTER_SENTIMENT_DS)
    except ValueError as e:
        st.error(f"Dataset not found during analysis: {e}")
        return pd.DataFrame(), pd.DataFrame()


    # Filter metadata for the current book
    interaction_metadata_book = [m for m in all_interaction_metadata if m.get("book_id") == book_id_filter]
    sentiment_metadata_book = [m for m in all_sentiment_metadata if m.get("book_id") == book_id_filter]

    # Define character indices based on CHARACTERS_OF_INTEREST used during ingestion
    # This assumes the 'characters_definition' in metadata matches CHARACTERS_OF_INTEREST
    try:
        char_map = {name: i for i, name in enumerate(CHARACTERS_OF_INTEREST)}
        target_char_idx = char_map[target_char]
        related_char_idx = char_map[related_char]
    except KeyError:
        st.error("Character mapping error. Ensure characters were in CHARACTERS_OF_INTEREST during ingestion.")
        return pd.DataFrame(), pd.DataFrame()

    evolution_data = []

    # Create a combined, sorted list of section contexts for the book
    # We'll use full_id to map between interaction and sentiment data for a section
    section_full_ids_in_book = sorted(list(set(
        [m['full_id'] for m in interaction_metadata_book] + [m['full_id'] for m in sentiment_metadata_book]
    )))


    for full_id in section_full_ids_in_book:
        interaction_strength = 0
        avg_sentiment_score = 0
        relevant_section_metadata = None

        # Find interaction tensor for this section
        interaction_meta_entry = next((m for m in interaction_metadata_book if m.get("full_id") == full_id), None)
        if interaction_meta_entry:
            relevant_section_metadata = interaction_meta_entry # Grab metadata once
            interaction_tensor = ts.get_tensor_by_record_id_from_list(CHARACTER_INTERACTION_DS, interaction_meta_entry['record_id'])
            if interaction_tensor is not None:
                # Check if both characters are "present" or "interact" in this section
                # This check depends on how interaction_tensor is defined.
                # If it's co-occurrence, then interaction_tensor[target_idx, related_idx] > 0 implies presence.
                if interaction_tensor[target_char_idx, related_char_idx].item() > 0:
                    interaction_strength = interaction_tensor[target_char_idx, related_char_idx].item()

                    # Find corresponding sentiment tensor for this section
                    sentiment_meta_entry = next((m for m in sentiment_metadata_book if m.get("full_id") == full_id), None)
                    if sentiment_meta_entry:
                        sentiment_tensor = ts.get_tensor_by_record_id_from_list(CHARACTER_SENTIMENT_DS, sentiment_meta_entry['record_id'])
                        if sentiment_tensor is not None: # Shape: [num_sentences, num_chars, 3 (neg, neut, pos)]
                            # Calculate avg sentiment for target_char in sentences where they appear in this section
                            target_char_sentiments_in_section = sentiment_tensor[:, target_char_idx, :] # [num_sentences, 3]
                            
                            # Consider only sentences where target_char is actually mentioned (has a sentiment entry)
                            valid_sentences_mask = torch.sum(target_char_sentiments_in_section, dim=1) > 0
                            if torch.any(valid_sentences_mask):
                                relevant_sentiments = target_char_sentiments_in_section[valid_sentences_mask]
                                # Convert one-hot sentiment to a score: POSITIVE=1, NEUTRAL=0, NEGATIVE=-1
                                scores = relevant_sentiments[:, 2] * 1 + relevant_sentiments[:, 1] * 0 + relevant_sentiments[:, 0] * (-1)
                                avg_sentiment_score = scores.float().mean().item()
                            else:
                                avg_sentiment_score = 0 # No sentences with the target character
                        else:
                            avg_sentiment_score = np.nan # Sentiment tensor missing
                    else:
                        avg_sentiment_score = np.nan # Sentiment metadata missing
                else:
                    # If no interaction strength, sentiment might not be as relevant for this specific pairing
                    interaction_strength = 0 # Ensure it's zero if no direct interaction
                    avg_sentiment_score = 0 # Or np.nan if you prefer to plot gaps

        if relevant_section_metadata: # Only add if we have some base metadata for the section
            evolution_data.append({
                "full_id": full_id,
                "chapter": relevant_section_metadata.get("chapter_title", "N/A"),
                "section_id_val": relevant_section_metadata.get("section_id", "N/A"), # for sorting
                "interaction_strength": interaction_strength,
                f"{target_char}_sentiment_score": avg_sentiment_score
            })

    if not evolution_data:
        st.info(f"No relevant sections found for {target_char} and {related_char} in this book.")
        return pd.DataFrame(), pd.DataFrame()

    df_evolution = pd.DataFrame(evolution_data)
    # Create a sortable combined key for plotting if chapter/section IDs are sortable strings
    df_evolution['display_order'] = df_evolution['full_id'] # Or a numeric sequence if available
    df_evolution = df_evolution.sort_values(by='display_order')

    return df_evolution[['display_order', 'chapter', 'interaction_strength']], \
           df_evolution[['display_order', 'chapter', f"{target_char}_sentiment_score"]]


# --- Streamlit UI ---
st.title("📚 Smart Story Analyzer Demo (Tensorus Concept)")
st.write("Analyzes character relationships and sentiment evolution using an embedded TensorStorage.")

if 'data_ingested' not in st.session_state:
    st.session_state.data_ingested = False

# Check if datasets exist (simple check, assumes if one exists, all might)
try:
    if SENTENCE_EMBEDDINGS_DS in ts.datasets and CHARACTER_SENTIMENT_DS in ts.datasets and CHARACTER_INTERACTION_DS in ts.datasets:
        # A bit of a heuristic: if datasets exist, assume data might have been ingested in a previous run
        # A more robust check would be to see if they contain data for STORY_DATA
        st.session_state.data_ingested = True
except Exception: # If ts.datasets itself is an issue
    pass


st.sidebar.header("Setup")
if not st.session_state.data_ingested :
    if st.sidebar.button("Load and Ingest Sample Story Data"):
        with st.spinner("Processing and ingesting story data... This may take a few minutes for NLP models."):
            ingest_story_data(STORY_DATA, CHARACTERS_OF_INTEREST)
        st.rerun()  # Use current Streamlit method to rerun the app
else:
    st.sidebar.success("Sample story data is loaded into TensorStorage.")
    if st.sidebar.button("Re-Ingest Data (Clears Existing Demo Data)"):
        # Simple clear for demo: re-initialize TensorStorage for these datasets
        st.session_state.ts = EmbeddedTensorStorage() # Re-init
        ts = st.session_state.ts # update global ref
        with st.spinner("Re-ingesting story data..."):
            ingest_story_data(STORY_DATA, CHARACTERS_OF_INTEREST)
        st.rerun()  # Use current Streamlit method to rerun the app


if st.session_state.data_ingested:
    st.header("Character Evolution Analysis")
    
    # Assuming only one book for this demo's STORY_DATA
    selected_book_id = list(STORY_DATA.keys())[0]
    st.subheader(f"Story: {STORY_DATA[selected_book_id]['title']}")
    
    col1, col2 = st.columns(2)
    with col1:
        main_character = st.selectbox("Select Main Character", options=CHARACTERS_OF_INTEREST, index=0, key="main_char_select")
    with col2:
        related_character_options = [c for c in CHARACTERS_OF_INTEREST if c != main_character]
        if not related_character_options:
            st.warning("Need at least two distinct characters from the list to analyze a relationship.")
            related_character = None
        else:
            related_character = st.selectbox("Select Related Character", options=related_character_options, index=0 if related_character_options else 0,  key="related_char_select")

    if main_character and related_character and main_character != related_character:
        if st.button(f"Analyze {main_character}'s Attitude Towards {related_character}"):
            with st.spinner(f"Analyzing evolution..."):
                df_interactions_plot, df_sentiments_plot = analyze_character_evolution(main_character, related_character, selected_book_id)

            if df_interactions_plot is not None and not df_interactions_plot.empty:
                st.subheader(f"Interaction Strength: {main_character} & {related_character}")
                st.caption("Interaction strength based on co-occurrence in sentences within sections. Higher is more frequent.")
                df_interactions_plot = df_interactions_plot.rename(columns={'display_order': 'Narrative Section'})
                st.line_chart(df_interactions_plot.set_index('Narrative Section')['interaction_strength'])
                with st.expander("Show Interaction Data Table"):
                    st.dataframe(df_interactions_plot)
            else:
                st.info(f"No significant interaction data found between {main_character} and {related_character} for combined analysis.")

            if df_sentiments_plot is not None and not df_sentiments_plot.empty:
                st.subheader(f"Sentiment Score of {main_character} (in sections also featuring {related_character})")
                st.caption(f"Sentiment: -1 (Negative) to 1 (Positive). Based on sentences where {main_character} appears.")
                sentiment_col_name = f"{main_character}_sentiment_score"
                df_sentiments_plot = df_sentiments_plot.rename(columns={'display_order': 'Narrative Section', sentiment_col_name: 'Sentiment Score'})

                if 'Sentiment Score' in df_sentiments_plot.columns:
                    st.line_chart(df_sentiments_plot.set_index('Narrative Section')['Sentiment Score'])
                    with st.expander("Show Sentiment Data Table"):
                        st.dataframe(df_sentiments_plot)
                else:
                     st.warning(f"Sentiment data column for {main_character} not found as expected.")
            else:
                st.info(f"No significant sentiment data found for {main_character} in the context of {related_character} for combined analysis.")
    elif main_character == related_character and main_character is not None:
        st.warning("Please select two different characters.")


    st.divider()
    st.header("Explore Stored Tensors (Raw Data Peek)")
    if ts.datasets: # Check if there are any datasets at all
        dataset_to_explore = st.selectbox("Select Dataset to Peek Into", options=list(ts.datasets.keys()), key="explore_ds_select")
        if st.button("Show First 3 Records from Selected Dataset"):
            try:
                records = ts.get_dataset_with_metadata(dataset_to_explore) # Use the embedded one
                if records:
                    st.write(f"Showing first {min(3, len(records))} of {len(records)} records from '{dataset_to_explore}':")
                    for r in records[:3]:
                        st.json({ # Use st.json for better display of dicts
                            "metadata": r.get('metadata', {}),
                            "tensor_shape": list(r['tensor'].shape) if 'tensor' in r and hasattr(r['tensor'], 'shape') else "N/A",
                            "tensor_dtype": str(r['tensor'].dtype) if 'tensor' in r and hasattr(r['tensor'], 'dtype') else "N/A",
                            "tensor_preview (first few elements)": str(r['tensor'].flatten()[:5].tolist()) if ('tensor' in r and r['tensor'].numel() > 0) else "Scalar/Empty"
                        })
                        st.markdown("---")
                else:
                    st.info(f"No records in dataset '{dataset_to_explore}'.")
            except ValueError as e:
                st.error(f"Error exploring dataset '{dataset_to_explore}': {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred while exploring: {e}")
    else:
        st.info("No datasets have been created in TensorStorage yet.")

else:
    st.info("Please load and ingest sample story data from the sidebar to enable analysis.")