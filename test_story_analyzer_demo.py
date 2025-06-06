# Conceptual Unit Tests for story_analyzer_demo.py
# These tests are primarily structural and illustrative due to Streamlit
# and heavy NLP model/NLTK data dependencies.
# Actual execution in CI/CD would require extensive mocking.

import pytest
import torch
import pandas as pd
from unittest.mock import patch, MagicMock, call

# Attempt to import functions from the main script.
# This might require mocking global variables if they are accessed at import time.
try:
    from story_analyzer_demo import (
        get_sentence_embeddings,
        analyze_character_sentiment_and_interaction,
        ingest_story_data,
        analyze_character_evolution,
        STORY_DATA,
        CHARACTERS_OF_INTEREST,
        SENTENCE_EMBEDDINGS_DS,
        CHARACTER_SENTIMENT_DS,
        CHARACTER_INTERACTION_DS
    )
    from tensor_storage_utils import EmbeddedTensorStorage
except ImportError as e:
    print(f"Test file import error (story_analyzer_demo): {e}. This is somewhat expected for Streamlit apps without full environment setup.")
    # Define dummy versions if imports fail, to allow test structure definition
    def get_sentence_embeddings(text): return None
    def analyze_character_sentiment_and_interaction(text, characters): return None, None
    def ingest_story_data(story_data, characters): pass
    def analyze_character_evolution(target_char, related_char, book_id_filter): return pd.DataFrame(), pd.DataFrame()
    STORY_DATA = {}
    CHARACTERS_OF_INTEREST = []
    SENTENCE_EMBEDDINGS_DS = "sent_mock_ds"
    CHARACTER_SENTIMENT_DS = "char_sent_mock_ds"
    CHARACTER_INTERACTION_DS = "char_inter_mock_ds"
    EmbeddedTensorStorage = MagicMock


# --- Test Data ---
SAMPLE_TEXT_STORY = "Alice met the Queen. The Queen was stern. Alice felt uneasy."
EMPTY_TEXT_STORY = ""
CHAR_LIST_STORY = ["Alice", "Queen"]

# --- Fixtures ---
@pytest.fixture
def mock_nlp_models_story():
    """
    Provides mock objects for global Hugging Face models and NLTK for story_analyzer.
    Patches globals used by the functions under test.
    """
    with patch('story_analyzer_demo.sentiment_analyzer', autospec=True) as mock_sa, \
         patch('story_analyzer_demo.embedding_tokenizer', autospec=True) as mock_et, \
         patch('story_analyzer_demo.embedding_model', autospec=True) as mock_em, \
         patch('story_analyzer_demo.nltk.sent_tokenize') as mock_sent_tokenize:

        mock_sa.return_value = [{'label': 'NEUTRAL', 'score': 0.8}]
        mock_em_output = MagicMock()
        # Simulate batch_size=1, num_sentences=3, hidden_dim=384 for sentence_embeds.mean(dim=1)
        mock_em_output.last_hidden_state = torch.rand((3, 1, 384)) # For get_sentence_embeddings, after tokenizer for multiple sentences
        mock_em.return_value = mock_em_output # model(**inputs) returns this object

        # Tokenizer output for sentence embeddings (list of sentences)
        # This part is tricky as tokenizer is called with a list of sentences.
        # Let's assume the tokenizer output for a list of 3 sentences would have batch_dim=3 (or 1 if processed one by one)
        # For `get_sentence_embeddings`, inputs = embedding_tokenizer(sentences, ...)
        # So, embedding_tokenizer should be configured to handle a list of sentences.
        # The output of embedding_tokenizer itself is not directly used beyond passing to the model.
        # Its call signature is what matters if we were to assert calls on it.

        mock_sent_tokenize.return_value = SAMPLE_TEXT_STORY.split(". ")
        yield mock_sa, mock_et, mock_em, mock_sent_tokenize

@pytest.fixture
def mock_tensor_storage_story():
    """
    Provides a MagicMock instance for EmbeddedTensorStorage specific to story analysis.
    """
    mock_storage = MagicMock(spec=EmbeddedTensorStorage)
    mock_storage.create_dataset = MagicMock()
    mock_storage.insert = MagicMock(return_value="mock_record_id_story")
    mock_storage.get_all_metadata_for_query = MagicMock(return_value=[])
    mock_storage.get_tensor_by_record_id_from_list = MagicMock(return_value=None)
    mock_storage.datasets = {}
    mock_storage.list_datasets = MagicMock(return_value=[])
    return mock_storage

# --- Test Cases for NLP Processing Functions ---

def test_get_sentence_embeddings_sample_text(mock_nlp_models_story):
    """Test get_sentence_embeddings with typical sample text."""
    _, mock_et, mock_em, mock_sent_tokenize = mock_nlp_models_story
    sentences = ["Alice met the Queen.", "The Queen was stern.", "Alice felt uneasy."]
    mock_sent_tokenize.return_value = sentences
    # Adjust mock_em output if tokenizer processes sentences individually or as a batch
    # If as a batch, tokenizer output (inputs to model) would be like:
    # {'input_ids': tensor(... shape (num_sentences, seq_len)), ...}
    # Then model output last_hidden_state would be (num_sentences, seq_len, hidden_dim)
    # After .mean(dim=1), it becomes (num_sentences, hidden_dim)
    mock_em.return_value.last_hidden_state = torch.rand((len(sentences), 5, 384)) # NumSentences, SeqLen, HiddenDim

    embeddings = get_sentence_embeddings(SAMPLE_TEXT_STORY)

    assert embeddings is not None, "Embedding should not be None for valid text."
    assert isinstance(embeddings, torch.Tensor), "Embedding should be a torch.Tensor."
    assert embeddings.ndim == 2, "Sentence embeddings should be 2D."
    assert embeddings.shape[0] == len(sentences), "Dimension 0 should be number of sentences."
    assert embeddings.shape[1] == 384, "Embedding dimension should match model's hidden size."
    mock_sent_tokenize.assert_called_with(SAMPLE_TEXT_STORY)
    mock_et.assert_called_once()
    mock_em.assert_called_once()

def test_get_sentence_embeddings_empty_text(mock_nlp_models_story):
    """Test get_sentence_embeddings with an empty string. Expects None."""
    embeddings = get_sentence_embeddings(EMPTY_TEXT_STORY)
    assert embeddings is None, "Embedding should be None for empty text."

def test_analyze_character_sentiment_and_interaction_sample(mock_nlp_models_story):
    """Test analyze_character_sentiment_and_interaction with sample text and characters."""
    mock_sa, _, _, mock_sent_tokenize = mock_nlp_models_story
    sentences = ["Alice met the Queen.", "The Queen was stern."]
    mock_sent_tokenize.return_value = sentences
    mock_sa.side_effect = [
        [{'label': 'NEUTRAL', 'score': 0.7}],
        [{'label': 'NEGATIVE', 'score': 0.9}]
    ]

    char_sentiment, interaction_matrix = analyze_character_sentiment_and_interaction(SAMPLE_TEXT_STORY, CHAR_LIST_STORY)

    assert char_sentiment is not None, "Character sentiment tensor should not be None."
    assert interaction_matrix is not None, "Interaction matrix should not be None."
    assert char_sentiment.shape == (len(sentences), len(CHAR_LIST_STORY), 3), "Char sentiment tensor shape mismatch."
    assert interaction_matrix.shape == (len(CHAR_LIST_STORY), len(CHAR_LIST_STORY)), "Interaction matrix shape mismatch."
    # Example: Alice (idx 0) in first sentence ("Alice met the Queen.") - NEUTRAL (idx 1)
    # Queen (idx 1) in second sentence ("The Queen was stern.") - NEGATIVE (idx 0)
    assert torch.argmax(char_sentiment[0, CHAR_LIST_STORY.index("Alice"), :]) == 1
    assert torch.argmax(char_sentiment[1, CHAR_LIST_STORY.index("Queen"), :]) == 0
    # Check interaction: Alice and Queen interact in the first sentence.
    assert interaction_matrix[CHAR_LIST_STORY.index("Alice"), CHAR_LIST_STORY.index("Queen")] >= 1


def test_analyze_character_sentiment_and_interaction_no_chars_in_text(mock_nlp_models_story):
    """Test with text not containing specified characters."""
    text_no_chars = "A generic sentence without specific names."
    _, _, _, mock_sent_tokenize = mock_nlp_models_story
    mock_sent_tokenize.return_value = [text_no_chars]

    char_sentiment, interaction_matrix = analyze_character_sentiment_and_interaction(text_no_chars, CHAR_LIST_STORY)

    assert char_sentiment is not None
    assert interaction_matrix is not None
    assert torch.all(interaction_matrix == 0), "Interaction matrix should be all zeros if no characters are found."
    # Sentiment for characters should be all zeros as they are not mentioned.
    assert torch.all(char_sentiment == 0)


def test_analyze_character_sentiment_and_interaction_empty_text(mock_nlp_models_story):
    """Test with empty input text. Expects (None, None)."""
    char_sentiment, interaction_matrix = analyze_character_sentiment_and_interaction(EMPTY_TEXT_STORY, CHAR_LIST_STORY)
    assert char_sentiment is None, "Character sentiment should be None for empty text."
    assert interaction_matrix is None, "Interaction matrix should be None for empty text."

# --- Test Cases for Data Ingestion and Analysis Logic ---

@patch('story_analyzer_demo.ts', new_callable=MagicMock)
@patch('story_analyzer_demo.get_sentence_embeddings')
@patch('story_analyzer_demo.analyze_character_sentiment_and_interaction')
def test_ingest_story_data_mocked_logic(
    mock_analyze_char_sent_inter,
    mock_get_sent_embeds,
    mock_ts_story,
    mock_nlp_models_story
):
    """Test ingest_story_data focusing on interactions with TensorStorage."""
    mock_get_sent_embeds.return_value = torch.rand(1, 384)
    mock_analyze_char_sent_inter.return_value = (torch.rand(1, len(CHAR_LIST_STORY), 3), torch.rand(len(CHAR_LIST_STORY), len(CHAR_LIST_STORY)))

    test_story_data = {"book_1": {"title": "Test Book", "chapters": [
        {"chapter_id": "ch1", "title": "Ch 1", "sections": [
            {"section_id": "s1", "text": "Alice walks."}
        ]}
    ]}}

    with patch('story_analyzer_demo.st') as mock_st_ingest:
        ingest_story_data(test_story_data, CHAR_LIST_STORY)

    assert mock_ts_story.create_dataset.call_count >= 3 # Called for each dataset name
    assert mock_ts_story.insert.call_count == 3

    dataset_calls = [c[0][0] for c in mock_ts_story.insert.call_args_list]
    assert SENTENCE_EMBEDDINGS_DS in dataset_calls
    assert CHARACTER_SENTIMENT_DS in dataset_calls
    assert CHARACTER_INTERACTION_DS in dataset_calls

    first_call_args = mock_ts_story.insert.call_args_list[0][0]
    assert first_call_args[2]["book_id"] == "book_1"
    assert first_call_args[2]["tensor_type"] is not None


@patch('story_analyzer_demo.ts', new_callable=MagicMock)
def test_analyze_character_evolution_mocked_storage(mock_ts_analysis, mock_nlp_models_story):
    """Test character evolution analysis logic with mocked TensorStorage returns."""
    target_char, related_char, book_id = "Alice", "Queen", "book_1"
    # Use a CHARACTERS_OF_INTEREST list consistent with how indices would be derived
    current_char_list = ["Alice", "Queen", "Rabbit"]
    alice_idx, queen_idx = current_char_list.index("Alice"), current_char_list.index("Queen")

    mock_interaction_meta = [{"book_id": book_id, "full_id": "b1_ch1_s1", "record_id": "inter_s1", "characters_definition": current_char_list, "chapter_title": "C1", "section_id": "s1"}]
    mock_sentiment_meta = [{"book_id": book_id, "full_id": "b1_ch1_s1", "record_id": "sent_s1", "characters_definition": current_char_list, "chapter_title": "C1", "section_id": "s1"}]

    def get_meta_side_effect(dataset_name):
        if dataset_name == CHARACTER_INTERACTION_DS: return mock_interaction_meta
        if dataset_name == CHARACTER_SENTIMENT_DS: return mock_sentiment_meta
        return []
    mock_ts_analysis.get_all_metadata_for_query.side_effect = get_meta_side_effect
    mock_ts_analysis.datasets = {CHARACTER_INTERACTION_DS: {}, CHARACTER_SENTIMENT_DS: {}}

    interaction_s1 = torch.zeros(len(current_char_list), len(current_char_list))
    interaction_s1[alice_idx, queen_idx] = 3
    interaction_s1[queen_idx, alice_idx] = 3
    sentiment_s1 = torch.zeros(1, len(current_char_list), 3)
    sentiment_s1[0, alice_idx, 2] = 1 # Alice: Positive (idx 2)

    def get_tensor_side_effect(dataset_name, record_id):
        if record_id == "inter_s1": return interaction_s1
        if record_id == "sent_s1": return sentiment_s1
        return None
    mock_ts_analysis.get_tensor_by_record_id_from_list.side_effect = get_tensor_side_effect

    # Patch the global CHARACTERS_OF_INTEREST for the duration of this test
    with patch('story_analyzer_demo.CHARACTERS_OF_INTEREST', current_char_list):
        df_interactions, df_sentiments = analyze_character_evolution(target_char, related_char, book_id)

    assert not df_interactions.empty, "Interactions DataFrame should not be empty."
    assert not df_sentiments.empty, "Sentiments DataFrame should not be empty."
    assert len(df_interactions) == 1, "Expected one row for one section in interactions."
    assert df_interactions.iloc[0]['interaction_strength'] == 3, "Interaction strength mismatch."
    assert df_sentiments.iloc[0][f'{target_char}_sentiment_score'] == 1.0, "Sentiment score mismatch."

def test_analyze_character_evolution_no_data_in_storage(mock_ts_analysis):
    """Test character evolution when TensorStorage returns no relevant metadata."""
    mock_ts_analysis.get_all_metadata_for_query.return_value = []
    mock_ts_analysis.datasets = {CHARACTER_INTERACTION_DS: {}, CHARACTER_SENTIMENT_DS: {}}

    df_interactions, df_sentiments = analyze_character_evolution("Alice", "Queen", "book_1")

    assert df_interactions.empty, "Interactions DataFrame should be empty if no data."
    assert df_sentiments.empty, "Sentiments DataFrame should be empty if no data."

# Note: To run these tests, one would typically use `pytest test_story_analyzer_demo.py`
# from the terminal. Ensure mocks are correctly configured to prevent actual NLP model
# loading or Streamlit UI interactions during automated testing.
# The ImportError handling at the top is a fallback for environments where
# the main script's dependencies might not be fully available or might trigger
# unwanted actions (like UI rendering) upon import.
