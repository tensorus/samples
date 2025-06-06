# Story Analyzer Demo with Simulated TensorStorage

This demo application, "Smart Story Analyzer," showcases how a simulated Tensor Database (using `EmbeddedTensorStorage`) can be used to analyze character relationships and sentiment evolution in literary texts. It leverages NLP techniques to process text and store derived tensor representations.

**Note:** This demo uses an *in-memory simulation* (`EmbeddedTensorStorage`) for illustrative purposes and does not connect to a full, external Tensor Database instance.

## Purpose

The demo aims to illustrate:
*   The generation and storage of diverse tensor types from text (sentence embeddings, character sentiment flows, interaction matrices).
*   How these structured tensors can be queried and processed to derive insights about narrative elements, such as the evolution of character sentiment and relationships.
*   The conceptual advantages of a Tensor Database approach for managing and analyzing complex, multi-faceted data from text.

## Key Features

*   **Literary Text Analysis**: Uses snippets from "Alice's Adventures in Wonderland" as sample data.
*   **Tensor Generation**:
    *   Sentence Embeddings (2D Tensor): Captures semantic meaning of individual sentences.
    *   Character Sentiment Flow (3D Tensor): Tracks sentiment (Negative, Neutral, Positive) for each character in each sentence.
    *   Character Interaction Matrix (2D Tensor): Quantifies co-occurrence of characters within sentences for each text section.
*   **Dynamic Analysis**:
    *   Analyzes interaction strength between selected characters.
    *   Tracks sentiment evolution for a target character, optionally in the context of interactions with another character.
*   **NLP Integration**:
    *   Uses Hugging Face `transformers` for sentence embeddings and sentiment analysis.
    *   Utilizes `nltk` (Natural Language Toolkit) for sentence tokenization. NLTK data (`punkt` and `stopwords`) will be downloaded automatically on the first run if not found.
*   **Interactive UI**: Built with Streamlit for selecting characters and visualizing analysis results.
*   **Simulated TensorStorage**: All generated tensors and metadata are stored and managed by the `EmbeddedTensorStorage` class from `tensor_storage_utils.py`.

## Setup and Running the Demo

1.  **Prerequisites**:
    *   Ensure Python 3.8+ is installed.
    *   It's highly recommended to use a Python virtual environment.

2.  **Clone the Repository and Install Dependencies**:
    *   Clone this repository to your local machine.
    *   Navigate to the repository's root directory.
    *   Install the required packages by running:
        ```bash
        pip install -r requirements.txt
        ```
    *   (The `requirements.txt` file should be present in the root of the repository).

3.  **Run the Demo**:
    *   Open your terminal and ensure your virtual environment is activated.
    *   Navigate to the repository's root directory.
    *   Execute the command:
        ```bash
        streamlit run story_analyzer_demo.py
        ```
    *   The application will open in your web browser.

4.  **NLTK Data Download (First Run)**:
    *   On its first run, the application will check for necessary NLTK data packages (`punkt` for sentence tokenization and `stopwords`). If these are not found, it will attempt to download them automatically. Please ensure you have an internet connection during the first run. Subsequent runs will use the cached data.

## How to Interact with the Demo

1.  **Load and Ingest Data**:
    *   In the sidebar, click the "Load and Ingest Sample Story Data" button.
    *   A spinner will indicate that the story snippets are being processed, tensors are being generated, and data is being ingested into the simulated TensorStorage. This might take a few moments, especially on the first run due to NLP model loading.
    *   A success message will appear in the sidebar upon completion.

2.  **Analyze Character Evolution**:
    *   Once the data is ingested, the "Character Evolution Analysis" section will become available.
    *   Select a "Main Character" from the first dropdown.
    *   Optionally, select a "Related Character" from the second dropdown if you want to analyze interactions and the main character's sentiment in the context of this related character. Select "None" if you only want to see the main character's overall sentiment evolution.
    *   Click the "Analyze..." button.
    *   The application will display line charts showing:
        *   Interaction strength between the selected characters over the narrative sections (if a related character is chosen).
        *   Sentiment score for the main character over the narrative sections.
    *   You can expand sections to view the underlying data tables.

3.  **Explore Stored Tensors**:
    *   In the "Explore Stored Tensors (Simulated TensorStorage Peek)" section, you can select a dataset (e.g., `sentence_embeddings_store`, `character_sentiment_store`).
    *   Click "Show First 3 Records from Selected Dataset" to view the metadata and a preview of the tensor data for the first few records in that dataset. This gives insight into how data is structured in the `EmbeddedTensorStorage`.

This demo illustrates how a Tensor Database philosophy, even when simulated, can support more complex and nuanced data analysis compared to traditional data stores by managing interconnected, multi-modal data representations.
