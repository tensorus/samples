# Financial News Impact RAG Demo

This demo application showcases how different context retrieval strategies for Retrieval Augmented Generation (RAG) can impact the inputs for a hypothetical AI model predicting the financial impact of news events. It contrasts a traditional Vector Database (VB) approach with a Tensor Database (TD) approach, simulated by `EmbeddedTensorStorage`.

**Note:** This demo uses an *in-memory simulation* (`EmbeddedTensorStorage`) to illustrate concepts and does not connect to a full, external Tensor Database instance.

## Purpose

The core idea is to demonstrate that providing an AI model with richer, multi-faceted, and structured tensor data (simulating a Tensor Database like Tensorus) can lead to more nuanced and potentially more accurate predictions compared to relying on single document-level embeddings typical of a Vector DB.

## Key Features

*   **Sample Financial News Events**: Uses a predefined set of hypothetical news events with associated market data.
*   **Simulated Data Ingestion**:
    *   Generates single vector embeddings for a simulated Vector DB.
    *   Generates multiple, diverse tensors for a simulated Tensor DB:
        *   Token-level embeddings (2D tensor).
        *   Simulated attention flow tensor (2D tensor).
        *   Structured sentiment tensor (1D tensor).
        *   Market context tensor (1D tensor with price/volatility for relevant assets).
*   **Interactive Context Retrieval**: Allows users to select a news event and retrieve context using both simulated approaches (VB vs. TD).
*   **Context Comparison**: Visually presents the retrieved data, highlighting the richer, multi-modal context provided by the Tensor Database simulation.
*   **NLP Integration**: Uses Hugging Face `transformers` for text embeddings and sentiment analysis.
*   **Streamlit UI**: Provides an interactive web interface to run the demonstration.

## Setup and Running the Demo

1.  **Prerequisites**:
    *   Ensure you have Python 3.8+ installed.
    *   It is highly recommended to create and activate a Python virtual environment.

2.  **Clone the Repository and Install Dependencies**:
    *   Clone this repository to your local machine.
    *   Navigate to the repository's root directory.
    *   Install the required packages by running:
        ```bash
        pip install -r requirements.txt
        ```
    *   (The `requirements.txt` file should be present in the root of the repository; it will be created in a subsequent step if it's not already there).

3.  **Run the Demo**:
    *   Open your terminal and navigate to the repository's root directory.
    *   Execute the following command:
        ```bash
        streamlit run financial_news_impact_demo.py
        ```
    *   The application should open in your web browser.

## How to Interact with the Demo

1.  **Load and Ingest Data**:
    *   In the sidebar, click the "Load & Ingest Sample Financial Events into Simulated DBs" button.
    *   Wait for the data processing and ingestion to complete (a spinner will be shown). This might take a moment on the first run as NLP models are loaded. A success message will appear in the sidebar.

2.  **Select a News Event**:
    *   Once data is ingested, use the selectbox in the main application area to choose a news event you wish to analyze.
    *   The original news text for the selected event will be displayed.

3.  **Retrieve and Compare Context**:
    *   **Vector DB Approach**: Click the "Retrieve VB Context (Simulated)" button. Observe the "Context Provided to Predictive Model (Simulated VB)" section. This typically includes a text snippet and information about a single overall news embedding.
    *   **Tensor DB Approach**: Click the "Retrieve TD Context (Simulated)" button. Observe the "Context Provided to Predictive Model (Simulated Tensorus)" section. This will show multiple structured tensors (token embeddings, attention, sentiment, market context).

4.  **Understand the Difference**:
    *   Compare the information retrieved by both methods.
    *   Read the explanations in the UI that describe why the richer, multi-modal context from the Tensor Database simulation could lead to better predictions for a sophisticated AI model.

This demo focuses on the *input* to a hypothetical predictive model. It illustrates that the quality, depth, and structure of this input context, as managed by a system like Tensorus (simulated here), can significantly enhance the potential for advanced AI-driven financial analysis.
