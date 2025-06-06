# Commercial-Grade Python Demos with TensorStorage Utility

This repository showcases a collection of commercial-grade Python demonstration applications. Each demo highlights specific data processing and analysis capabilities, leveraging a shared, simulated Tensor Database utility for managing complex data structures. The demos are built with Streamlit for interactive user interfaces and integrate various NLP tools.

## Features

*   **Two Demo Applications**:
    *   **Financial News Impact Analyzer**: Demonstrates how different context retrieval strategies (Vector DB vs. Tensor DB) can impact the inputs for a financial prediction model. Utilizes Hugging Face Transformers for embeddings and sentiment analysis.
    *   **Story Analyzer**: Analyzes character relationships and sentiment evolution in literary texts, showcasing the storage and retrieval of structured tensor data (sentence embeddings, sentiment flows, interaction matrices). Uses Hugging Face Transformers and NLTK.
*   **Simulated Tensor Database**: Features `EmbeddedTensorStorage` (in `tensor_storage_utils.py`), an in-memory Python class simulating functionalities of a Tensor Database like Tensorus. It allows for organized storage and retrieval of tensors and their associated metadata.
*   **NLP Integration**:
    *   Utilizes Hugging Face `transformers` for tasks like text embedding generation and sentiment analysis.
    *   Employs `nltk` (Natural Language Toolkit) for sentence tokenization in the Story Analyzer demo.
*   **Interactive UIs**: Built with Streamlit, providing user-friendly interfaces to interact with the demos and visualize results.
*   **Conceptual Unit Tests**: Includes illustrative unit test structures (`pytest` style) for both demo applications, showcasing how one might approach testing such systems.

## Structure

Key files and their purpose:

*   `financial_news_impact_demo.py`: Streamlit application demonstrating RAG context comparison for financial news.
*   `story_analyzer_demo.py`: Streamlit application for analyzing character sentiment and interactions in stories.
*   `tensor_storage_utils.py`: Contains the `EmbeddedTensorStorage` class, a simulated Tensor Database used by both demos.
*   `test_financial_news_impact_demo.py`: Conceptual unit tests for the financial news demo.
*   `test_story_analyzer_demo.py`: Conceptual unit tests for the story analyzer demo.
*   `requirements.txt`: Lists project dependencies.
*   `README.md`: This file - provides an overview of the repository.
*   `README_financial_news_impact_demo.md`: Specific details for the Financial News Impact demo.
*   `README_story_analyzer_demo.md`: Specific details for the Story Analyzer demo.

## Setup and Running Demos

1.  **Clone the Repository**:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a Virtual Environment (Recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    Make sure you have `requirements.txt` in your project directory. Then run:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Demos**:
    *   **Financial News Impact Demo**:
        ```bash
        streamlit run financial_news_impact_demo.py
        ```
    *   **Story Analyzer Demo**:
        ```bash
        streamlit run story_analyzer_demo.py
        ```

## Troubleshooting / Offline Usage

Both demo applications rely on external resources that are typically downloaded on their first run. This requires an active internet connection.

### Hugging Face Transformers Models

The demos download pre-trained models from the Hugging Face Hub using the `transformers` library. The specific models used are:
*   `sentence-transformers/all-MiniLM-L6-v2` (for text embeddings)
*   `distilbert-base-uncased-finetuned-sst-2-english` (for sentiment analysis)

**Offline Mode:**
If you need to run these demos in an environment without internet access, you'll need to download these models beforehand on a machine with internet access.
1.  **Download Models**: You can download models manually or by running the demos once with an internet connection, which will cache them locally (usually in `~/.cache/huggingface/transformers/`).
2.  **Using Offline Mode**: Refer to the official Hugging Face documentation for detailed instructions on setting up and using offline mode:
    *   [Hugging Face Transformers Offline Mode Documentation](https://huggingface.co/docs/transformers/installation#offline-mode)
3.  This typically involves ensuring the downloaded model files are in the correct cache directory or setting environment variables such as `TRANSFORMERS_OFFLINE=1` and potentially `HF_HOME` to point to your cache.

### NLTK Data Packages

The Story Analyzer demo (`story_analyzer_demo.py`) uses NLTK for sentence tokenization and requires the `punkt` and `stopwords` data packages.
*   **Automatic Download**: The script attempts to download these automatically on the first run if they are not found. This requires an internet connection.
*   **Manual Download**: If automatic download fails, you can download these packages manually:
    1.  Open a Python interpreter in your activated virtual environment:
        ```python
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
        ```
    2.  Alternatively, use the command line:
        ```bash
        python -m nltk.downloader punkt stopwords
        ```
    *   For more information, refer to the [NLTK Data Documentation](https://www.nltk.org/data.html).

The applications are designed to provide error messages if these resources cannot be accessed, guiding you to these offline setup instructions.

## Note on TensorStorage

The `EmbeddedTensorStorage` class provided in `tensor_storage_utils.py` is a simplified, in-memory simulation designed for these demonstrations. It is not intended for production use as a persistent, scalable Tensor Database. It serves to illustrate the concepts of storing and retrieving complex tensor data structures within the context of the demo applications. For production scenarios, a dedicated Tensor Database solution like Tensorus would be more appropriate.

---

*This README provides a general guide to the repository. For specific details on each demo, please refer to their respective README files.*
