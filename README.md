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
*   `requirements.txt`: Lists project dependencies (to be created in a subsequent step).
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
    Make sure you have `requirements.txt` in your project directory (it will be created in a later step). Then run:
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
        On the first run of the Story Analyzer demo, NLTK will download necessary data packages (`punkt` for sentence tokenization and `stopwords`). Ensure you have an internet connection.

## Note on TensorStorage

The `EmbeddedTensorStorage` class provided in `tensor_storage_utils.py` is a simplified, in-memory simulation designed for these demonstrations. It is not intended for production use as a persistent, scalable Tensor Database. It serves to illustrate the concepts of storing and retrieving complex tensor data structures within the context of the demo applications. For production scenarios, a dedicated Tensor Database solution like Tensorus would be more appropriate.

---

*This README provides a general guide to the repository. For specific details on each demo, please refer to their respective README files.*
