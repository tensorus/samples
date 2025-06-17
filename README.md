# Tensorus: Sample Applications & Concept Demos

Welcome to the Tensorus sample applications repository! This collection of demos is designed to illustrate the core concepts and potential of **Tensorus**, an agentic tensor database.

## What is Tensorus?

**Tensorus** is an agentic tensor database designed to power advanced AI applications by managing and enabling operations on rich, multi-dimensional tensor data. It goes beyond traditional databases, and even vector databases, by treating data as interconnected, structured tensors. This approach allows AI agents to perform complex analyses, understand nuanced relationships, and gain deeper insights from diverse data modalities.

The goal of Tensorus is to provide a robust foundation for building AI systems that can:
*   Understand and reason about complex data structures.
*   Fuse information from multiple sources and modalities.
*   Model and predict dynamic interactions and evolutions.
*   Enable AI agents to actively query, manipulate, and learn from data in more sophisticated ways.

## Purpose of This Repository

This repository hosts a series of sample applications and use cases that demonstrate the capabilities and conceptual underpinnings of Tensorus. Each demo is a self-contained Streamlit application that explores a specific problem domain, showcasing how Tensorus's approach to data management and interaction can lead to more powerful AI solutions.

**Important Note on Simulation:** To make these demos easily runnable, understandable, and self-contained, they currently use an *in-memory simulation* of Tensorus's core storage and data handling functionalities (typically through a Python class like `EmbeddedTensorStorage` within each demo script). They are designed for conceptual illustration and **do not require a separate Tensorus database installation.**

## Available Demos

Here's an overview of the sample applications you can explore:

### 1. Financial News Impact Demo

*   **Description:** This demo illustrates how Tensorus can provide richer, multi-modal context for AI models to predict the impact of financial news on a network of assets. It contrasts the Tensorus approach (using multiple, structured tensors per news event) with traditional vector database approaches (using a single embedding per event) for Retrieval Augmented Generation (RAG).
*   **Key Tensorus Concepts Demonstrated:**
    *   Storing multi-modal data (news text, market features, sentiment scores) as distinct, yet related, tensors.
    *   Enabling Retrieval Augmented Generation (RAG) with rich, multi-faceted tensor context rather than single vectors.
    *   Representing textual nuances (e.g., token-level embeddings, simulated attention flows) and quantitative data (e.g., market context, sentiment metrics) in a structured tensor format.
    *   Facilitating more informed inputs for hypothetical downstream predictive models by providing comprehensive event profiles.
*   **How to Run:** For detailed prerequisites, setup, and execution instructions, please see:
    *   ➡️ **[Details and Setup Instructions](README_financial_news_impact_demo.md)**

### 2. Smart Story Analyzer Demo

*   **Description:** This demo showcases how Tensorus can be used to analyze character relationships and sentiment evolution within literary texts. It processes story snippets, stores various analytical representations as tensors, and allows users to explore how character dynamics change throughout the narrative.
*   **Key Tensorus Concepts Demonstrated:**
    *   Transforming textual narratives into multiple structured tensors (e.g., sentence embeddings, character-specific sentiment flows, character interaction matrices).
    *   Analyzing the temporal evolution of relationships and sentiments by operating on sequences of these tensors.
    *   Enabling more nuanced queries about character dynamics beyond simple keyword or similarity searches.
    *   Storing and retrieving complex relational data (like character interactions and sentiments) in a way that AI agents can easily process for insights.
*   **How to Run:** For detailed prerequisites, setup, and execution instructions, please see:
    *   ➡️ **[Details and Setup Instructions](README_story_analyzer_demo.md)**
### 3. MCP Time Series Demo

*   **Description:** Demonstrates the Tensorus MCP server and client with a simple time series dataset. The demo inserts a synthetic sine wave using MCP calls and retrieves it back.
*   **How to Run:** See ➡️ **[README_mcp_time_series_demo.md](README_mcp_time_series_demo.md)**

### 4. MCP Endpoint Demo

*   **Description:** A small Streamlit page that interacts with the official
    Tensorus MCP server. It demonstrates basic dataset and tensor operations
    using the `TensorusMCPClient`.
*   **How to Run:** See ➡️ **[README_MCP_endpoint_demo.md](README_MCP_endpoint_demo.md)**


---

*(More demos may be added over time, showcasing other use cases and Tensorus features.)*

## General Notes on Running Demos

*   **Streamlit Applications:** Each demo in this repository is a self-contained Streamlit application. Streamlit provides an easy way to create interactive web apps for machine learning and data science projects.
*   **Prerequisites:**
    *   Generally, you will need **Python 3.8+** and `pip` (the Python package installer) installed on your system.
    *   **Install Dependencies:** Install all required Python packages by running the following command from the root directory of this project:
        ```bash
        pip install -r requirements.txt
        ```
    *   **NLTK Tokenizer:** Some demos may also require NLTK's 'punkt' tokenizer. If you encounter issues, download it by running:
        ```bash
        python -m nltk.downloader punkt
        ```
*   **Simulation Aspect:** As mentioned, these demos use an embedded, in-memory simulation of Tensorus's storage and data handling capabilities (via a Python class within each script). They are designed for conceptual illustration and do not require a separate, external Tensorus database installation. This makes them easy to download and run directly.
*   **Data Ingestion:** Most demos will include a step (usually a button in the Streamlit UI) to "Load and Ingest Sample Data." This process involves generating tensor representations from raw sample data (like text or structured information) and populating the in-memory simulated TensorStorage. This step might take a few moments, especially on the first run, as NLP models might need to be downloaded and initialized.

We encourage you to explore these demos to get a better understanding of how Tensorus can empower next-generation AI applications. Please refer to the specific README file within each demo's directory for detailed instructions.
