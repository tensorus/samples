# Tensorus Flask Demo Page

## Overview

This project is a Flask-based web application designed to showcase the features and capabilities of the Tensorus platform. It provides an interactive and user-friendly interface to demonstrate conceptual examples of Tensorus's core functionalities, its model zoo, dataset integration, and agent-driven operations. All backend data and operations in this demo are mocked and do not require a live Tensorus instance.

## Features Demonstrated

The demo application is organized into several key sections:

*   **Home Page (`/`):** A welcoming landing page that provides a brief overview of Tensorus and the demo itself.
*   **Core Features (`/core-features`):**
    *   **Tensor Storage & Retrieval:** Demonstrates creating, viewing, and listing mock tensors with metadata.
    *   **Schema Enforcement:** Conceptually shows how dataset schemas can be applied during tensor ingestion.
    *   **Natural Query Language (NQL):** A simplified interface to query mock tensors using NQL-like commands.
    *   **Tensor Operations:** Simulates performing operations (e.g., add, multiply by scalar, transpose) on mock tensors.
*   **Model Showcase (`/models`):**
    *   Presents a curated list of example machine learning models available in the Tensorus ecosystem.
    *   Displays model descriptions, categories, example inputs/outputs, and (placeholder) documentation links.
    *   Includes a "Mock Predict" feature for an example model.
*   **Dataset Showcase (`/datasets`):**
    *   Highlights various types of datasets (e.g., Image, Time Series, Tabular) that Tensorus can manage.
    *   Shows dataset descriptions, mock properties, and example data structures.
*   **Agent Dashboard (`/agents`):**
    *   Simulates a dashboard for monitoring and interacting with Tensorus agents (e.g., Data Ingestion, Reinforcement Learning, AutoML).
    *   Displays agent status, configuration, and mock logs with periodic updates.
    *   Allows mock "start" and "stop" actions for agents.

## Technology Stack

*   **Backend:** Python, Flask
*   **Frontend:** HTML, CSS, JavaScript
*   **UI Framework:** Bootstrap 5, Bootstrap Icons
*   **Fonts:** Google Fonts (Inter)
*   **WSGI Server (for deployment):** Gunicorn

## Project Structure

The main components of the project are organized as follows:

*   `app.py`: The main Flask application file containing route definitions, API logic, and mock data.
*   `requirements.txt`: Lists the Python dependencies for the project (Flask, Gunicorn).
*   `Procfile`: Declares the command for starting the application on PaaS platforms (e.g., Heroku).
*   `templates/`: Contains HTML templates used by Flask for rendering pages.
    *   `base.html`: Base layout template including navbar, footer, and common CSS/JS links.
    *   `index.html`: Home page.
    *   `core_features.html`: Page for core Tensorus feature demos.
    *   `models.html`: Page for the model showcase.
    *   `datasets.html`: Page for the dataset showcase.
    *   `agents.html`: Page for the agent dashboard.
*   `static/`: Contains static assets like CSS and JavaScript files.
    *   `css/style.css`: Custom stylesheets for global and component-specific styling.
    *   `js/`: Contains JavaScript files for frontend interactivity on different pages (`core_features.js`, `models.js`, `datasets.js`, `agents.js`).

## Setup and Running

To run this demo application locally, follow these steps:

1.  **Clone the Repository (if applicable):**
    ```bash
    # git clone <repository-url>
    # cd tensorus-flask-demo
    ```

2.  **Create and Activate a Python Virtual Environment:**
    It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate    # On Windows
    ```

3.  **Install Dependencies:**
    Install the required Python packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application:**

    *   **For Development/Local Testing (using Flask's built-in server):**
        ```bash
        python app.py
        ```
        The application will typically be available at `http://127.0.0.1:5000`.
        Note: The `debug=True` mode is currently removed from `app.run()` in `app.py` for deployment preparedness, but you can temporarily re-enable it for development if needed.

    *   **For Production-like Local Testing (using Gunicorn):**
        Gunicorn is specified in the `Procfile` and `requirements.txt` for production deployments.
        ```bash
        gunicorn app:app -b 0.0.0.0:5000
        ```
        This will also run the application on `http://127.0.0.1:5000` (or `http://localhost:5000`). You can change the port as needed (e.g., `-b 0.0.0.0:8000`).

## Pages Overview

*   **Home (`/`):** Provides an introduction to the Tensorus demo.
*   **Core Features (`/core-features`):** Explore demonstrations of tensor manipulation, NQL, schema concepts, and basic tensor operations.
*   **Model Showcase (`/models`):** View a catalog of example models, their descriptions, and mock predictive behavior.
*   **Dataset Showcase (`/datasets`):** Discover different types of datasets Tensorus can handle, along with their mock properties.
*   **Agent Dashboard (`/agents`):** Monitor and simulate interactions with various Tensorus agents.
