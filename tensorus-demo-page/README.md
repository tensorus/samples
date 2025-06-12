# Tensorus Demo Gallery Page

This is the frontend application for showcasing conceptual demos of **Tensorus**, an agentic tensor database. This page is built as a multi-page React application using Vite, providing a gallery of available demos and detailed information for each.

## Features

*   **Demo Gallery:** A home page displaying all available Tensorus demos in a card-based layout.
*   **Detailed Demo Pages:** Individual pages for each demo, providing:
    *   Clear instructions on how to run the specific Streamlit demo locally.
    *   An embedded `iframe` to display the locally running Streamlit application.
    *   A comprehensive description.
    *   Key features and concepts highlighted.
    *   Links to the demo's source code or specific README for full setup details.
*   **Embedded Local Demos:** Interactive Streamlit demos are embedded directly. Users need to run the Streamlit applications locally for them to appear in the gallery.
*   **Clean, Responsive Design:** Styled for a clean, modern aesthetic with basic responsiveness for different screen sizes.
*   **Scalable:** Designed to easily accommodate new demos in the future.

## Getting Started / How to Run This Page

To get this React application running locally:

### Prerequisites

*   **Node.js:** Ensure you have Node.js installed (v16+ recommended). You can download it from [nodejs.org](https://nodejs.org/).
*   **npm or yarn:** A compatible package manager (npm usually comes with Node.js).

### Setup and Running

1.  **Clone the Main Repository:**
    If you haven't already, clone the main Tensorus project repository.
    ```bash
    # Replace <repository_url> with the actual URL of the Tensorus main repository
    git clone <repository_url>
    cd <repository_name> # e.g., cd tensorus
    ```

2.  **Navigate to the Demo Page Directory:**
    From the root of the main Tensorus project:
    ```bash
    cd tensorus-demo-page
    ```

3.  **Install Dependencies:**
    Using npm:
    ```bash
    npm install
    ```
    Or using yarn:
    ```bash
    yarn install
    ```

4.  **Run the Development Server:**
    Using npm:
    ```bash
    npm run dev
    ```
    Or using yarn:
    ```bash
    yarn dev
    ```

5.  **View in Browser:**
    Open your web browser and navigate to the URL provided by Vite (usually `http://localhost:5173/tensorus-demo-page/` or similar, note the base path). You should see the Tensorus Demo Gallery page.

## Adding a New Demo to the Gallery

To add a new demo to this gallery:

1.  **Prepare Demo Information:** Gather the following details for your new demo:
    *   `id`: A unique string identifier (e.g., `my-new-demo`).
    *   `title`: The display title of the demo.
    *   `shortDescription`: A brief (1-2 sentence) description for the gallery card.
    *   `longDescription`: A more detailed description for the demo's individual page.
    *   `thumbnailUrl`: URL for a thumbnail image (e.g., `https://placehold.co/600x360/FF9500/FFFFFF/png?text=New+Demo`).
    *   `visualsPath`: URL for a larger visual for the demo's page (e.g., `https://placehold.co/800x450/FF9500/FFFFFF/png?text=New+Demo+Detail`).
    *   `keyFeatures`: An array of strings listing key features or concepts.
    *   `readmeLink`: A URL to the demo's specific README file or main source file for instructions on how to run it.
    *   `tags`: An array of strings for relevant tags (e.g., `['New Feature', 'AI', 'Data']`).

2.  **Update `src/demoData.js`:**
    Open the `tensorus-demo-page/src/demoData.js` file.
    Add a new JavaScript object containing your demo's information to the `demos` array. Follow the structure of the existing demo objects.

    Example:
    ```javascript
    // In tensorus-demo-page/src/demoData.js
    export const demos = [
      // ... existing demos ...
      {
        id: 'my-new-demo',
        title: 'My Awesome New Demo',
        shortDescription: 'This demo showcases an amazing new capability.',
        longDescription: 'A full explanation of how this new demo works and what it demonstrates about Tensorus.',
        thumbnailUrl: 'https://placehold.co/600x360/FF9500/FFFFFF/png?text=New+Demo',
        visualsPath: 'https://placehold.co/800x450/FF9500/FFFFFF/png?text=New+Demo+Detail',
        keyFeatures: [
          'Feature A of new demo.',
          'Concept B it highlights.',
          'Uses advanced technique C.'
        ],
        readmeLink: 'https://github.com/your-repo/link-to-your-demo-readme.md',
        tags: ['New', 'Awesome', 'AI']
      }
    ];
    ```

3.  **Verify:**
    After saving `demoData.js`, the development server (if running) should automatically reload. Your new demo will appear in the gallery.

## Project Structure Highlights

*   `public/`: Static assets.
*   `src/`: Source code.
    *   `components/`: Reusable React components (e.g., `Layout.jsx`).
    *   `pages/`: Components representing full pages (e.g., `HomePage.jsx`, `DemoPage.jsx`).
    *   `demoData.js`: Centralized data for all demos.
    *   `App.jsx`: Main application component with routing setup.
    *   `main.jsx`: Entry point of the React application.
*   `vite.config.js`: Vite configuration, including the `base` path for deployment.

## Contributing

Contributions to improve this demo gallery are welcome! Please follow standard open-source practices (fork, branch, pull request). For major changes or new features, consider opening an issue first to discuss.
