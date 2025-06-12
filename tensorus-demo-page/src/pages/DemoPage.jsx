import React from 'react';
import { useParams, Link } from 'react-router-dom';
import { demos } from '../demoData';
import './DemoPage.css';

const DemoPage = () => {
  const { demoId } = useParams();
  const demo = demos.find(d => d.id === demoId);

  if (!demo) {
    return (
      <div className="demo-page-container">
        <div className="demo-page not-found-page">
          <h2>Demo Not Found</h2>
          <p>The demo you are looking for (ID: {demoId}) does not exist.</p>
          <Link to="/" className="back-link">← Back to Demo Gallery</Link>
        </div>
      </div>
    );
  }

  const streamlitAppUrl = `http://localhost:${demo.localPort || 8501}`;

  return (
    <div className="demo-page-container">
      <div className="demo-page">
        <header className="demo-page-header">
          <h1>{demo.title}</h1>
        </header>

        <section className="demo-section demo-instructions">
          <h3>Running the Demo Locally</h3>
          <p>This demo is an interactive Streamlit application that runs on your local machine. To view and interact with it below, please follow these steps:</p>
          <ol>
            <li>
              Navigate to the root directory of the 'tensorus' project in your terminal
              (this is the directory containing the `requirements.txt` file and the demo Python scripts).
            </li>
            <li>
              (If you haven't done so already for this project) Install the required Python packages by running:
              <pre><code>pip install -r requirements.txt</code></pre>
            </li>
            <li>
              Download NLTK's 'punkt' tokenizer if needed (usually only once per Python environment):
              <pre><code>python -m nltk.downloader punkt</code></pre>
            </li>
            <li>
              Now, run the specific command for this demo:
              <pre><code>{demo.streamlitCommand}</code></pre>
            </li>
            <li>Once the Streamlit app is running (it usually opens in a new browser tab as well), it should appear in the embedded frame below.</li>
            <li>If port {demo.localPort || 8501} is in use, Streamlit may pick the next available one. The embed below targets port {demo.localPort || 8501}. For now, please ensure this demo runs on the target port or stop other Streamlit apps.</li>
          </ol>
        </section>

        <section className="demo-section demo-embed">
          <h3>Live Demo</h3>
          <div className="iframe-container">
            <iframe
              src={streamlitAppUrl}
              title={`${demo.title} - Live Streamlit Demo`}
              width="100%"
              height="700px" // Initial height, can be adjusted with CSS
              allow="cross-origin-isolated" // May or may not be needed
            ></iframe>
          </div>
           <p className="embed-note">Note: The embedded demo above requires the Streamlit application to be running locally on your machine as per the instructions.</p>
        </section>

        <section className="demo-section demo-description">
          <h3>Overview</h3>
          <p>{demo.longDescription}</p>
        </section>

        <section className="demo-section demo-visuals">
          <h3>Conceptual Visuals / Screenshots</h3>
          <img src={demo.visualsPath} alt={`${demo.title} Visual Representation`} />
          <p className="visuals-note">This is a static visual. The interactive demo is above.</p>
        </section>

        <section className="demo-section demo-features">
          <h3>Key Features & Concepts</h3>
          <ul>
            {demo.keyFeatures.map((feature, index) => (
              <li key={index}>{feature}</li>
            ))}
          </ul>
        </section>

        <div className="demo-section demo-tags-section">
          <h3>Tags</h3>
          <div className="demo-tags">
            {demo.tags && demo.tags.map(tag => <span key={tag} className="demo-tag">{tag}</span>)}
          </div>
        </div>

        <section className="demo-section demo-source-code-cta">
          <a href={demo.readmeLink} target="_blank" rel="noopener noreferrer" className="cta-button">
            View Source & Full Setup (README)
          </a>
        </section>

        <Link to="/" className="back-link">← Back to Demo Gallery</Link>
      </div>
    </div>
  );
};

export default DemoPage;
