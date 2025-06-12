import React from 'react';
import { useParams, Link } from 'react-router-dom';
import { demos } from '../demoData'; // Import data from the new file
import './DemoPage.css';

const DemoPage = () => {
  const { demoId } = useParams();
  // Find the demo data by id
  const demo = demos.find(d => d.id === demoId);

  if (!demo) {
    return (
      <div className="demo-page-container"> {/* Added a container for consistent centering */}
        <div className="demo-page not-found-page">
          <h2>Demo Not Found</h2>
          <p>The demo you are looking for (ID: {demoId}) does not exist.</p>
          <Link to="/" className="back-link">← Back to Demo Gallery</Link>
        </div>
      </div>
    );
  }

  return (
    <div className="demo-page-container">
      <div className="demo-page">
        <header className="demo-page-header">
          <h1>{demo.title}</h1>
        </header>

        <section className="demo-section demo-description">
          <h3>Overview</h3>
          <p>{demo.longDescription}</p>
        </section>

        <section className="demo-section demo-visuals">
          <h3>Visual Showcase</h3>
          <img src={demo.visualsPath} alt={`${demo.title} Visual Representation`} />
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

        <section className="demo-section demo-cta">
          <a href={demo.readmeLink} target="_blank" rel="noopener noreferrer" className="cta-button">
            View on GitHub & Run Demo (README)
          </a>
        </section>

        <Link to="/" className="back-link">← Back to Demo Gallery</Link>
      </div>
    </div>
  );
};

export default DemoPage;
