import React from 'react';
import { Link } from 'react-router-dom';
import { demos } from '../demoData'; // Import data from the new file
import './HomePage.css';

const HomePage = () => {
  return (
    <div className="home-page">
      <header className="home-header">
        <h1>Tensorus Conceptual Demos</h1>
        <p>Explore interactive demos showcasing the capabilities of Tensorus, an agentic tensor database.</p>
      </header>
      <section className="demo-gallery">
        {demos.map((demo) => (
          <Link to={`/demos/${demo.id}`} key={demo.id} className="demo-card-link">
            <div className="demo-card">
              <img src={demo.thumbnailUrl} alt={`${demo.title} Thumbnail`} className="demo-thumbnail" />
              <div className="demo-card-content">
                <h2>{demo.title}</h2>
                <p>{demo.shortDescription}</p>
                <div className="demo-tags">
                  {demo.tags && demo.tags.map(tag => <span key={tag} className="demo-tag">{tag}</span>)}
                </div>
                <span className="learn-more-btn">Learn More</span>
              </div>
            </div>
          </Link>
        ))}
      </section>
    </div>
  );
};

export default HomePage;
