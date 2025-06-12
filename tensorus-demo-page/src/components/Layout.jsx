import React from 'react';
import { Link, Outlet } from 'react-router-dom';
import './Layout.css'; // We'll create this next

const Layout = () => {
  return (
    <div className="app-container">
      <header className="app-header">
        <nav className="app-nav">
          <Link to="/" className="nav-logo">Tensorus Demos</Link>
          <div className="nav-links">
            <Link to="/" className="nav-link">Home</Link>
            {/* Add more links here as pages are created */}
          </div>
        </nav>
      </header>
      <main className="main-content">
        <Outlet /> {/* Child routes will render here */}
      </main>
      <footer className="app-footer">
        <p>&copy; {new Date().getFullYear()} Tensorus Project. All rights reserved.</p>
      </footer>
    </div>
  );
};

export default Layout;
