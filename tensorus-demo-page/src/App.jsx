import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import HomePage from './pages/HomePage'; // Import HomePage
import DemoPage from './pages/DemoPage';   // Import DemoPage
import './App.css';

function App() {
  return (
    <Router basename="/tensorus-demo-page/">
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<HomePage />} />
          <Route path="demos/:demoId" element={<DemoPage />} />
          {/* Add other top-level pages here if needed */}
        </Route>
      </Routes>
    </Router>
  );
}

export default App;
