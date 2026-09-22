import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import OverviewPage from './pages/OverviewPage';
import AnalyzerPage from './pages/AnalyzerPage';
import EvidencePage from './pages/EvidencePage';
import ExperimentsPage from './pages/ExperimentsPage';
import { fetchDomains } from './api/client';

export default function App() {
  const [activeNav, setActiveNav] = useState('analyzer'); // Default to Response Analyzer
  const [activeDomain, setActiveDomain] = useState('healthcare');
  const [domains, setDomains] = useState([]);

  useEffect(() => {
    fetchDomains().then(data => {
      if (data.domains) {
        setDomains(data.domains);
      }
    });
  }, []);

  const handleChangeDomain = () => {
    setActiveDomain(prev => prev === 'healthcare' ? 'software_development' : 'healthcare');
  };

  return (
    <div className="app-wrapper">
      <div className="main-layout" style={{ minHeight: '100vh' }}>
        <Sidebar
          activeNav={activeNav}
          setActiveNav={setActiveNav}
          activeDomain={activeDomain}
          onChangeDomain={handleChangeDomain}
        />

        <main className="content-body">
          {activeNav === 'overview' && (
            <OverviewPage
              domains={domains}
              onStartAnalysis={() => setActiveNav('analyzer')}
            />
          )}
          {activeNav === 'analyzer' && (
            <AnalyzerPage
              domains={domains}
              onViewEvidence={() => setActiveNav('evidence')}
            />
          )}
          {(activeNav === 'dataset' || activeNav === 'evidence') && (
            <EvidencePage domains={domains} />
          )}
          {(activeNav === 'methods' || activeNav === 'benchmarks' || activeNav === 'settings' || activeNav === 'help') && (
            <ExperimentsPage domains={domains} />
          )}
        </main>
      </div>
    </div>
  );
}
