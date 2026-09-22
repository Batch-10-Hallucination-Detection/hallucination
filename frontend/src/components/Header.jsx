import React from 'react';
import { ShieldCheck, Sun, Bell, LayoutDashboard, Search, BookOpen, BarChart2, FileText } from 'lucide-react';

export default function Header({ activeTab, setActiveTab }) {
  const tabs = [
    { id: 'overview', label: 'Overview', icon: LayoutDashboard },
    { id: 'analyzer', label: 'Response Analyzer', icon: Search },
    { id: 'evidence', label: 'Evidence & Knowledge', icon: BookOpen },
    { id: 'experiments', label: 'Experiment Matrix (E1–E7)', icon: BarChart2 },
    { id: 'reports', label: 'Reports', icon: FileText }
  ];

  return (
    <header className="top-navbar">
      <div className="brand-section">
        <div className="brand-icon">
          <ShieldCheck size={24} />
        </div>
        <div>
          <div className="brand-title">Hallucination Detection System</div>
          <div className="brand-subtitle">Major Project • Batch C10 • VVIT CSE (AI & ML)</div>
        </div>
      </div>

      <nav className="top-tabs">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`top-tab-btn ${isActive ? 'active' : ''}`}
            >
              <Icon size={16} />
              {tab.label}
            </button>
          );
        })}
      </nav>

      <div className="top-controls">
        <button className="icon-btn" title="Toggle Theme">
          <Sun size={18} />
        </button>
        <button className="icon-btn" title="Notifications">
          <Bell size={18} />
        </button>
        <div className="avatar-badge">AI</div>
      </div>
    </header>
  );
}
