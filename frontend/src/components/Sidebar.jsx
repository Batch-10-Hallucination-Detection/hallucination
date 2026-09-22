import React from 'react';
import { 
  ShieldCheck, Search, Database, BookOpen, Cpu, BarChart2, Settings, HelpCircle, Heart, Code, LayoutDashboard
} from 'lucide-react';

export default function Sidebar({ activeNav, setActiveNav, activeDomain, onChangeDomain }) {
  const menuItems = [
    { id: 'overview', label: 'Overview', icon: LayoutDashboard },
    { id: 'analyzer', label: 'Analyzer', icon: Search },
    { id: 'dataset', label: 'Dataset', icon: Database },
    { id: 'evidence', label: 'Evidence Base', icon: BookOpen },
    { id: 'methods', label: 'Methods', icon: Cpu },
    { id: 'benchmarks', label: 'Benchmarks', icon: BarChart2 },
    { id: 'settings', label: 'Settings', icon: Settings },
    { id: 'help', label: 'Help & Docs', icon: HelpCircle },
  ];

  return (
    <aside className="left-sidebar">
      <div>
        {/* Brand Logo Section at Top of Sidebar */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', paddingBottom: '20px', marginBottom: '16px', borderBottom: '1px solid var(--border-color)' }}>
          <div className="brand-icon" style={{ width: '38px', height: '38px', flexShrink: 0 }}>
            <ShieldCheck size={22} />
          </div>
          <div>
            <div style={{ fontSize: '15px', fontWeight: 800, color: '#0f172a', lineHeight: 1.2 }}>
              Hallucination Detection
            </div>
            <div style={{ fontSize: '11px', color: '#64748b', fontWeight: 600 }}>
              Batch C10 • VVIT CSE
            </div>
          </div>
        </div>

        {/* Sidebar Navigation */}
        <div className="sidebar-nav">
          {menuItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeNav === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveNav(item.id)}
                className={`sidebar-item ${isActive ? 'active' : ''}`}
              >
                <Icon size={18} />
                {item.label}
              </button>
            );
          })}
        </div>
      </div>

      <div className="sidebar-footer-card">
        <div className="sidebar-footer-title">Active Domain</div>
        <div className="sidebar-footer-value">
          {activeDomain === 'healthcare' ? <Heart size={16} color="#ef4444" /> : <Code size={16} color="#3b82f6" />}
          {activeDomain === 'healthcare' ? 'Healthcare & Clinical' : 'Software Development'}
        </div>
        <button 
          className="btn-outline" 
          style={{ width: '100%', fontSize: '12px', padding: '6px 12px' }}
          onClick={onChangeDomain}
        >
          Change Domain
        </button>
      </div>
    </aside>
  );
}
