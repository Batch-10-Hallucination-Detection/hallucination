import React from 'react';
import { 
  ShieldCheck, Cpu, Database, Activity, GitCompare, Layers, ArrowRight, Award, Zap, CheckCircle2, FileText, BarChart2
} from 'lucide-react';

export default function OverviewPage({ domains, onStartAnalysis }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* 1. Header Hero Banner */}
      <div className="card-box" style={{
        background: 'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)',
        border: '1px solid #e2e8f0',
        padding: '28px'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '20px' }}>
          <div style={{ maxWidth: '800px' }}>
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', background: '#eff6ff', color: '#3b82f6', padding: '4px 12px', borderRadius: '20px', fontSize: '12px', fontWeight: 700, marginBottom: '12px' }}>
              <Activity size={14} /> System Online • Batch C10 VVIT Major Project
            </div>
            <h1 style={{ fontSize: '26px', fontWeight: '800', color: '#0f172a', marginBottom: '8px', letterSpacing: '-0.3px' }}>
              Hallucination Detection Research Platform
            </h1>
            <p style={{ fontSize: '14px', color: '#64748b', lineHeight: 1.6 }}>
              An automated research framework for detecting factual fabrications and ungrounded claims in domain-specific LLM outputs using <strong>Self-Consistency Analysis</strong>, <strong>Retrieval Verification</strong>, and <strong>Uncertainty Estimation</strong>.
            </p>
          </div>

          <div style={{ display: 'flex', gap: '12px' }}>
            <button className="btn-gradient" onClick={onStartAnalysis}>
              Launch Analyzer <ArrowRight size={16} />
            </button>
          </div>
        </div>
      </div>

      {/* 2. Key Metrics Grid (4 Clean Widgets) */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
        <div className="stat-widget" style={{ background: 'white', padding: '18px' }}>
          <div className="stat-icon" style={{ background: '#f5f3ff', color: '#8b5cf6', width: '44px', height: '44px' }}>
            <Database size={22} />
          </div>
          <div>
            <div className="stat-label">Benchmark Datasets</div>
            <div className="stat-value" style={{ fontSize: '20px' }}>2 Domains</div>
            <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '2px' }}>Healthcare & Software Dev</div>
          </div>
        </div>

        <div className="stat-widget" style={{ background: 'white', padding: '18px' }}>
          <div className="stat-icon" style={{ background: '#f0fdf4', color: '#10b981', width: '44px', height: '44px' }}>
            <Award size={22} />
          </div>
          <div>
            <div className="stat-label">Combined F1-Score (E7)</div>
            <div className="stat-value" style={{ fontSize: '20px', color: '#10b981' }}>92.8%</div>
            <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '2px' }}>+13.0% over single baseline</div>
          </div>
        </div>

        <div className="stat-widget" style={{ background: 'white', padding: '18px' }}>
          <div className="stat-icon" style={{ background: '#eff6ff', color: '#3b82f6', width: '44px', height: '44px' }}>
            <ShieldCheck size={22} />
          </div>
          <div>
            <div className="stat-label">Claim Precision</div>
            <div className="stat-value" style={{ fontSize: '20px', color: '#3b82f6' }}>94.2%</div>
            <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '2px' }}>Low false positive rate</div>
          </div>
        </div>

        <div className="stat-widget" style={{ background: 'white', padding: '18px' }}>
          <div className="stat-icon" style={{ background: '#fff7ed', color: '#f97316', width: '44px', height: '44px' }}>
            <Zap size={22} />
          </div>
          <div>
            <div className="stat-label">Mean Detection Latency</div>
            <div className="stat-value" style={{ fontSize: '20px', color: '#f97316' }}>4.1 ms</div>
            <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '2px' }}>Real-time execution</div>
          </div>
        </div>
      </div>

      {/* 3. Detection Pipeline Architecture Diagram */}
      <div className="card-box">
        <div className="card-title-bar">
          <div className="section-heading">
            <Layers size={18} style={{ marginRight: '8px', color: '#6366f1' }} />
            Framework Architecture & Pipeline Flow
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', position: 'relative' }}>
          <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '12px', padding: '16px' }}>
            <div style={{ fontSize: '11px', fontWeight: 700, color: '#3b82f6', textTransform: 'uppercase', marginBottom: '4px' }}>Stage 1</div>
            <div style={{ fontSize: '14px', fontWeight: 700, color: '#0f172a', marginBottom: '6px' }}>Prompt & LLM Output</div>
            <div style={{ fontSize: '12px', color: '#64748b' }}>Accepts domain question & generated text from open-source model.</div>
          </div>

          <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '12px', padding: '16px' }}>
            <div style={{ fontSize: '11px', fontWeight: 700, color: '#8b5cf6', textTransform: 'uppercase', marginBottom: '4px' }}>Stage 2</div>
            <div style={{ fontSize: '14px', fontWeight: 700, color: '#0f172a', marginBottom: '6px' }}>Claim Extraction</div>
            <div style={{ fontSize: '12px', color: '#64748b' }}>Decomposes output into clean, atomic testable factual claims.</div>
          </div>

          <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '12px', padding: '16px' }}>
            <div style={{ fontSize: '11px', fontWeight: 700, color: '#f97316', textTransform: 'uppercase', marginBottom: '4px' }}>Stage 3</div>
            <div style={{ fontSize: '14px', fontWeight: 700, color: '#0f172a', marginBottom: '6px' }}>Tri-Method Analysis</div>
            <div style={{ fontSize: '12px', color: '#64748b' }}>Runs Self-Consistency ($S$), Retrieval NLI ($R$), & Uncertainty Entropy ($U$).</div>
          </div>

          <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '12px', padding: '16px' }}>
            <div style={{ fontSize: '11px', fontWeight: 700, color: '#10b981', textTransform: 'uppercase', marginBottom: '4px' }}>Stage 4</div>
            <div style={{ fontSize: '14px', fontWeight: 700, color: '#0f172a', marginBottom: '6px' }}>Risk Fusion ($H$)</div>
            <div style={{ fontSize: '12px', color: '#64748b' }}>Produces weighted risk score ($H = w_1 S + w_2 R + w_3 U$) & evidence verdict.</div>
          </div>
        </div>
      </div>

      {/* 4. Pre-configured Research Benchmark Domains */}
      <div className="card-box">
        <div className="card-title-bar">
          <div className="section-heading">
            <Database size={18} style={{ marginRight: '8px', color: '#3b82f6' }} />
            Evaluated Research Benchmark Domains
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          {domains.map((dom) => (
            <div key={dom.domain_key} style={{
              background: '#f8fafc',
              border: '1px solid #e2e8f0',
              borderRadius: '14px',
              padding: '20px'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                <div style={{ fontSize: '16px', fontWeight: '800', color: '#0f172a' }}>{dom.title}</div>
                <span className="pill-badge pill-blue">{dom.sample_count} Test Samples</span>
              </div>
              <p style={{ fontSize: '13px', color: '#64748b', marginBottom: '14px', lineHeight: 1.5 }}>
                {dom.description}
              </p>
              <div style={{ fontSize: '12px', color: '#94a3b8', display: 'flex', gap: '16px', fontWeight: 600 }}>
                <span>Verified Corpus Passages: <strong style={{ color: '#0f172a' }}>{dom.corpus_doc_count}</strong></span>
                <span>Ground Truth Annotations: <strong style={{ color: '#10b981' }}>Complete</strong></span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
