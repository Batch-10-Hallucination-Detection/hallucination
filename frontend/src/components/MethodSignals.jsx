import React from 'react';

export default function MethodSignals({ scRisk = 0.0, retRisk = 1.0, uncScore = 0.78 }) {
  const scPct = (scRisk * 100).toFixed(1);
  const retPct = (retRisk * 100).toFixed(1);
  const uncPct = (uncScore * 100).toFixed(1);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
      {/* Box 1: Self-Consistency Risk */}
      <div style={{ background: '#f8fafc', border: '1px solid #e5e7eb', borderRadius: '12px', padding: '14px' }}>
        <div style={{ fontSize: '11px', color: '#64748b', fontWeight: 600, marginBottom: '8px' }}>
          Self-Consistency Risk (S)
        </div>
        <div style={{ fontSize: '20px', fontWeight: '800', color: '#3b82f6', marginBottom: '8px' }}>
          {scPct}%
        </div>
        {/* Sparkline Graphic */}
        <svg width="100%" height="24" viewBox="0 0 100 24" style={{ marginBottom: '8px' }}>
          <path d="M0,18 Q25,20 50,12 T100,18" fill="none" stroke="#3b82f6" strokeWidth="2" />
        </svg>
        <div style={{ fontSize: '10px', color: '#94a3b8', fontWeight: 600 }}>
          Intraclass Agreement: <span style={{ color: '#3b82f6' }}>Low</span>
        </div>
      </div>

      {/* Box 2: Retrieval Verification */}
      <div style={{ background: '#f8fafc', border: '1px solid #e5e7eb', borderRadius: '12px', padding: '14px' }}>
        <div style={{ fontSize: '11px', color: '#64748b', fontWeight: 600, marginBottom: '8px' }}>
          Retrieval Verification (R)
        </div>
        <div style={{ fontSize: '20px', fontWeight: '800', color: '#ef4444', marginBottom: '8px' }}>
          {retPct}%
        </div>
        {/* Sparkline Graphic */}
        <svg width="100%" height="24" viewBox="0 0 100 24" style={{ marginBottom: '8px' }}>
          <path d="M0,20 L30,5 L60,18 L100,2" fill="none" stroke="#ef4444" strokeWidth="2" />
        </svg>
        <div style={{ fontSize: '10px', color: '#94a3b8', fontWeight: 600 }}>
          Evidence Status: <span style={{ color: '#ef4444' }}>Contradicted</span>
        </div>
      </div>

      {/* Box 3: Uncertainty Score */}
      <div style={{ background: '#f8fafc', border: '1px solid #e5e7eb', borderRadius: '12px', padding: '14px' }}>
        <div style={{ fontSize: '11px', color: '#64748b', fontWeight: 600, marginBottom: '8px' }}>
          Uncertainty Score (U)
        </div>
        <div style={{ fontSize: '20px', fontWeight: '800', color: '#f97316', marginBottom: '8px' }}>
          {uncPct}%
        </div>
        {/* Sparkline Graphic */}
        <svg width="100%" height="24" viewBox="0 0 100 24" style={{ marginBottom: '8px' }}>
          <path d="M0,15 Q30,2 60,20 T100,10" fill="none" stroke="#f97316" strokeWidth="2" />
        </svg>
        <div style={{ fontSize: '10px', color: '#94a3b8', fontWeight: 600 }}>
          Semantic Entropy: <span style={{ color: '#f97316' }}>High</span>
        </div>
      </div>
    </div>
  );
}
