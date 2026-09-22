import React from 'react';

export default function CombinedRiskWeights({ w1 = 0.30, w2 = 0.50, w3 = 0.20, riskScore = 0.50 }) {
  const p1 = Math.round(w1 * 100);
  const p2 = Math.round(w2 * 100);
  const p3 = Math.round(w3 * 100);
  const percentage = Math.round(riskScore * 100);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', justifyContent: 'space-between' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '20px', padding: '8px 0' }}>
        {/* Multi-segmented Donut representation */}
        <div style={{ position: 'relative', width: '100px', height: '100px', flexShrink: 0 }}>
          <svg width="100" height="100" viewBox="0 0 100 100">
            {/* Segment 1: w1 (Blue) */}
            <circle cx="50" cy="50" r="38" fill="none" stroke="#3b82f6" strokeWidth="12" strokeDasharray="238" strokeDashoffset="166" style={{ transform: 'rotate(-90deg)', transformOrigin: '50% 50%' }} />
            {/* Segment 2: w2 (Red) */}
            <circle cx="50" cy="50" r="38" fill="none" stroke="#ef4444" strokeWidth="12" strokeDasharray="238" strokeDashoffset="119" style={{ transform: 'rotate(18deg)', transformOrigin: '50% 50%' }} />
            {/* Segment 3: w3 (Orange) */}
            <circle cx="50" cy="50" r="38" fill="none" stroke="#f97316" strokeWidth="12" strokeDasharray="238" strokeDashoffset="190" style={{ transform: 'rotate(198deg)', transformOrigin: '50% 50%' }} />
          </svg>
          <div style={{
            position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '20px', fontWeight: '800', color: '#1e293b'
          }}>
            {percentage}%
          </div>
        </div>

        {/* Legend */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ width: '10px', height: '10px', borderRadius: '3px', background: '#3b82f6' }}></span>
            <span style={{ color: '#64748b', fontWeight: 600 }}>w1 (S): <strong>{p1}%</strong></span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ width: '10px', height: '10px', borderRadius: '3px', background: '#ef4444' }}></span>
            <span style={{ color: '#64748b', fontWeight: 600 }}>w2 (R): <strong>{p2}%</strong></span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ width: '10px', height: '10px', borderRadius: '3px', background: '#f97316' }}></span>
            <span style={{ color: '#64748b', fontWeight: 600 }}>w3 (U): <strong>{p3}%</strong></span>
          </div>
        </div>
      </div>

      <div style={{ fontSize: '11px', color: '#94a3b8', background: '#f8fafc', padding: '6px 12px', borderRadius: '6px', textAlign: 'center', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
        Formula: w1*S + w2*R + w3*U
      </div>
    </div>
  );
}
