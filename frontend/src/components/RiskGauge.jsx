import React from 'react';

export default function RiskGauge({ score = 0.50, badge = "High Risk" }) {
  const percentage = Math.round(score * 100);
  
  let color = "#ef4444"; // Red for High Risk
  let badgeClass = "pill-red";
  if (score < 0.35) {
    color = "#10b981"; // Green
    badgeClass = "pill-green";
  } else if (score < 0.65) {
    color = "#ef4444"; // Matches screenshot 50% red donut
    badgeClass = "pill-red";
  }

  const strokeDashoffset = 251 - (251 * percentage) / 100;

  return (
    <div style={{ textAlign: 'center', padding: '10px' }}>
      <div style={{ position: 'relative', width: '150px', height: '150px', margin: '0 auto' }}>
        <svg width="150" height="150" viewBox="0 0 100 100">
          <circle
            cx="50" cy="50" r="40"
            fill="none"
            stroke="#f1f5f9"
            strokeWidth="10"
          />
          <circle
            cx="50" cy="50" r="40"
            fill="none"
            stroke={color}
            strokeWidth="10"
            strokeDasharray="251"
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            style={{ transition: 'stroke-dashoffset 0.8s ease', transform: 'rotate(-90deg)', transformOrigin: '50% 50%' }}
          />
        </svg>
        <div style={{
          position: 'absolute',
          top: 0, left: 0, right: 0, bottom: 0,
          display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center'
        }}>
          <span style={{ fontSize: '32px', fontWeight: '800', color: color, lineHeight: 1 }}>
            {percentage}%
          </span>
          <span style={{ fontSize: '10px', color: '#94a3b8', textTransform: 'capitalize', marginTop: '4px', fontWeight: 600 }}>
            Hallucination Risk
          </span>
        </div>
      </div>
      <div style={{ marginTop: '12px' }}>
        <span className={`pill-badge ${badgeClass}`} style={{ fontSize: '12px', padding: '4px 18px' }}>
          {badge}
        </span>
      </div>
    </div>
  );
}
