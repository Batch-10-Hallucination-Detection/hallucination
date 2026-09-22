import React from 'react';

export default function ClaimTable({ claims = [], onViewEvidence }) {
  if (!claims || claims.length === 0) {
    return (
      <div style={{ padding: '20px', textAlign: 'center', color: '#94a3b8', fontSize: '13px' }}>
        No claims extracted yet. Execute analysis to view claim breakdown.
      </div>
    );
  }

  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="analysis-table">
        <thead>
          <tr>
            <th style={{ width: '40px' }}>#</th>
            <th>Extracted Claim</th>
            <th>Retrieval Status</th>
            <th>Self-Consistency</th>
            <th>Uncertainty</th>
            <th>Risk Score</th>
            <th>Verdict</th>
          </tr>
        </thead>
        <tbody>
          {claims.map((claim, idx) => {
            const isContradicted = claim.retrieval_status === "CONTRADICTED" || claim.overall_risk_score >= 0.5;
            const riskPct = Math.round((claim.overall_risk_score || 0.50) * 100);

            return (
              <tr key={claim.claim_id || idx}>
                <td style={{ fontWeight: 700, color: '#64748b' }}>{idx + 1}</td>
                
                <td style={{ maxWidth: '320px', fontWeight: 600, color: '#1e293b', lineHeight: 1.4 }}>
                  {claim.claim}
                </td>

                <td>
                  <span className={`pill-badge ${isContradicted ? 'pill-red' : 'pill-green'}`}>
                    {claim.retrieval_status || 'Contradicted'}
                  </span>
                  <div style={{ fontSize: '10px', color: '#94a3b8', marginTop: '4px', fontWeight: 600 }}>
                    {isContradicted ? 'Evidence conflicts' : 'Verified by evidence'}
                  </div>
                </td>

                <td>
                  <div style={{ fontWeight: 700, fontSize: '12px', color: '#1e293b' }}>
                    {(claim.self_consistency_risk || 0.0).toFixed(1)} / 1
                  </div>
                  <div style={{ fontSize: '10px', color: '#94a3b8', fontWeight: 600 }}>
                    {claim.self_consistency_risk >= 0.5 ? 'Very Low' : 'High'}
                  </div>
                  <div style={{ width: '60px', height: '4px', background: '#e2e8f0', borderRadius: '2px', marginTop: '4px' }}>
                    <div style={{ width: `${(1 - (claim.self_consistency_risk || 0)) * 100}%`, height: '100%', background: '#3b82f6', borderRadius: '2px' }}></div>
                  </div>
                </td>

                <td>
                  <div style={{ fontWeight: 700, fontSize: '12px', color: '#1e293b' }}>
                    {(claim.uncertainty_score || 0.78).toFixed(2)}
                  </div>
                  <div style={{ fontSize: '10px', color: '#f97316', fontWeight: 600 }}>
                    {claim.uncertainty_score >= 0.5 ? 'High' : 'Low'}
                  </div>
                  <div style={{ width: '60px', height: '4px', background: '#e2e8f0', borderRadius: '2px', marginTop: '4px' }}>
                    <div style={{ width: `${(claim.uncertainty_score || 0.78) * 100}%`, height: '100%', background: '#ef4444', borderRadius: '2px' }}></div>
                  </div>
                </td>

                <td>
                  <div style={{ position: 'relative', width: '38px', height: '38px' }}>
                    <svg width="38" height="38" viewBox="0 0 36 36">
                      <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="#f1f5f9" strokeWidth="4" />
                      <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="#ef4444" strokeWidth="4" strokeDasharray={`${riskPct}, 100`} />
                    </svg>
                    <span style={{ position: 'absolute', top: '9px', left: '6px', fontSize: '11px', fontWeight: 800, color: '#ef4444' }}>
                      {riskPct}%
                    </span>
                  </div>
                </td>

                <td>
                  <span style={{ color: '#ef4444', fontWeight: 800, fontSize: '13px' }}>
                    High Risk
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
