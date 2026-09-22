import React from 'react';
import { AlertTriangle, CheckCircle2, HelpCircle, FileText } from 'lucide-react';

export default function ClaimCard({ claim }) {
  const isFlagged = claim.classification === "POTENTIAL_HALLUCINATION" || claim.retrieval_status === "CONTRADICTED";
  const isSupported = claim.classification === "SUPPORTED" && claim.retrieval_status === "SUPPORTED";
  
  let badgeClass = "badge-success";
  let statusIcon = <CheckCircle2 size={16} className="text-emerald" />;
  if (isFlagged) {
    badgeClass = "badge-danger";
    statusIcon = <AlertTriangle size={16} className="text-rose" />;
  } else if (claim.retrieval_status === "UNVERIFIED") {
    badgeClass = "badge-warning";
    statusIcon = <HelpCircle size={16} className="text-amber" />;
  }

  return (
    <div className={`claim-card ${isFlagged ? 'flagged' : isSupported ? 'supported' : 'unverified'}`}>
      <div className="claim-header">
        <span className="claim-id">{claim.claim_id} • {claim.claim_type || 'general_factual'}</span>
        <span className={`badge ${badgeClass}`}>
          {statusIcon}
          {claim.retrieval_status}
        </span>
      </div>

      <div className="claim-text">"{claim.claim}"</div>

      <div className="claim-signals">
        <div className="signal-item">
          <span>Retrieval Status:</span>
          <span className="signal-val">{claim.retrieval_status}</span>
        </div>
        <div className="signal-item">
          <span>Inconsistency Risk (S):</span>
          <span className="signal-val">{(claim.self_consistency_risk * 100).toFixed(0)}%</span>
        </div>
        <div className="signal-item">
          <span>Uncertainty (U):</span>
          <span className="signal-val">{(claim.uncertainty_score * 100).toFixed(0)}%</span>
        </div>
        <div className="signal-item">
          <span>Combined Risk:</span>
          <span className="signal-val" style={{ color: claim.overall_risk_score >= 0.5 ? '#f43f5e' : '#10b981' }}>
            {(claim.overall_risk_score * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      {claim.explanation && (
        <div style={{ marginTop: '10px', fontSize: '13px', color: '#cbd5e1', fontStyle: 'italic', display: 'flex', gap: '6px', alignItems: 'flex-start' }}>
          <FileText size={14} style={{ marginTop: '2px', flexShrink: 0, color: '#6366f1' }} />
          <span>{claim.explanation}</span>
        </div>
      )}
    </div>
  );
}
