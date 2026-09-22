import React, { useState, useEffect } from 'react';
import { Database, FileText, CheckCircle2, ShieldCheck } from 'lucide-react';

export default function EvidencePage({ domains }) {
  const [selectedDomain, setSelectedDomain] = useState('healthcare');
  const [samples, setSamples] = useState([]);

  useEffect(() => {
    fetch(`http://localhost:8000/api/dataset/samples?domain=${selectedDomain}`)
      .then(res => res.json())
      .then(data => setSamples(data.samples || []))
      .catch(err => console.error(err));
  }, [selectedDomain]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
      <div className="card-glass">
        <div className="card-header">
          <div className="card-title">
            <Database size={20} /> Domain Knowledge Store & Evidence Corpus
          </div>
          <div style={{ width: '220px' }}>
            <select
              className="form-select"
              value={selectedDomain}
              onChange={(e) => setSelectedDomain(e.target.value)}
            >
              {domains.map(d => (
                <option key={d.domain_key} value={d.domain_key}>{d.title}</option>
              ))}
            </select>
          </div>
        </div>
        <p style={{ fontSize: '14px', color: '#94a3b8', marginBottom: '20px' }}>
          Trusted evidence documents used by the <strong>Retrieval Verification Engine</strong> to perform semantic search, vector matching, and Natural Language Inference (NLI) claim verification.
        </p>

        {/* Benchmark samples & evidence */}
        <div className="grid-2">
          {samples.map((sample) => (
            <div key={sample.sample_id} style={{
              background: 'rgba(15, 23, 42, 0.7)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '12px',
              padding: '20px'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
                <span className="claim-id">[{sample.sample_id}]</span>
                <span className={`badge ${sample.ground_truth.is_hallucinated ? 'badge-danger' : 'badge-success'}`}>
                  {sample.ground_truth.is_hallucinated ? 'Ground Truth: Hallucinated' : 'Ground Truth: Supported'}
                </span>
              </div>
              
              <div style={{ fontSize: '14px', fontWeight: '700', color: '#f8fafc', marginBottom: '8px' }}>
                Prompt: "{sample.prompt}"
              </div>
              
              <div style={{ fontSize: '13px', color: '#cbd5e1', background: 'rgba(0,0,0,0.3)', padding: '10px', borderRadius: '8px', marginBottom: '12px' }}>
                <strong>LLM Response:</strong> "{sample.response}"
              </div>

              {sample.ground_truth.hallucinated_claims.length > 0 && (
                <div style={{ fontSize: '12px', color: '#f43f5e', marginBottom: '8px' }}>
                  <strong>Annotated False Claims:</strong> {sample.ground_truth.hallucinated_claims.join(", ")}
                </div>
              )}

              <div style={{ fontSize: '12px', color: '#06b6d4' }}>
                <strong>Category:</strong> {sample.ground_truth.category}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
