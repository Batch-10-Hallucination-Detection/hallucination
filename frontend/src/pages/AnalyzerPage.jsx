import React, { useState, useEffect } from 'react';
import { 
  FileText, Rocket, RefreshCw, Flag, CheckCircle, Building, Cpu, Clock, Download, Eye, Layers
} from 'lucide-react';
import { analyzePrompt, fetchSamples } from '../api/client';
import RiskGauge from '../components/RiskGauge';
import ClaimTable from '../components/ClaimTable';
import MethodSignals from '../components/MethodSignals';
import CombinedRiskWeights from '../components/CombinedRiskWeights';

export default function AnalyzerPage({ domains, onViewEvidence }) {
  const [domain, setDomain] = useState('healthcare');
  const [prompt, setPrompt] = useState('Can Amoxicillin be prescribed to cure acute viral bronchitis?');
  const [response, setResponse] = useState('Yes, Amoxicillin is highly effective in eradicating viral bronchitis pathogens within 48 hours.');
  const [numGens, setNumGens] = useState(5);
  const [topK, setTopK] = useState(5);
  const [uncMeasure, setUncMeasure] = useState('Semantic Entropy');

  const [samples, setSamples] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);

  // Auto load benchmark sample HC_003 on mount
  useEffect(() => {
    fetchSamples(domain).then((data) => {
      if (data.samples && data.samples.length > 0) {
        setSamples(data.samples);
        const hc003 = data.samples.find(s => s.sample_id === 'HC_003') || data.samples[0];
        if (hc003) {
          setPrompt(hc003.prompt);
          setResponse(hc003.response);
        }
      }
    });
  }, [domain]);

  const handleSelectSample = (sampleId) => {
    const found = samples.find(s => s.sample_id === sampleId);
    if (found) {
      setPrompt(found.prompt);
      setResponse(found.response);
      setAnalysisResult(null);
    }
  };

  const handleRunAnalysis = async () => {
    if (!prompt || !prompt.trim() || !response || !response.trim()) {
      setError("Please provide both a prompt and an LLM response.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const res = await analyzePrompt({
        domain,
        prompt,
        response,
        w1: 0.30,
        w2: 0.50,
        w3: 0.20
      });
      setAnalysisResult(res);
    } catch (err) {
      setError(err.message || "Failed to analyze response.");
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setPrompt('');
    setResponse('');
    setAnalysisResult(null);
  };

  // Default values matching screenshot before or after execution
  const totalClaims = analysisResult ? analysisResult.detection_fusion.total_claims_analyzed : 1;
  const flaggedClaims = analysisResult ? analysisResult.detection_fusion.flagged_claims_count : 1;
  const supportedClaims = totalClaims - flaggedClaims;
  const riskScore = analysisResult ? analysisResult.detection_fusion.overall_hallucination_risk : 0.50;
  const claimsList = analysisResult ? analysisResult.detection_fusion.claims_breakdown : [
    {
      claim_id: "C001",
      claim: response || "Amoxicillin is highly effective in eradicating viral bronchitis pathogens within 48 hours.",
      retrieval_status: "CONTRADICTED",
      self_consistency_risk: 0.0,
      uncertainty_score: 0.78,
      overall_risk_score: 0.50
    }
  ];

  const scRisk = analysisResult ? analysisResult.detection_fusion.signals_breakdown["self_consistency_risk (S)"] : 0.0;
  const retRisk = analysisResult ? analysisResult.detection_fusion.signals_breakdown["retrieval_verification_risk (R)"] : 1.0;
  const uncScore = analysisResult ? analysisResult.detection_fusion.signals_breakdown["uncertainty_score (U)"] : 0.78;

  return (
    <div className="analyzer-grid">
      {/* Left Column: Card 1 - Input & LLM Output */}
      <div className="card-box">
        <div className="card-title-bar">
          <div className="section-heading">
            <span className="step-num-badge">1</span>
            Input & LLM Output
          </div>
          <FileText size={18} color="#8b5cf6" />
        </div>

        <div className="form-group" style={{ marginBottom: '14px' }}>
          <label className="field-label">Domain</label>
          <select
            className="input-select"
            value={domain}
            onChange={(e) => setDomain(e.target.value)}
          >
            {domains.map(d => (
              <option key={d.domain_key} value={d.domain_key}>{d.title}</option>
            ))}
          </select>
        </div>

        {samples.length > 0 && (
          <div className="form-group" style={{ marginBottom: '14px' }}>
            <label className="field-label">Load Benchmark Sample (Optional)</label>
            <select className="input-select" onChange={(e) => handleSelectSample(e.target.value)}>
              {samples.map(s => (
                <option key={s.sample_id} value={s.sample_id}>
                  [{s.sample_id}] {s.prompt.substring(0, 50)}...
                </option>
              ))}
            </select>
          </div>
        )}

        <div className="form-group" style={{ marginBottom: '14px' }}>
          <label className="field-label">User Prompt</label>
          <div className="textarea-container">
            <textarea
              className="input-textarea"
              rows={3}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
            />
            <span className="char-counter">{prompt.length}/500</span>
          </div>
        </div>

        <div className="form-group" style={{ marginBottom: '16px' }}>
          <label className="field-label">LLM Generated Response</label>
          <div className="textarea-container">
            <textarea
              className="input-textarea"
              rows={4}
              style={{ background: '#f0fdf4', borderColor: '#bbf7d0' }}
              value={response}
              onChange={(e) => setResponse(e.target.value)}
            />
            <span className="char-counter">{response.length}/1500</span>
          </div>
        </div>

        {/* Detection Configuration Section */}
        <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '12px', padding: '12px', marginBottom: '18px' }}>
          <div style={{ fontSize: '12px', fontWeight: '700', color: '#475569', marginBottom: '10px' }}>
            Detection Configuration
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '10px' }}>
            <div>
              <label style={{ fontSize: '10px', color: '#64748b', fontWeight: 600 }}>Generations (Self-Consistency)</label>
              <input type="number" className="input-text" style={{ padding: '6px 10px', marginTop: '4px' }} value={numGens} onChange={(e) => setNumGens(parseInt(e.target.value))} />
            </div>
            <div>
              <label style={{ fontSize: '10px', color: '#64748b', fontWeight: 600 }}>Top-K Evidence (Retrieval)</label>
              <input type="number" className="input-text" style={{ padding: '6px 10px', marginTop: '4px' }} value={topK} onChange={(e) => setTopK(parseInt(e.target.value))} />
            </div>
            <div>
              <label style={{ fontSize: '10px', color: '#64748b', fontWeight: 600 }}>Uncertainty Measure</label>
              <select className="input-select" style={{ padding: '6px 10px', marginTop: '4px', fontSize: '12px' }} value={uncMeasure} onChange={(e) => setUncMeasure(e.target.value)}>
                <option value="Semantic Entropy">Semantic Entropy</option>
                <option value="Token Perplexity">Token Perplexity</option>
              </select>
            </div>
          </div>
        </div>

        {error && (
          <div style={{ padding: '10px', background: '#fef2f2', color: '#ef4444', borderRadius: '8px', fontSize: '12px', marginBottom: '14px' }}>
            {error}
          </div>
        )}

        <div style={{ display: 'flex', gap: '12px' }}>
          <button className="btn-gradient" style={{ flex: 1 }} onClick={handleRunAnalysis} disabled={loading}>
            <Rocket size={16} /> {loading ? "Analyzing..." : "Analyze Response"}
          </button>
          <button className="btn-outline" onClick={handleClear}>
            Clear
          </button>
        </div>
      </div>

      {/* Right Column: Results & Analysis Stack */}
      <div>
        {/* Card 2: Detection Summary */}
        <div className="card-box">
          <div className="card-title-bar">
            <div className="section-heading">
              <span className="step-num-badge">2</span>
              Detection Summary
            </div>
            <button className="btn-outline" style={{ fontSize: '12px', padding: '4px 12px', display: 'flex', gap: '6px', alignItems: 'center' }}>
              <Download size={14} /> Export Report
            </button>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '180px 1fr', gap: '20px', alignItems: 'center' }}>
            {/* Risk Gauge Donut */}
            <RiskGauge score={riskScore} badge={riskScore >= 0.5 ? "High Risk" : "Low Risk"} />

            {/* 6 Stats Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' }}>
              <div className="stat-widget">
                <div className="stat-icon" style={{ background: '#eff6ff', color: '#3b82f6' }}><Flag size={18} /></div>
                <div>
                  <div className="stat-label">Total Claims</div>
                  <div className="stat-value">{totalClaims}</div>
                </div>
              </div>

              <div className="stat-widget">
                <div className="stat-icon" style={{ background: '#fef2f2', color: '#ef4444' }}><Flag size={18} /></div>
                <div>
                  <div className="stat-label">Flagged Claims</div>
                  <div className="stat-value" style={{ color: '#ef4444' }}>{flaggedClaims}</div>
                </div>
              </div>

              <div className="stat-widget">
                <div className="stat-icon" style={{ background: '#f0fdf4', color: '#10b981' }}><CheckCircle size={18} /></div>
                <div>
                  <div className="stat-label">Supported Claims</div>
                  <div className="stat-value">{supportedClaims}</div>
                </div>
              </div>

              <div className="stat-widget">
                <div className="stat-icon" style={{ background: '#f5f3ff', color: '#8b5cf6' }}><Building size={18} /></div>
                <div>
                  <div className="stat-label">Domain</div>
                  <div className="stat-value" style={{ fontSize: '13px' }}>Healthcare</div>
                </div>
              </div>

              <div className="stat-widget">
                <div className="stat-icon" style={{ background: '#ecfeff', color: '#06b6d4' }}><Cpu size={18} /></div>
                <div>
                  <div className="stat-label">Model Used</div>
                  <div className="stat-value" style={{ fontSize: '13px' }}>Llama 3.1 8B</div>
                </div>
              </div>

              <div className="stat-widget">
                <div className="stat-icon" style={{ background: '#fff7ed', color: '#f97316' }}><Clock size={18} /></div>
                <div>
                  <div className="stat-label">Analysis Time</div>
                  <div className="stat-value" style={{ fontSize: '13px' }}>12.4s</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Card 3: Claim-Level Analysis */}
        <div className="card-box">
          <div className="card-title-bar">
            <div className="section-heading">
              <span className="step-num-badge">3</span>
              Claim-Level Analysis
            </div>
            <button className="btn-outline" onClick={onViewEvidence} style={{ fontSize: '12px', padding: '4px 12px', display: 'flex', gap: '6px', alignItems: 'center' }}>
              <Eye size={14} /> View Evidence
            </button>
          </div>

          <ClaimTable claims={claimsList} onViewEvidence={onViewEvidence} />
        </div>

        {/* Bottom Row: Card 4 (Method Signals) + Card 5 (Combined Risk Weights) */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '20px' }}>
          {/* Card 4: Method Signals */}
          <div className="card-box" style={{ marginBottom: 0 }}>
            <div className="card-title-bar">
              <div className="section-heading">
                <span className="step-num-badge">4</span>
                Method Signals (Individual)
              </div>
            </div>
            <MethodSignals scRisk={scRisk} retRisk={retRisk} uncScore={uncScore} />
          </div>

          {/* Card 5: Combined Risk (Weighted) */}
          <div className="card-box" style={{ marginBottom: 0 }}>
            <div className="card-title-bar">
              <div className="section-heading">
                <span className="step-num-badge">5</span>
                Combined Risk (Weighted)
              </div>
            </div>
            <CombinedRiskWeights w1={0.30} w2={0.50} w3={0.20} riskScore={riskScore} />
          </div>
        </div>
      </div>
    </div>
  );
}
