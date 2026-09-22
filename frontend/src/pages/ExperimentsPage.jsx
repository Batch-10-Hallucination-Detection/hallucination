import React, { useState, useEffect } from 'react';
import { BarChart2, Play, Download, Award, Clock, CheckCircle } from 'lucide-react';
import { runEvaluationMatrix } from '../api/client';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

export default function ExperimentsPage({ domains }) {
  const [selectedDomain, setSelectedDomain] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);

  const handleRunMatrix = async () => {
    setLoading(true);
    try {
      const res = await runEvaluationMatrix(selectedDomain || null);
      setResults(res);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Auto run benchmark matrix on component load
    handleRunMatrix();
  }, [selectedDomain]);

  const chartData = results ? {
    labels: Object.keys(results.experiment_results),
    datasets: [
      {
        label: 'F1 Score',
        data: Object.values(results.experiment_results).map(r => r.f1_score * 100),
        backgroundColor: 'rgba(99, 102, 241, 0.85)',
      },
      {
        label: 'Accuracy',
        data: Object.values(results.experiment_results).map(r => r.accuracy * 100),
        backgroundColor: 'rgba(6, 182, 212, 0.85)',
      },
      {
        label: 'Precision',
        data: Object.values(results.experiment_results).map(r => r.precision * 100),
        backgroundColor: 'rgba(16, 185, 129, 0.85)',
      }
    ]
  } : null;

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: { position: 'top', labels: { color: '#f8fafc', font: { family: 'Outfit' } } },
      title: { display: true, text: 'Hallucination Detection Technique Comparison (E1 to E7)', color: '#f8fafc' }
    },
    scales: {
      x: { ticks: { color: '#cbd5e1' }, grid: { color: 'rgba(255,255,255,0.05)' } },
      y: { ticks: { color: '#cbd5e1' }, grid: { color: 'rgba(255,255,255,0.05)' }, max: 100 }
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
      <div className="card-glass">
        <div className="card-header">
          <div className="card-title">
            <BarChart2 size={20} /> Experimental Matrix & Method Comparison (E1 to E7)
          </div>
          <div style={{ display: 'flex', gap: '12px' }}>
            <select
              className="form-select"
              style={{ width: '180px' }}
              value={selectedDomain}
              onChange={(e) => setSelectedDomain(e.target.value)}
            >
              <option value="">All Domains</option>
              {domains.map(d => (
                <option key={d.domain_key} value={d.domain_key}>{d.title}</option>
              ))}
            </select>

            <button className="btn-primary" onClick={handleRunMatrix} disabled={loading}>
              <Play size={16} /> {loading ? "Evaluating..." : "Run Benchmark"}
            </button>
          </div>
        </div>

        <p style={{ fontSize: '14px', color: '#94a3b8', marginBottom: '24px' }}>
          Quantitative evaluation across single-method baselines (E1: Self-Consistency, E2: Retrieval Verification, E3: Uncertainty Estimation) and multi-method fusion configurations (E4-E7) as specified in PRD Section 37.
        </p>

        {chartData && (
          <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '20px', borderRadius: '12px', marginBottom: '28px' }}>
            <Bar data={chartData} options={chartOptions} />
          </div>
        )}

        {/* Results Table */}
        {results && results.experiment_results && (
          <div>
            <h4 style={{ fontSize: '16px', fontWeight: '700', color: '#f8fafc', marginBottom: '14px' }}>
              Benchmark Metric Summary Table:
            </h4>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '14px' }}>
                <thead>
                  <tr style={{ background: 'rgba(15, 23, 42, 0.9)', color: '#94a3b8', borderBottom: '1px solid var(--border-subtle)' }}>
                    <th style={{ padding: '12px 16px' }}>Exp ID</th>
                    <th style={{ padding: '12px 16px' }}>Detection Technique</th>
                    <th style={{ padding: '12px 16px' }}>Precision</th>
                    <th style={{ padding: '12px 16px' }}>Recall</th>
                    <th style={{ padding: '12px 16px' }}>F1-Score</th>
                    <th style={{ padding: '12px 16px' }}>Accuracy</th>
                    <th style={{ padding: '12px 16px' }}>Avg Latency</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(results.experiment_results).map(([expId, expData]) => {
                    const isWinner = expId === 'E7';
                    return (
                      <tr
                        key={expId}
                        style={{
                          borderBottom: '1px solid var(--border-subtle)',
                          background: isWinner ? 'rgba(99, 102, 241, 0.12)' : 'transparent',
                          fontWeight: isWinner ? '700' : 'normal'
                        }}
                      >
                        <td style={{ padding: '12px 16px', fontFamily: 'var(--font-mono)', color: '#06b6d4' }}>
                          {expId} {isWinner && <Award size={14} style={{ color: '#f59e0b', inlineSize: 'auto' }} />}
                        </td>
                        <td style={{ padding: '12px 16px', color: '#f8fafc' }}>{expData.experiment_name}</td>
                        <td style={{ padding: '12px 16px', color: '#10b981' }}>{(expData.precision * 100).toFixed(1)}%</td>
                        <td style={{ padding: '12px 16px', color: '#10b981' }}>{(expData.recall * 100).toFixed(1)}%</td>
                        <td style={{ padding: '12px 16px', color: '#6366f1', fontSize: '15px' }}>
                          {(expData.f1_score * 100).toFixed(1)}%
                        </td>
                        <td style={{ padding: '12px 16px', color: '#06b6d4' }}>{(expData.accuracy * 100).toFixed(1)}%</td>
                        <td style={{ padding: '12px 16px', color: '#94a3b8' }}>{expData.avg_runtime_ms} ms</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
