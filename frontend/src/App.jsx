import React, { useState, useEffect } from 'react';
import { apiService } from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('workspace');
  const [domain, setDomain] = useState('auto');
  const [prompt, setPrompt] = useState('What is the recommended initial dose of lisinopril for essential hypertension in non-hypovolemic adults, and what is the maximum recommended daily dosage?');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [experiments, setExperiments] = useState([]);

  // Live Domain Detection State
  const [detectedDomainInfo, setDetectedDomainInfo] = useState(null);

  // Datasets state
  const [datasetsSummary, setDatasetsSummary] = useState(null);
  const [selectedDatasetId, setSelectedDatasetId] = useState('medhallu');
  const [selectedDatasetDetails, setSelectedDatasetDetails] = useState(null);
  const [datasetFilter, setDatasetFilter] = useState('all');

  // Debounced Live Domain Classification
  useEffect(() => {
    if (!prompt || !prompt.trim()) {
      setDetectedDomainInfo(null);
      return;
    }

    const timer = setTimeout(() => {
      apiService.classifyDomain(prompt).then(res => {
        setDetectedDomainInfo(res.data);
      }).catch(console.error);
    }, 250);

    return () => clearTimeout(timer);
  }, [prompt]);

  const handleAnalyze = async () => {
    setLoading(true);
    try {
      const res = await apiService.submitAnalysis({
        domain,
        model: 'llama-3-8b',
        prompt,
        response: response || undefined,
        detection_methods: ['eigenscore', 'self_consistency', 'semantic_entropy', 'retrieval']
      });
      setResults(res.data);
      setActiveTab('claims');
    } catch (err) {
      console.error(err);
      alert('Error running analysis');
    } finally {
      setLoading(false);
    }
  };

  // Load evaluation metrics & datasets on mount
  useEffect(() => {
    apiService.getEvaluationMetrics().then(res => {
      setExperiments(res.data.experiments || []);
    }).catch(console.error);

    apiService.getDatasets().then(res => {
      setDatasetsSummary(res.data);
    }).catch(console.error);
  }, []);

  // Fetch dataset details when selected
  useEffect(() => {
    if (selectedDatasetId) {
      apiService.getDatasetDetails(selectedDatasetId).then(res => {
        setSelectedDatasetDetails(res.data);
      }).catch(console.error);
    }
  }, [selectedDatasetId]);

  const loadSampleIntoWorkspace = (sample) => {
    setDomain('auto');
    setPrompt(sample.prompt);
    setResponse(sample.response);
    setActiveTab('workspace');
  };

  const filteredDatasets = datasetsSummary?.all_datasets?.filter(d => {
    if (datasetFilter === 'all') return true;
    if (datasetFilter === 'primary') return d.tier === 'primary';
    if (datasetFilter === 'additional') return d.tier === 'additional';
    return d.domain === datasetFilter;
  }) || [];

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col font-sans">
      <header className="border-b border-slate-800 bg-slate-900/90 backdrop-blur sticky top-0 z-50 px-8 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-sky-500 to-indigo-600 flex items-center justify-center font-bold text-white shadow-md shadow-sky-500/20">
            H
          </div>
          <div>
            <h1 className="text-lg font-bold text-sky-400">HaluDetect Research Platform</h1>
            <p className="text-xs text-slate-400">Domain-Adaptive Detection & 14 Curated Benchmarks</p>
          </div>
        </div>

        <div className="flex gap-4 text-sm">
          <button onClick={() => setActiveTab('workspace')} className={`pb-1 transition ${activeTab === 'workspace' ? 'border-b-2 border-sky-400 text-white font-semibold' : 'text-slate-400 hover:text-slate-200'}`}>Workspace</button>
          <button onClick={() => setActiveTab('claims')} className={`pb-1 transition ${activeTab === 'claims' ? 'border-b-2 border-sky-400 text-white font-semibold' : 'text-slate-400 hover:text-slate-200'}`}>Claims</button>
          <button onClick={() => setActiveTab('datasets')} className={`pb-1 transition flex items-center gap-1.5 ${activeTab === 'datasets' ? 'border-b-2 border-sky-400 text-white font-semibold' : 'text-slate-400 hover:text-slate-200'}`}>
            Datasets
            <span className="px-1.5 py-0.2 rounded-full text-[10px] font-bold bg-sky-500/20 text-sky-300 border border-sky-500/30">14</span>
          </button>
          <button onClick={() => setActiveTab('evaluation')} className={`pb-1 transition ${activeTab === 'evaluation' ? 'border-b-2 border-sky-400 text-white font-semibold' : 'text-slate-400 hover:text-slate-200'}`}>Ablation (E1–E7)</button>
        </div>
      </header>

      <main className="p-8 max-w-6xl mx-auto w-full flex-1">
        {/* TAB 1: WORKSPACE */}
        {activeTab === 'workspace' && (
          <div className="bg-slate-800/40 border border-slate-800 rounded-2xl p-6 space-y-5">
            <div className="flex justify-between items-center">
              <div>
                <h2 className="text-lg font-semibold text-white">Domain-Aware Detection Workspace</h2>
                <p className="text-xs text-slate-400">System first identifies the query's domain and answers accordingly</p>
              </div>
              <button 
                onClick={() => setActiveTab('datasets')}
                className="text-xs text-sky-400 hover:text-sky-300 underline font-medium"
              >
                Browse 14 Benchmark Datasets &rarr;
              </button>
            </div>

            {/* Live Domain Classification Banner */}
            {detectedDomainInfo && (
              <div className="p-3.5 rounded-xl bg-gradient-to-r from-sky-950/40 via-indigo-950/30 to-slate-900 border border-sky-500/30 flex items-center justify-between text-xs">
                <div className="flex items-center gap-2.5">
                  <span className="flex h-2.5 w-2.5 relative">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-sky-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-sky-500"></span>
                  </span>
                  <div>
                    <span className="text-slate-400">Domain Identified: </span>
                    <span className="font-bold text-white uppercase tracking-wider">{detectedDomainInfo.domain_name}</span>
                    <span className="ml-2 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-sky-500/20 text-sky-300 border border-sky-500/30">
                      {(detectedDomainInfo.confidence * 100).toFixed(0)}% Confidence
                    </span>
                  </div>
                </div>
                <div className="text-[11px] text-slate-400 max-w-sm text-right truncate">
                  {detectedDomainInfo.rationale}
                </div>
              </div>
            )}

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-xs uppercase text-slate-400 font-bold block mb-1">Domain Mode</label>
                <select value={domain} onChange={e => setDomain(e.target.value)} className="bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm w-full text-slate-200">
                  <option value="auto">⚡ Auto-Detect Domain (Recommended)</option>
                  <option value="healthcare">Healthcare & Clinical Medicine</option>
                  <option value="software">Software Development</option>
                  <option value="biomedical">Biomedical Research</option>
                  <option value="general">General / Open Knowledge</option>
                </select>
              </div>
              <div>
                <label className="text-xs uppercase text-slate-400 font-bold block mb-1">Target Model</label>
                <select className="bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm w-full text-slate-200">
                  <option value="llama-3-8b">Meta-Llama-3-8B-Instruct</option>
                  <option value="mistral-7b">Mistral-7B-Instruct-v0.3</option>
                  <option value="qwen-2.5-7b">Qwen-2.5-7B-Instruct</option>
                </select>
              </div>
            </div>

            <div>
              <div className="flex justify-between items-center mb-1">
                <label className="text-xs uppercase text-slate-400 font-bold block">Prompt / Question</label>
                <span className="text-[11px] text-slate-500">Try medical, software code, biomedical, or general trivia queries</span>
              </div>
              <textarea 
                value={prompt} 
                onChange={e => setPrompt(e.target.value)} 
                rows={2} 
                className="bg-slate-900 border border-slate-700 rounded-lg p-3 text-sm w-full text-slate-200 focus:outline-none focus:border-sky-500" 
              />
            </div>

            {/* Quick Domain Presets */}
            <div className="flex items-center gap-2 pt-1 text-xs">
              <span className="text-slate-500 text-[11px] uppercase font-bold">Quick Examples:</span>
              <button
                onClick={() => {
                  setPrompt("In Python 3.11, how does asyncio.TaskGroup handle unhandled exceptions from concurrent tasks?");
                  setResponse("");
                }}
                className="px-2.5 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-sky-300 border border-slate-700 text-[11px] transition"
              >
                💻 Python Asyncio
              </button>
              <button
                onClick={() => {
                  setPrompt("What is the initial oral dose of lisinopril for essential hypertension in non-hypovolemic adults, and what is the maximum recommended dose?");
                  setResponse("");
                }}
                className="px-2.5 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-emerald-300 border border-slate-700 text-[11px] transition"
              >
                🩺 Lisinopril Dosage
              </button>
              <button
                onClick={() => {
                  setPrompt("Which gene encodes the catalytic subunit of PI3K alpha, and in which malignancies are activating mutations common?");
                  setResponse("");
                }}
                className="px-2.5 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-indigo-300 border border-slate-700 text-[11px] transition"
              >
                🧬 PIK3CA Oncology
              </button>
              <button
                onClick={() => {
                  setPrompt("When did the Apollo 11 lunar module land on the Moon, and which astronauts stepped onto the surface?");
                  setResponse("");
                }}
                className="px-2.5 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-amber-300 border border-slate-700 text-[11px] transition"
              >
                🌍 Apollo 11 Landing
              </button>
            </div>

            <div>
              <label className="text-xs uppercase text-slate-400 font-bold block mb-1">LLM Response (Optional)</label>
              <textarea 
                value={response} 
                onChange={e => setResponse(e.target.value)} 
                rows={3} 
                placeholder="Leave blank for domain-adaptive generation engine to automatically answer according to the detected domain..." 
                className="bg-slate-900 border border-slate-700 rounded-lg p-3 text-sm w-full text-slate-200 focus:outline-none focus:border-sky-500" 
              />
            </div>


            <button 
              onClick={handleAnalyze} 
              disabled={loading} 
              className="w-full py-3 rounded-lg bg-sky-500 hover:bg-sky-400 transition font-semibold text-white text-sm shadow-lg shadow-sky-500/20 disabled:opacity-50"
            >
              {loading ? 'Checking Domain, Generating Response & Running Multi-Signal Detection...' : 'Check Domain & Run Analysis'}
            </button>
          </div>
        )}

        {/* TAB 2: CLAIMS */}
        {activeTab === 'claims' && (
          <div className="space-y-4">
            <h2 className="text-lg font-semibold text-white">Claim-Level Verification</h2>
            {results ? (
              <div className="space-y-4">
                {/* Domain Verification Header */}
                <div className="p-4 rounded-xl bg-slate-800/80 border border-slate-700 space-y-3">
                  <div className="flex justify-between items-center">
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-xs text-slate-400">Domain Identified:</span>
                        <span className="px-2 py-0.5 rounded uppercase text-xs font-bold bg-sky-500/20 text-sky-400 border border-sky-500/30">
                          {results.domain_name || results.domain}
                        </span>
                        {results.domain_confidence && (
                          <span className="text-xs text-emerald-400 font-mono">
                            {(results.domain_confidence * 100).toFixed(0)}% Confidence
                          </span>
                        )}
                      </div>
                      <span className="text-xs text-slate-300 block">{results.domain_rationale}</span>
                    </div>

                    <div className="text-right">
                      <span className="text-xs text-slate-400 block">Overall Risk:</span>
                      <span className="text-xl font-bold">{(results.overall_risk_score * 100).toFixed(1)}%</span>
                    </div>
                  </div>

                  {/* Domain-Adapted Response Generated */}
                  <div className="pt-2 border-t border-slate-700/80">
                    <span className="text-[11px] uppercase font-bold text-slate-400 block mb-1">Domain-Tailored Answer Generated:</span>
                    <p className="text-xs text-slate-200 bg-slate-900/90 p-3 rounded-lg border border-slate-800 font-mono leading-relaxed">
                      {results.response}
                    </p>
                  </div>
                </div>

                {/* Claims Breakdown */}
                {results.claims.map((c, i) => (
                  <div key={i} className="p-4 rounded-xl bg-slate-800/40 border border-slate-800 space-y-2">
                    <div className="flex justify-between items-center text-xs">
                      <span className="font-mono text-slate-400">{c.claim_id}</span>
                      <span className={`px-2 py-0.5 rounded font-bold ${c.final_score >= 0.5 ? 'bg-rose-500/20 text-rose-400' : 'bg-emerald-500/20 text-emerald-400'}`}>
                        {c.classification.toUpperCase()}
                      </span>
                    </div>
                    <div className="text-sm font-medium">"{c.claim}"</div>
                    <div className="grid grid-cols-4 gap-2 text-xs pt-2">
                      <div className="p-2 bg-slate-900 rounded">EigenScore: {c.signals.eigenscore?.toFixed(2) ?? '--'}</div>
                      <div className="p-2 bg-slate-900 rounded">Self-Consistency: {c.signals.self_consistency?.toFixed(2) ?? '--'}</div>
                      <div className="p-2 bg-slate-900 rounded">Semantic Entropy: {c.signals.semantic_entropy?.toFixed(2) ?? '--'}</div>
                      <div className="p-2 bg-slate-900 rounded">Retrieval: {c.signals.retrieval?.toFixed(2) ?? '--'}</div>
                    </div>
                    <div className="text-xs text-slate-400 pt-1"><strong>Attribution:</strong> {c.explanation}</div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-slate-500 text-sm">No analysis results yet. Run a prompt in Workspace.</p>
            )}
          </div>
        )}

        {/* TAB 3: DATASETS (14 DATASETS) */}
        {activeTab === 'datasets' && (
          <div className="space-y-6">
            <div className="flex justify-between items-center">
              <div>
                <h2 className="text-lg font-semibold text-white">Benchmark Dataset Suite (14 Loaded)</h2>
                <p className="text-xs text-slate-400">Curated claim-level benchmarks across Healthcare, Software, and General QA</p>
              </div>
              <div className="flex gap-2 text-xs">
                {['all', 'primary', 'additional', 'healthcare', 'software'].map(filterKey => (
                  <button
                    key={filterKey}
                    onClick={() => setDatasetFilter(filterKey)}
                    className={`px-3 py-1.5 rounded-lg capitalize border ${
                      datasetFilter === filterKey 
                        ? 'bg-sky-500 border-sky-400 text-white font-semibold' 
                        : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
                    }`}
                  >
                    {filterKey}
                  </button>
                ))}
              </div>
            </div>

            {/* Datasets Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredDatasets.map(ds => {
                const isSelected = selectedDatasetId === ds.id;
                return (
                  <div
                    key={ds.id}
                    onClick={() => setSelectedDatasetId(ds.id)}
                    className={`p-4 rounded-xl border cursor-pointer transition flex flex-col justify-between ${
                      isSelected 
                        ? 'bg-sky-950/30 border-sky-500 shadow-md shadow-sky-500/10' 
                        : 'bg-slate-800/40 border-slate-800 hover:border-slate-700'
                    }`}
                  >
                    <div>
                      <div className="flex justify-between items-start mb-2">
                        <span className="font-bold text-white text-sm">{ds.name}</span>
                        <span className={`text-[10px] px-2 py-0.5 rounded uppercase font-bold tracking-wider ${
                          ds.tier === 'primary' 
                            ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' 
                            : 'bg-sky-500/20 text-sky-400 border border-sky-500/30'
                        }`}>
                          {ds.tier}
                        </span>
                      </div>
                      <p className="text-xs text-slate-300 font-medium mb-1">{ds.category}</p>
                      <p className="text-[11px] text-slate-400 line-clamp-2 mb-3">{ds.description}</p>
                    </div>

                    <div className="pt-2 border-t border-slate-800/80 flex justify-between items-center text-[11px] text-slate-400">
                      <span className="capitalize font-mono text-slate-300">{ds.domain}</span>
                      <span>{ds.claims_count} claims</span>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Selected Dataset Details & Sample Explorer */}
            {selectedDatasetDetails && (
              <div className="bg-slate-800/60 border border-slate-700 rounded-2xl p-6 space-y-4">
                <div className="flex justify-between items-center border-b border-slate-700 pb-3">
                  <div>
                    <h3 className="text-base font-bold text-white flex items-center gap-2">
                      {selectedDatasetDetails.metadata.name}
                      <span className="text-xs font-normal text-slate-400">({selectedDatasetDetails.metadata.category})</span>
                    </h3>
                    <p className="text-xs text-slate-300 mt-1">{selectedDatasetDetails.metadata.description}</p>
                    <p className="text-xs text-sky-400 font-medium mt-0.5">Focus: {selectedDatasetDetails.metadata.focus}</p>
                  </div>
                </div>

                <div className="space-y-4">
                  <h4 className="text-xs uppercase font-bold text-slate-400 tracking-wider">Benchmark Samples & Annotated Claims</h4>
                  {selectedDatasetDetails.samples?.map((sample, idx) => (
                    <div key={idx} className="p-4 bg-slate-900/90 rounded-xl border border-slate-800 space-y-3">
                      <div className="flex justify-between items-center text-xs">
                        <span className="font-mono text-slate-400">{sample.id} &bull; {sample.topic}</span>
                        <button
                          onClick={() => loadSampleIntoWorkspace(sample)}
                          className="px-3 py-1 rounded bg-sky-500/20 hover:bg-sky-500/40 text-sky-300 border border-sky-500/30 font-medium text-xs transition"
                        >
                          Load into Workspace &rarr;
                        </button>
                      </div>

                      <div>
                        <span className="text-[11px] uppercase font-bold text-slate-500 block mb-0.5">Prompt:</span>
                        <p className="text-xs text-slate-200">{sample.prompt}</p>
                      </div>

                      <div>
                        <span className="text-[11px] uppercase font-bold text-slate-500 block mb-0.5">Model Response:</span>
                        <p className="text-xs text-slate-300 bg-slate-950 p-2.5 rounded-lg border border-slate-800/80 font-mono">{sample.response}</p>
                      </div>

                      {/* Claims List */}
                      <div>
                        <span className="text-[11px] uppercase font-bold text-slate-500 block mb-1">Ground Truth Claims:</span>
                        <div className="space-y-1.5">
                          {sample.claims?.map((claim, cIdx) => (
                            <div key={cIdx} className="flex justify-between items-center p-2 rounded bg-slate-800/60 border border-slate-800 text-xs">
                              <span className="text-slate-300 font-medium">"{claim.text}"</span>
                              <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                                claim.ground_truth === 0 
                                  ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' 
                                  : 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                              }`}>
                                {claim.label.toUpperCase()} {claim.ground_truth === 1 ? '(Hallucination)' : '(Faithful)'}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 4: EVALUATION (E1-E7) */}
        {activeTab === 'evaluation' && (
          <div className="space-y-4">
            <h2 className="text-lg font-semibold text-white">E1–E7 Experimental Ablation Suite</h2>
            <div className="overflow-x-auto bg-slate-800/40 border border-slate-800 rounded-2xl">
              <table className="min-w-full text-xs text-left">
                <thead className="bg-slate-800 text-slate-400 border-b border-slate-700">
                  <tr>
                    <th className="p-3">Exp</th>
                    <th className="p-3">Configuration</th>
                    <th className="p-3">Accuracy</th>
                    <th className="p-3">Precision</th>
                    <th className="p-3">Recall</th>
                    <th className="p-3 text-sky-400">F1</th>
                    <th className="p-3 text-indigo-400">AUROC</th>
                    <th className="p-3">Latency</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800">
                  {experiments.map((e, idx) => (
                    <tr key={idx} className={e.experiment === 'E7' ? 'bg-sky-950/20 font-bold' : ''}>
                      <td className="p-3 font-mono">{e.experiment}</td>
                      <td className="p-3">{e.configuration}</td>
                      <td className="p-3">{e.accuracy?.toFixed(4) ?? '--'}</td>
                      <td className="p-3">{e.precision?.toFixed(4) ?? '--'}</td>
                      <td className="p-3">{e.recall?.toFixed(4) ?? '--'}</td>
                      <td className="p-3 text-sky-400">{e.f1_score?.toFixed(4) ?? '--'}</td>
                      <td className="p-3 text-indigo-400">{e.auroc?.toFixed(4) ?? '--'}</td>
                      <td className="p-3">{e.latency_ms ?? '--'}ms</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
