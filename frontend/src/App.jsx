import { useState } from 'react'
import './App.css'
import RadarChart from './components/RadarChart'

function App() {
  const [input, setInput] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const analyzeResponse = async () => {
    const responses = input
      .split('\n')
      .map((response) => response.trim())
      .filter((response) => response.length > 0)

    if (responses.length === 0) {
      setError('Please enter at least one response.')
      return
    }

    setLoading(true)
    setError('')

    try {
      const response = await fetch('http://127.0.0.1:8000/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          responses: responses,
          token_confidence: 0.8,
        }),
      })

      if (!response.ok) {
        throw new Error('Analysis request failed.')
      }

      const data = await response.json()

      setResult(data)
    } catch (err) {
      setError(
        'Could not connect to the analysis server. Make sure FastAPI is running.'
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">

      {/* Header */}
      <header className="header">
        <div>
          <h1>Hallucination Detection</h1>
          <p>Domain-Specific LLM Output Analysis</p>
        </div>

        <div className="status">
          <span className="status-dot"></span>
          Member 4 Engine
        </div>
      </header>


      {/* Input Section */}
      <section className="input-section">
        <h2>Analyze LLM Responses</h2>

        <textarea
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder={
            'Enter multiple LLM responses, one per line...\n\nExample:\nPython was created by Guido van Rossum.\nGuido van Rossum created Python.'
          }
          rows="8"
        ></textarea>

        <button
          className="analyze-btn"
          onClick={analyzeResponse}
          disabled={loading}
        >
          {loading ? 'Analyzing...' : 'Analyze Response'}
        </button>

        {error && (
          <p className="error-message">
            {error}
          </p>
        )}
      </section>


      {/* Results */}
      {result && (
        <>
          <section className="results-section">

            <h2>Analysis Results</h2>

            <div className="metrics-grid">

              <div className="metric-card">
                <h3>Self-Consistency</h3>

                <div className="metric-value">
                  {result.consistency?.score?.toFixed(2)}
                </div>

                <p>
                  Risk:{' '}
                  {result.consistency?.risk?.toFixed(2)}
                </p>
              </div>


              <div className="metric-card">
                <h3>Semantic Entropy</h3>

                <div className="metric-value">
                  {result.semantic_entropy?.entropy?.toFixed(4)}
                </div>

                <p>
                  Normalized:{' '}
                  {result.semantic_entropy?.normalized?.toFixed(4)}
                </p>
              </div>


              <div className="metric-card">
                <h3>Uncertainty</h3>

                <div className="metric-value">
                  {result.uncertainty?.score?.toFixed(4)}
                </div>

                <p>
                  Confidence:{' '}
                  {result.uncertainty?.token_confidence?.toFixed(2)}
                </p>
              </div>


              <div className="metric-card">
                <h3>Claims Extracted</h3>

                <div className="metric-value">
                  {result.claims?.length || 0}
                </div>

                <p>Atomic claims detected</p>
              </div>

            </div>
          </section>


          {/* Radar Chart */}
          <section className="chart-section">

            <h2>Analysis Overview</h2>

            <div className="chart-container">

              <RadarChart
                consistency={
                  result.consistency?.score || 0
                }

                entropy={
                  result.semantic_entropy?.normalized || 0
                }

                uncertainty={
                  result.uncertainty?.score || 0
                }

                confidence={
                  result.uncertainty?.token_confidence || 0
                }
              />

            </div>
          </section>


          {/* Atomic Claims */}
          <section className="claims-section">

            <h2>Atomic Claims</h2>

            <div className="claims-list">

              {result.claims?.map((claim, index) => (

                <div
                  className="claim"
                  key={index}
                >
                  <span>{index + 1}</span>

                  {claim}
                </div>

              ))}

            </div>

          </section>

        </>
      )}

    </div>
  )
}

export default App