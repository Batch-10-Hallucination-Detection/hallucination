const API_BASE = "http://127.0.0.1:8000/api";

export async function fetchDomains() {
  try {
    const res = await fetch(`${API_BASE}/domains`);
    return await res.json();
  } catch (err) {
    console.error("API error fetching domains:", err);
    return { domains: [] };
  }
}

export async function fetchSamples(domain = "") {
  try {
    const url = domain ? `${API_BASE}/dataset/samples?domain=${domain}` : `${API_BASE}/dataset/samples`;
    const res = await fetch(url);
    return await res.json();
  } catch (err) {
    console.error("API error fetching samples:", err);
    return { samples: [] };
  }
}

export async function analyzePrompt(payload) {
  try {
    const res = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Analysis request failed.");
    }
    return await res.json();
  } catch (err) {
    console.error("API error analyzing prompt:", err);
    throw err;
  }
}

export async function runEvaluationMatrix(domain = null) {
  try {
    const res = await fetch(`${API_BASE}/evaluation/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ domain })
    });
    return await res.json();
  } catch (err) {
    console.error("API error running evaluation:", err);
    throw err;
  }
}
