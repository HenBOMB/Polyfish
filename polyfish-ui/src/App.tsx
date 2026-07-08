import { useState, useEffect } from 'react';
import './index.css';

interface GameState {
  evaluation?: {
    advantage: number;
  };
  policyDistribution?: Record<string, number>;
  mctsAnalysis?: {
    total_iterations: number;
    evaluations: any[];
  };
}

interface TrainStatus {
  isRunning: boolean;
  pid: number | null;
  log: string;
}

function App() {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [trainStatus, setTrainStatus] = useState<TrainStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Poll the game state and training status every 1 second
    const pollInterval = setInterval(async () => {
      try {
        const [stateRes, trainRes] = await Promise.all([
          fetch('http://localhost:3000/current').catch(() => null),
          fetch('http://localhost:3000/train/status').catch(() => null),
        ]);

        if (stateRes && stateRes.ok) {
          const stateData = await stateRes.json();
          setGameState(stateData);
          setError(null);
        } else {
          setError('Backend disconnected or not responding (Game State).');
        }

        if (trainRes && trainRes.ok) {
          const trainData = await trainRes.json();
          setTrainStatus(trainData);
        }
      } catch (err) {
        setError('Failed to fetch telemetry from Rust backend.');
      }
    }, 1000);

    return () => clearInterval(pollInterval);
  }, []);

  const triggerAutostep = async () => {
    try {
      const res = await fetch('http://localhost:3000/autostep', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ iterations: 400 }),
      });
      if (res.ok) {
        const data = await res.json();
        setGameState(data);
      }
    } catch (err) {
      console.error("Failed to autostep:", err);
    }
  };

  return (
    <div className="dashboard-layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-title">Polyfish AI</div>
        <nav style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <a href="#" style={{ color: 'var(--color-on-primary)', textDecoration: 'none', fontWeight: 500 }}>Overview</a>
          <a href="#" style={{ color: 'var(--color-border)', textDecoration: 'none', fontWeight: 400 }}>Self-Play Metrics</a>
          <a href="#" style={{ color: 'var(--color-border)', textDecoration: 'none', fontWeight: 400 }}>MCTS Stats</a>
        </nav>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1 className="header-title">Training Analytics</h1>
            <p style={{ color: 'var(--color-secondary)', fontSize: '0.875rem' }}>
              {trainStatus?.isRunning ? `Training active (PID: ${trainStatus.pid})` : 'Training idle / Interactive mode'}
            </p>
          </div>
          <div>
            <button 
              onClick={triggerAutostep}
              style={{
                background: 'var(--color-primary)',
                color: 'var(--color-on-primary)',
                border: 'none',
                padding: '8px 16px',
                borderRadius: '4px',
                cursor: 'pointer',
                fontFamily: 'var(--font-mono)',
                fontWeight: 600
              }}
            >
              Trigger AI Step
            </button>
          </div>
        </header>

        {error && (
          <div style={{ background: '#331111', color: '#ff4444', padding: '12px', borderRadius: '4px', marginBottom: '16px', border: '1px solid #551111' }}>
            ⚠️ {error}
          </div>
        )}

        {/* KPI Grid */}
        <section className="kpi-grid">
          <div className="card">
            <h2 className="card-title">P1 Advantage Score</h2>
            <div className="card-value">
              {gameState?.evaluation?.advantage !== undefined ? gameState.evaluation.advantage.toFixed(2) : '--'}
            </div>
            <div className="card-trend-up">Value Head Evaluation</div>
          </div>
          <div className="card">
            <h2 className="card-title">Policy Distribution Spread</h2>
            <div className="card-value">
              {gameState?.policyDistribution ? Object.keys(gameState.policyDistribution).length : 0} Moves
            </div>
            <div className="card-trend-down">Considered by Neural Net</div>
          </div>
          <div className="card">
            <h2 className="card-title">MCTS Iterations</h2>
            <div className="card-value">
              {gameState?.mctsAnalysis?.total_iterations || 0}
            </div>
            <div className="card-trend-up">Search Depth</div>
          </div>
          <div className="card">
            <h2 className="card-title">Status</h2>
            <div className="card-value" style={{ fontSize: '1.25rem' }}>
              {trainStatus?.isRunning ? 'Running' : 'Polling...'}
            </div>
            <div className="card-trend-down">Auto-refresh: 1s</div>
          </div>
        </section>

        {/* Charts Grid */}
        <section className="chart-grid">
          <div className="card" style={{ gridColumn: 'span 2' }}>
            <h2 className="card-title">Trainer Log Tail</h2>
            <div className="chart-placeholder" style={{ 
              justifyContent: 'flex-start', 
              alignItems: 'flex-start', 
              padding: '16px', 
              overflowY: 'auto',
              textAlign: 'left'
            }}>
              <pre style={{ margin: 0, color: 'var(--color-on-surface)' }}>
                {trainStatus?.log || 'No active training logs detected.'}
              </pre>
            </div>
          </div>
          <div className="card">
            <h2 className="card-title">Live Policy Probabilities</h2>
            <div className="chart-placeholder" style={{ 
              flexDirection: 'column',
              justifyContent: 'flex-start',
              padding: '16px',
              overflowY: 'auto'
            }}>
              {gameState?.policyDistribution ? (
                Object.entries(gameState.policyDistribution)
                  .sort(([,a], [,b]) => b - a)
                  .map(([key, prob]) => (
                    <div key={key} style={{ display: 'flex', justifyContent: 'space-between', width: '100%', marginBottom: '4px' }}>
                      <span style={{ fontSize: '0.75rem', color: 'var(--color-border)' }}>Move {key}</span>
                      <span style={{ color: 'var(--color-primary)' }}>{(prob * 100).toFixed(1)}%</span>
                    </div>
                  ))
              ) : (
                <span style={{ color: 'var(--color-border)' }}>Awaiting MCTS data...</span>
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
