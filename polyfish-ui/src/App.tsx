import './index.css';

function App() {
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
        <header>
          <h1 className="header-title">Training Analytics</h1>
          <p style={{ color: 'var(--color-secondary)', fontSize: '0.875rem' }}>Epoch 10 | 530 avg moves/game</p>
        </header>

        {/* KPI Grid */}
        <section className="kpi-grid">
          <div className="card">
            <h2 className="card-title">Avg Captures</h2>
            <div className="card-value">6.0</div>
            <div className="card-trend-down">↓ 7.6% (Trending Down)</div>
          </div>
          <div className="card">
            <h2 className="card-title">Avg Harvests</h2>
            <div className="card-value">12.8</div>
            <div className="card-trend-down">↓ 8.5% (Trending Down)</div>
          </div>
          <div className="card">
            <h2 className="card-title">Avg Research</h2>
            <div className="card-value">22.9</div>
            <div className="card-trend-up">↑ 2.2% (Local Optimum)</div>
          </div>
          <div className="card">
            <h2 className="card-title">Stalled Moves</h2>
            <div className="card-value">~402</div>
            <div className="card-trend-down">⚠️ 75% of total moves</div>
          </div>
        </section>

        {/* Charts Grid */}
        <section className="chart-grid">
          <div className="card">
            <h2 className="card-title">Moves by Type</h2>
            <div className="chart-placeholder">
              [ Stacked Line Chart Visualization ]
            </div>
          </div>
          <div className="card">
            <h2 className="card-title">Value Head Accuracy (Win Prob)</h2>
            <div className="chart-placeholder">
              [ Value vs MCTS Visit Count ]
            </div>
          </div>
          <div className="card">
            <h2 className="card-title">Policy Distribution</h2>
            <div className="chart-placeholder">
              [ Unexplored Moves (0.0 Policy) ]
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
