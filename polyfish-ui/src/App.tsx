import { useState, useEffect, useMemo } from 'react';
import { Routes, Route, NavLink, Navigate } from 'react-router-dom';
import './index.css';

interface GameState {
  evaluation?: {
    advantage: number;
  };
  policyDistribution?: Record<string, number>;
  mctsAnalysis?: {
    total_iterations: number;
    evaluations: any[];
    principal_variation?: string[];
  };
  state?: {
    settings?: {
      mode: number;
    };
  };
}

interface TrainStatus {
  isRunning: boolean;
  pid: number | null;
  log: string;
}

interface Milestone {
  id: string;
  targetIter: number;
  message: string;
}

const getModeName = (mode?: number) => {
  switch (mode) {
    case 1:
    case 3: return 'Perfection';
    case 2:
    case 4: return 'Domination';
    case 5: return 'Custom';
    case 6: return 'Sandbox';
    case 7: return 'Tutorial';
    default: return `Unknown (${mode})`;
  }
};

function App() {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [trainStatus, setTrainStatus] = useState<TrainStatus | null>(null);
  const [metrics, setMetrics] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [cpuUsage, setCpuUsage] = useState<number[]>([]);
  const [cores, setCores] = useState<number>(12);
  const [activeTab, setActiveTab] = useState<string>('overview');
  const [hiddenSeries, setHiddenSeries] = useState<Record<string, boolean>>({});

  const toggleSeries = (key: string) => {
    setHiddenSeries(prev => ({ ...prev, [key]: !prev[key] }));
  };
  const [tribes, setTribes] = useState<string[]>(['Imperius', 'Imperius']);
  const [iterations, setIterations] = useState<number>(1000);
  const [learningRate, setLearningRate] = useState<number>(0.001);
  const [mctsIters, setMctsIters] = useState<number>(200);
  const [gamemode, setGamemode] = useState<number>(2);
  const [milestones, setMilestones] = useState<Milestone[]>([]);
  const [acknowledged, setAcknowledged] = useState<string[]>(() => {
    try { return JSON.parse(localStorage.getItem('ackMilestones') || '[]'); } catch { return []; }
  });
  const [pendingMilestone, setPendingMilestone] = useState<Milestone | null>(null);

  const updateConfig = async (updates: Partial<{cores: number, tribes: string[], iterations: number, learningRate: number, mctsIters: number, gamemode: number, milestones: Milestone[]}>) => {
    const newConfig = { cores, tribes, iterations, learningRate, mctsIters, gamemode, milestones, ...updates };
    await fetch('http://localhost:3000/config', { 
      method: 'POST', 
      headers: { 'Content-Type': 'application/json' }, 
      body: JSON.stringify(newConfig) 
    }).catch(console.error);
  };

  useEffect(() => {
    // Fetch initial config
    fetch('http://localhost:3000/config')
      .then(res => res.json())
      .then(data => {
        if (data.cores) setCores(data.cores);
        if (data.tribes) setTribes(data.tribes);
        if (data.iterations) setIterations(data.iterations);
        if (data.learningRate) setLearningRate(data.learningRate);
        if (data.mctsIters) setMctsIters(data.mctsIters);
        if (data.gamemode) setGamemode(data.gamemode);
        if (data.milestones) setMilestones(data.milestones);
      })
      .catch(console.error);

    // Poll the game state, training status, and metrics every 1 second
    const pollInterval = setInterval(async () => {
      try {
        const [stateRes, trainRes, metricsRes, cpuRes] = await Promise.all([
          fetch('http://localhost:3000/current').catch(() => null),
          fetch('http://localhost:3000/train/status').catch(() => null),
          fetch('http://localhost:3000/metrics').catch(() => null),
          fetch('http://localhost:3000/system/cpu').catch(() => null),
        ]);

        if (cpuRes && cpuRes.ok) {
          const cpuData = await cpuRes.json();
          setCpuUsage(cpuData.cores || []);
        }

        if (stateRes && stateRes.ok) {
          const stateData = await stateRes.json();
          setGameState(prevState => ({
            ...stateData,
            policyDistribution: stateData.policyDistribution || prevState?.policyDistribution,
            mctsAnalysis: stateData.mctsAnalysis || prevState?.mctsAnalysis
          }));
          setError(null);
        } else {
          setError('Backend disconnected or not responding (Game State).');
        }

        if (trainRes && trainRes.ok) {
          const trainData = await trainRes.json();
          setTrainStatus(trainData);
        }
        
        if (metricsRes && metricsRes.ok) {
          const metricsData = await metricsRes.json();
          setMetrics(metricsData.metrics || []);
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

  const triggerAnalysis = async () => {
    try {
      const res = await fetch('http://localhost:3000/autostep', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ iterations: mctsIters, dry_run: true }),
      });
      if (res.ok) {
        const data = await res.json();
        setGameState(prevState => ({
          ...prevState,
          policyDistribution: data.policyDistribution,
          mctsAnalysis: data.mctsAnalysis
        }));
      }
    } catch (err) {
      console.error("Failed to analyze:", err);
    }
  };


  const currentIter = useMemo(() => {
    let iter = 0;
    if (trainStatus?.isRunning) {
      if (metrics.length > 0) {
        iter = metrics[metrics.length - 1].iteration;
      }
      const iterMatches = trainStatus.log.match(/Starting Iteration (\d+)/g);
      if (iterMatches && iterMatches.length > 0) {
        const matchText = iterMatches[iterMatches.length - 1];
        const parsedIter = parseInt(matchText.replace('Starting Iteration ', ''), 10);
        if (!isNaN(parsedIter) && parsedIter > iter) {
          iter = parsedIter;
        }
      }
    }
    return iter;
  }, [metrics, trainStatus]);

  useEffect(() => {
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js';
    script.async = true;
    document.head.appendChild(script);
    return () => { document.head.removeChild(script); };
  }, []);

  useEffect(() => {
    if (currentIter > 0) {
      const reached = milestones.find(m => currentIter >= m.targetIter && !acknowledged.includes(m.id));
      if (reached && !pendingMilestone) {
        setPendingMilestone(reached);
      }
    }
  }, [currentIter, milestones, acknowledged, pendingMilestone]);

  const triggerConfetti = () => {
    if ((window as any).confetti) {
      const duration = 5 * 1000;
      const animationEnd = Date.now() + duration;
      const defaults = { startVelocity: 30, spread: 360, ticks: 60, zIndex: 10000 };

      const randomInRange = (min: number, max: number) => Math.random() * (max - min) + min;

      const interval: any = setInterval(function() {
        const timeLeft = animationEnd - Date.now();
        if (timeLeft <= 0) return clearInterval(interval);

        const particleCount = 50 * (timeLeft / duration);
        (window as any).confetti(Object.assign({}, defaults, { particleCount,
          origin: { x: randomInRange(0.1, 0.3), y: Math.random() - 0.2 }
        }));
        (window as any).confetti(Object.assign({}, defaults, { particleCount,
          origin: { x: randomInRange(0.7, 0.9), y: Math.random() - 0.2 }
        }));
      }, 250);
    }
  };
  const progressPercent = iterations > 0 ? Math.min(100, Math.max(0, (currentIter / iterations) * 100)) : 0;

  // Calculate Speed
  let itersPerHour = 0;
  let gamesPerMinute = 0;
  if (metrics.length > 1) {
    const recentMetrics = metrics.slice(-10);
    const startMetric = recentMetrics[0];
    const endMetric = recentMetrics[recentMetrics.length - 1];
    const timeDiffSeconds = endMetric.timestamp - startMetric.timestamp;
    if (timeDiffSeconds > 0) {
      const iterDiff = endMetric.iteration - startMetric.iteration;
      itersPerHour = (iterDiff / timeDiffSeconds) * 3600;
      // Assuming 10 games per iteration based on bash script defaults
      gamesPerMinute = (itersPerHour * 10) / 60;
    }
  }

  return (
    <div className="dashboard-layout">
      {pendingMilestone && (
        <div className="milestone-overlay">
          <div className="milestone-content">
            <div className="milestone-title">Milestone Reached!</div>
            <div className="milestone-desc">{pendingMilestone.message}</div>
            <button 
              className="milestone-btn" 
              onClick={() => {
                triggerConfetti();
                const newAck = [...acknowledged, pendingMilestone.id];
                setAcknowledged(newAck);
                localStorage.setItem('ackMilestones', JSON.stringify(newAck));
                setPendingMilestone(null);
              }}
            >
              Claim Reward & Continue
            </button>
          </div>
        </div>
      )}
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-title">Polyfish AI</div>
        <nav style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '20px' }}>
          <NavLink to="/" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`} end>Overview</NavLink>
          <NavLink to="/metrics" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>Self-Play Metrics</NavLink>
          <NavLink to="/mcts" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>MCTS Stats</NavLink>
          <NavLink to="/settings" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>Settings</NavLink>
        </nav>
        
        <div style={{ marginTop: 'auto', padding: '20px', borderTop: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>System Status</div>
          <div style={{ display: 'flex', alignItems: 'center', marginTop: '8px', fontFamily: 'var(--font-mono)', fontSize: '0.9rem', color: 'var(--text-main)' }}>
            <div className={`pulse-indicator ${trainStatus?.isRunning ? '' : 'idle'}`}></div>
            {trainStatus?.isRunning ? 'ONLINE' : 'STANDBY'}
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <h1 className="header-title">Training Console</h1>
            <p style={{ color: 'var(--text-muted)', fontSize: '1rem', fontFamily: 'var(--font-mono)', marginTop: '4px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              {!error ? (
                <div className="pulse-indicator" style={{ margin: 0 }}></div>
              ) : (
                <div className="pulse-indicator idle" style={{ margin: 0 }}></div>
              )}
              {trainStatus?.isRunning ? `> PID: ${trainStatus.pid} // Core sequence active` : (!error ? '> Backend online // Awaiting training initialization' : '> Backend offline // System idle')}
            </p>
          </div>
          <div style={{ display: 'flex', gap: '24px', alignItems: 'center' }}>
            {!error ? (
              <button 
                className="cyber-button"
                disabled
              >
                Server Online
              </button>
            ) : (
              <button 
                onClick={async () => {
                  await fetch('/api/start-server', { method: 'POST' }).catch(console.error);
                }}
                className="cyber-button"
              >
                Start Server
              </button>
            )}
            
            {trainStatus?.isRunning ? (
              <button 
                className="cyber-button"
                style={{ borderColor: 'var(--neon-red)', color: 'var(--neon-red)', boxShadow: '0 0 10px rgba(255,42,42,0.2) inset' }}
                onClick={async () => {
                  if (window.confirm("Are you sure you want to halt the training sequence? Unsaved progress in the current iteration may be lost.")) {
                    await fetch('http://localhost:3000/train/halt', { method: 'POST' }).catch(console.error);
                  }
                }}
              >
                Halt Training
              </button>
            ) : (
              <button 
                onClick={async () => {
                  await fetch('http://localhost:3000/train', { method: 'POST' }).catch(console.error);
                }}
                className="cyber-button purple"
                disabled={!!error}
              >
                Run Training
              </button>
            )}
          </div>
        </header>

        {/* Progress Bar */}
        {trainStatus?.isRunning && (
          <div style={{ marginTop: '20px', marginBottom: '20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', fontSize: '0.85rem', fontFamily: 'var(--font-mono)', color: 'var(--neon-cyan)' }}>
              <span>TRAINING PROGRESS // ITERATION {currentIter} OF {iterations}</span>
              <div style={{ display: 'flex', gap: '16px' }}>
                <span style={{ color: 'var(--text-muted)' }}>SPEED: {itersPerHour.toFixed(1)} ITERS/HR // {gamesPerMinute.toFixed(1)} GAMES/MIN</span>
                <span>{progressPercent.toFixed(1)}%</span>
              </div>
            </div>
            <div style={{ width: '100%', height: '8px', background: 'rgba(255, 255, 255, 0.05)', borderRadius: '4px', overflow: 'hidden', border: '1px solid var(--border-subtle)' }}>
              <div style={{ 
                width: `${progressPercent}%`, 
                height: '100%', 
                background: 'var(--neon-cyan)', 
                boxShadow: '0 0 10px var(--neon-cyan)',
                transition: 'width 0.5s ease'
              }}></div>
            </div>
          </div>
        )}

        {error && (
          <div style={{ background: 'rgba(255, 42, 42, 0.1)', color: 'var(--neon-red)', padding: '16px', borderRadius: '4px', border: '1px solid var(--neon-red)', fontFamily: 'var(--font-mono)' }}>
            [ERROR] {error}
          </div>
        )}

        <Routes>
          <Route path="/" element={
            <>
        <section className="kpi-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '20px', marginBottom: '24px' }}>
          {/* Row 1: High Level Training Outcomes */}
          {gameState?.state?.settings?.mode === 2 || gameState?.state?.settings?.mode === 4 ? (
            <div className="card" style={{ gridColumn: 'span 2', background: 'linear-gradient(135deg, rgba(255, 42, 42, 0.1) 0%, rgba(0,0,0,0.5) 100%)', borderColor: 'var(--neon-red)' }}>
              <h2 className="card-title">Military Averages (Last Iter)</h2>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px', marginTop: '8px' }}>
                <div>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', fontFamily: 'var(--font-mono)' }}>CAPTURES</div>
                  <div style={{ color: 'var(--neon-red)', fontSize: '1.5rem', fontWeight: 700, textShadow: '0 0 10px rgba(255,42,42,0.4)' }}>
                    {metrics.length > 0 ? metrics[metrics.length - 1].avg_captures.toFixed(1) : '0.0'}
                  </div>
                </div>
                <div>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', fontFamily: 'var(--font-mono)' }}>ATTACKS</div>
                  <div style={{ color: 'var(--neon-red)', fontSize: '1.5rem', fontWeight: 700, textShadow: '0 0 10px rgba(255,42,42,0.4)' }}>
                    {metrics.length > 0 ? metrics[metrics.length - 1].avg_attacks.toFixed(1) : '0.0'}
                  </div>
                </div>
                <div>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', fontFamily: 'var(--font-mono)' }}>ABILITIES</div>
                  <div style={{ color: 'var(--neon-red)', fontSize: '1.5rem', fontWeight: 700, textShadow: '0 0 10px rgba(255,42,42,0.4)' }}>
                    {metrics.length > 0 ? metrics[metrics.length - 1].avg_ability.toFixed(1) : '0.0'}
                  </div>
                </div>
              </div>
              <div className="card-trend-up" style={{ color: 'var(--neon-red)', marginTop: 'auto' }}>
                DOMINATION ROLLOUT MICROSTATS // SECONDARY: {metrics.length > 0 ? metrics[metrics.length - 1].avg_harvests.toFixed(1) : '0.0'} HARVESTS | {metrics.length > 0 ? metrics[metrics.length - 1].avg_builds.toFixed(1) : '0.0'} BUILDS | {metrics.length > 0 ? metrics[metrics.length - 1].avg_research.toFixed(1) : '0.0'} TECHS
              </div>
            </div>
          ) : (
            <div className="card" style={{ gridColumn: 'span 2', background: 'linear-gradient(135deg, rgba(176, 38, 255, 0.1) 0%, rgba(0,0,0,0.5) 100%)', borderColor: 'var(--neon-purple)' }}>
              <h2 className="card-title">Economy & Score (Last Iter)</h2>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px', marginTop: '8px' }}>
                <div>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', fontFamily: 'var(--font-mono)' }}>HIGH SCORE</div>
                  <div style={{ color: 'var(--neon-purple)', fontSize: '1.5rem', fontWeight: 700, textShadow: '0 0 10px rgba(176,38,255,0.4)' }}>
                    {metrics.length > 0 ? Math.max(...metrics.map(m => m.max_score)).toLocaleString() : '0'}
                  </div>
                </div>
                <div>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', fontFamily: 'var(--font-mono)' }}>HARVESTS</div>
                  <div style={{ color: 'var(--neon-purple)', fontSize: '1.5rem', fontWeight: 700, textShadow: '0 0 10px rgba(176,38,255,0.4)' }}>
                    {metrics.length > 0 ? metrics[metrics.length - 1].avg_harvests.toFixed(1) : '0.0'}
                  </div>
                </div>
                <div>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', fontFamily: 'var(--font-mono)' }}>BUILDS</div>
                  <div style={{ color: 'var(--neon-purple)', fontSize: '1.5rem', fontWeight: 700, textShadow: '0 0 10px rgba(176,38,255,0.4)' }}>
                    {metrics.length > 0 ? metrics[metrics.length - 1].avg_builds.toFixed(1) : '0.0'}
                  </div>
                </div>
              </div>
              <div className="card-trend-up" style={{ color: 'var(--neon-purple)', marginTop: 'auto' }}>
                PERFECTION ROLLOUT MICROSTATS // SECONDARY: {metrics.length > 0 ? metrics[metrics.length - 1].avg_captures.toFixed(1) : '0.0'} CAPTURES | {metrics.length > 0 ? metrics[metrics.length - 1].avg_attacks.toFixed(1) : '0.0'} ATTACKS | {metrics.length > 0 ? metrics[metrics.length - 1].avg_ability.toFixed(1) : '0.0'} ABILITIES
              </div>
            </div>
          )}
          
          <div className="card" style={{ gridColumn: 'span 2', background: 'linear-gradient(135deg, rgba(0, 240, 255, 0.1) 0%, rgba(0,0,0,0.5) 100%)', borderColor: 'var(--neon-cyan)' }}>
            <h2 className="card-title">Total Games Played</h2>
            <div className="card-value" style={{ fontSize: '2.5rem', color: 'var(--neon-cyan)' }}>
              {currentIter * 10}
            </div>
            <div className="card-trend-up" style={{ color: 'var(--neon-cyan)' }}>LIFETIME SELF-PLAY // LOCAL</div>
          </div>

          {/* Row 2: Current Game Inference Diagnostics */}
          <div className="card" style={{ gridColumn: 'span 2', background: 'linear-gradient(135deg, rgba(57, 255, 20, 0.1) 0%, rgba(0,0,0,0.5) 100%)', borderColor: 'var(--neon-green)' }}>
            <h2 className="card-title">P1 Advantage Score</h2>
            <div className="card-value" style={{ fontSize: '2rem', color: 'var(--neon-green)', textShadow: '0 0 15px rgba(57, 255, 20, 0.4)' }}>
              {gameState?.evaluation?.advantage !== undefined ? gameState.evaluation.advantage.toFixed(4) : '0.0000'}
            </div>
            <div className="card-trend-up" style={{ color: 'var(--neon-green)' }}>V-HEAD EVAL // TENSOR</div>
          </div>
          
          <div className="card" style={{ gridColumn: 'span 1', background: 'linear-gradient(135deg, rgba(252, 238, 10, 0.1) 0%, rgba(0,0,0,0.5) 100%)', borderColor: 'var(--neon-yellow)' }}>
            <h2 className="card-title">Avg Steps</h2>
            <div className="card-value" style={{ color: 'var(--neon-yellow)', textShadow: '0 0 15px rgba(252, 238, 10, 0.4)' }}>
              {metrics.length > 0 ? metrics[metrics.length - 1].avg_steps.toFixed(1) : '0.0'} 
              <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginLeft: '8px', textShadow: 'none' }}>/ GAME</span>
            </div>
            <div className="card-trend-down" style={{ color: 'var(--neon-yellow)' }}>GAME LENGTH // LOCAL</div>
          </div>
          
          <div className="card" style={{ gridColumn: 'span 1', background: 'linear-gradient(135deg, rgba(255, 115, 0, 0.1) 0%, rgba(0,0,0,0.5) 100%)', borderColor: 'var(--neon-orange)' }}>
            <h2 className="card-title">Network Loss</h2>
            <div className="card-value" style={{ color: 'var(--neon-orange)', textShadow: '0 0 15px rgba(255, 115, 0, 0.4)' }}>
              {metrics.length > 0 ? metrics[metrics.length - 1].loss.toFixed(4) : '0.0000'}
            </div>
            <div className="card-trend-up" style={{ color: 'var(--neon-orange)' }}>TRAINING LOOP // ERROR</div>
          </div>
        </section>

        {/* Charts Grid */}
        <section className="chart-grid">
          <div className="card" style={{ gridColumn: 'span 2' }}>
            <h2 className="card-title">Engine Output Stream</h2>
            <div className="chart-placeholder" style={{ 
              justifyContent: 'flex-start', 
              alignItems: 'flex-start', 
              padding: '20px', 
              overflowY: 'auto',
              textAlign: 'left'
            }}>
              <pre style={{ margin: 0, color: 'var(--text-main)', fontSize: '0.85rem', lineHeight: '1.4' }}>
                {trainStatus?.log ? trainStatus.log.slice(-5000) : '> Waiting for process to start...'}
              </pre>
            </div>
          </div>
          <div className="card">
            <h2 className="card-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              Live Policy Distribution
              <button 
                onClick={triggerAnalysis} 
                className="cyber-button"
                style={{ fontSize: '0.7rem', padding: '4px 8px', minWidth: 'auto', border: '1px solid var(--neon-cyan)', background: 'transparent', color: 'var(--neon-cyan)', margin: 0 }}
              >
                ANALYZE STATE
              </button>
            </h2>
            <div className="chart-placeholder" style={{ 
              flexDirection: 'column',
              justifyContent: 'flex-start',
              padding: '20px',
              overflowY: 'auto'
            }}>
              {gameState?.policyDistribution ? (
                Object.entries(gameState.policyDistribution)
                  .sort(([,a], [,b]) => b - a)
                  .map(([key, prob]) => (
                    <div key={key} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%', marginBottom: '12px' }}>
                      <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>MOV_{key.padStart(3, '0')}</span>
                      <div style={{ flex: 1, margin: '0 16px', height: '4px', background: 'rgba(0,0,0,0.5)', borderRadius: '2px', overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: `${prob * 100}%`, background: 'var(--neon-cyan)', boxShadow: '0 0 8px var(--neon-cyan)' }}></div>
                      </div>
                      <span style={{ color: 'var(--neon-cyan)', fontSize: '0.9rem', width: '48px', textAlign: 'right' }}>{(prob * 100).toFixed(1)}%</span>
                    </div>
                  ))
              ) : (
                <div style={{ color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '8px', marginTop: 'auto', marginBottom: 'auto' }}>
                  <div className="pulse-indicator"></div> AWAITING INFERENCE...
                </div>
              )}
            </div>
          </div>
        </section>
        </>
          } />

          <Route path="/metrics" element={
        <section className="chart-grid">
          <div className="card" style={{ gridColumn: 'span 3' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h2 className="card-title">Self-Play Metrics</h2>
              <div style={{ display: 'flex', gap: '16px', fontSize: '0.85rem', fontFamily: 'var(--font-mono)' }}>
                <span onClick={() => toggleSeries('max_score')} style={{ cursor: 'pointer', opacity: hiddenSeries['max_score'] ? 0.4 : 1, color: 'var(--neon-purple)', display: 'flex', alignItems: 'center', gap: '6px' }}><div style={{width: '12px', height: '4px', background: 'var(--neon-purple)'}}></div> Max Score</span>
                <span onClick={() => toggleSeries('avg_score')} style={{ cursor: 'pointer', opacity: hiddenSeries['avg_score'] ? 0.4 : 1, color: 'var(--neon-cyan)', display: 'flex', alignItems: 'center', gap: '6px' }}><div style={{width: '12px', height: '4px', background: 'var(--neon-cyan)'}}></div> Avg Score</span>
              </div>
            </div>
            <div className="chart-placeholder" style={{ padding: '20px', height: '250px' }}>
              {metrics.length > 0 ? (
                <svg width="100%" height="100%" viewBox="0 0 1000 200" preserveAspectRatio="none">
                  {/* Grid Lines */}
                  {[0, 1, 2, 3, 4].map(i => (
                    <line key={i} x1="0" y1={i * 50} x2="1000" y2={i * 50} stroke="rgba(0, 240, 255, 0.1)" strokeWidth="1" />
                  ))}
                  
                  {/* Lines */}
                  {[
                    { key: 'max_score', color: 'var(--neon-purple)' },
                    { key: 'avg_score', color: 'var(--neon-cyan)' }
                  ].map(lineConfig => {
                    if (hiddenSeries[lineConfig.key]) return null;
                    const activeKeys = ['max_score', 'avg_score'].filter(k => !hiddenSeries[k]);
                    const maxPossible = activeKeys.length > 0 
                      ? Math.max(10, ...metrics.flatMap(d => activeKeys.map(k => d[k] || 0)))
                      : 10;
                    return (
                      <polyline 
                        key={`line-${lineConfig.key}`}
                        fill="none" 
                        stroke={lineConfig.color} 
                        strokeWidth="3" 
                        opacity={lineConfig.key === 'max_score' ? "0.6" : "1"}
                        points={metrics.map((m, i) => {
                          const x = (i / Math.max(1, metrics.length - 1)) * 1000;
                          const y = 200 - (((m as any)[lineConfig.key] || 0) / maxPossible) * 200;
                          return `${x},${y}`;
                        }).join(' ')} 
                      />
                    );
                  })}
                  
                  {/* Data Points */}
                  {[
                    { key: 'max_score', color: 'var(--neon-purple)' },
                    { key: 'avg_score', color: 'var(--neon-cyan)' }
                  ].map(lineConfig => {
                    if (hiddenSeries[lineConfig.key]) return null;
                    const activeKeys = ['max_score', 'avg_score'].filter(k => !hiddenSeries[k]);
                    const maxPossible = activeKeys.length > 0 
                      ? Math.max(10, ...metrics.flatMap(d => activeKeys.map(k => d[k] || 0)))
                      : 10;
                    return (
                      <g key={`points-${lineConfig.key}`}>
                        {metrics.length <= 50 && metrics.map((m, i) => {
                          const x = (i / Math.max(1, metrics.length - 1)) * 1000;
                          const y = 200 - (((m as any)[lineConfig.key] || 0) / maxPossible) * 200;
                          return <circle key={`point-${lineConfig.key}-${i}`} cx={x} cy={y} r="3" fill="var(--bg-base)" stroke={lineConfig.color} strokeWidth="1.5" />;
                        })}
                      </g>
                    );
                  })}
                </svg>
              ) : (
                <span style={{ color: 'var(--text-muted)' }}>No training data available. Run training loop to populate.</span>
              )}
            </div>
          </div>

          <div className="card" style={{ gridColumn: 'span 3', marginTop: '20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h2 className="card-title" style={{ margin: 0 }}>Moves by type (Economy) (Avg / game)</h2>
              <div style={{ display: 'flex', gap: '16px', fontSize: '0.85rem', fontFamily: 'var(--font-mono)' }}>
                <span onClick={() => toggleSeries('avg_harvests')} style={{ cursor: 'pointer', opacity: hiddenSeries['avg_harvests'] ? 0.4 : 1, color: '#ef4444', display: 'flex', alignItems: 'center', gap: '6px' }}><div style={{width: '12px', height: '4px', background: '#ef4444'}}></div> Harvests</span>
                <span onClick={() => toggleSeries('avg_builds')} style={{ cursor: 'pointer', opacity: hiddenSeries['avg_builds'] ? 0.4 : 1, color: '#f59e0b', display: 'flex', alignItems: 'center', gap: '6px' }}><div style={{width: '12px', height: '4px', background: '#f59e0b'}}></div> Builds</span>
                <span onClick={() => toggleSeries('avg_research')} style={{ cursor: 'pointer', opacity: hiddenSeries['avg_research'] ? 0.4 : 1, color: '#eab308', display: 'flex', alignItems: 'center', gap: '6px' }}><div style={{width: '12px', height: '4px', background: '#eab308'}}></div> Research</span>
              </div>
            </div>
            <div className="chart-placeholder" style={{ padding: '20px', height: '200px' }}>
              {metrics.length > 0 ? (
                <svg width="100%" height="100%" viewBox="0 0 1000 200" preserveAspectRatio="none">
                  {[0, 1, 2, 3, 4].map(i => (
                    <line key={`grid-eco-${i}`} x1="0" y1={i * 50} x2="1000" y2={i * 50} stroke="rgba(255, 255, 255, 0.05)" strokeWidth="1" />
                  ))}
                  {[
                    { key: 'avg_harvests', color: '#ef4444' },
                    { key: 'avg_builds', color: '#f59e0b' },
                    { key: 'avg_research', color: '#eab308' },
                  ].map(lineConfig => {
                    if (hiddenSeries[lineConfig.key]) return null;
                    const activeKeys = ['avg_harvests', 'avg_builds', 'avg_research'].filter(k => !hiddenSeries[k]);
                    const maxPossible = activeKeys.length > 0 
                      ? Math.max(10, ...metrics.flatMap(d => activeKeys.map(k => d[k] || 0)))
                      : 10;
                    return (
                      <polyline 
                        key={`line-eco-${lineConfig.key}`}
                        fill="none" 
                        stroke={lineConfig.color} 
                        strokeWidth="3" 
                        points={metrics.map((m, i) => {
                          const x = (i / Math.max(1, metrics.length - 1)) * 1000;
                          const y = 200 - (((m as any)[lineConfig.key] || 0) / maxPossible) * 200;
                          return `${x},${y}`;
                        }).join(' ')} 
                      />
                    );
                  })}
                </svg>
              ) : (
                <span style={{ color: 'var(--text-muted)' }}>No training data available. Run training loop to populate.</span>
              )}
            </div>
          </div>

          <div className="card" style={{ gridColumn: 'span 3', marginTop: '20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h2 className="card-title" style={{ margin: 0 }}>Moves by type (Military) (Avg / game)</h2>
              <div style={{ display: 'flex', gap: '16px', fontSize: '0.85rem', fontFamily: 'var(--font-mono)' }}>
                <span onClick={() => toggleSeries('avg_captures')} style={{ cursor: 'pointer', opacity: hiddenSeries['avg_captures'] ? 0.4 : 1, color: '#3b82f6', display: 'flex', alignItems: 'center', gap: '6px' }}><div style={{width: '12px', height: '4px', background: '#3b82f6'}}></div> Captures</span>
                <span onClick={() => toggleSeries('avg_attacks')} style={{ cursor: 'pointer', opacity: hiddenSeries['avg_attacks'] ? 0.4 : 1, color: '#14b8a6', display: 'flex', alignItems: 'center', gap: '6px' }}><div style={{width: '12px', height: '4px', background: '#14b8a6'}}></div> Attacks</span>
                <span onClick={() => toggleSeries('avg_ability')} style={{ cursor: 'pointer', opacity: hiddenSeries['avg_ability'] ? 0.4 : 1, color: '#8b5cf6', display: 'flex', alignItems: 'center', gap: '6px' }}><div style={{width: '12px', height: '4px', background: '#8b5cf6'}}></div> Ability</span>
                <span onClick={() => toggleSeries('avg_steps')} style={{ cursor: 'pointer', opacity: hiddenSeries['avg_steps'] ? 0.4 : 1, color: '#9ca3af', display: 'flex', alignItems: 'center', gap: '6px' }}><div style={{width: '12px', height: '4px', background: '#9ca3af'}}></div> Step</span>
              </div>
            </div>
            <div className="chart-placeholder" style={{ padding: '20px', height: '200px' }}>
              {metrics.length > 0 ? (
                <svg width="100%" height="100%" viewBox="0 0 1000 200" preserveAspectRatio="none">
                  {[0, 1, 2, 3, 4].map(i => (
                    <line key={`grid-mil-${i}`} x1="0" y1={i * 50} x2="1000" y2={i * 50} stroke="rgba(255, 255, 255, 0.05)" strokeWidth="1" />
                  ))}
                  {[
                    { key: 'avg_captures', color: '#3b82f6' },
                    { key: 'avg_attacks', color: '#14b8a6' },
                    { key: 'avg_ability', color: '#8b5cf6' },
                    { key: 'avg_steps', color: '#9ca3af' }
                  ].map(lineConfig => {
                    if (hiddenSeries[lineConfig.key]) return null;
                    const activeKeys = ['avg_captures', 'avg_attacks', 'avg_ability', 'avg_steps'].filter(k => !hiddenSeries[k]);
                    const maxPossible = activeKeys.length > 0 
                      ? Math.max(10, ...metrics.flatMap(d => activeKeys.map(k => d[k] || 0)))
                      : 10;
                    return (
                      <polyline 
                        key={`line-mil-${lineConfig.key}`}
                        fill="none" 
                        stroke={lineConfig.color} 
                        strokeWidth="3" 
                        points={metrics.map((m, i) => {
                          const x = (i / Math.max(1, metrics.length - 1)) * 1000;
                          const y = 200 - ((((m as any)[lineConfig.key] || 0) / maxPossible) * 200);
                          return `${x},${y}`;
                        }).join(' ')} 
                      />
                    );
                  })}
                </svg>
              ) : (
                <span style={{ color: 'var(--text-muted)' }}>No training data available. Run training loop to populate.</span>
              )}
            </div>
          </div>
        </section>
          } />

          <Route path="/mcts" element={
          <section className="chart-grid">
            <div className="card" style={{ gridColumn: 'span 3' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <h2 className="card-title" style={{ margin: 0 }}>MCTS Search Analysis</h2>
                <button 
                  onClick={triggerAutostep}
                  className="cyber-button"
                  style={{ padding: '6px 12px', fontSize: '0.85rem' }}
                  disabled={!!error}
                >
                  Run Sandbox Simulation
                </button>
              </div>
              
              {!gameState?.mctsAnalysis ? (
                <div style={{ color: 'var(--text-muted)', padding: '20px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div className="pulse-indicator"></div> Awaiting first simulation step...
                </div>
              ) : (
                <div style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                    <div style={{ background: 'rgba(0,0,0,0.3)', padding: '16px', borderRadius: '4px', border: '1px solid var(--border-subtle)' }}>
                      <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '8px', textTransform: 'uppercase' }}>Total Iterations</div>
                      <div style={{ fontSize: '1.5rem', color: 'var(--neon-cyan)', fontFamily: 'var(--font-mono)' }}>
                        {gameState.mctsAnalysis.total_iterations}
                      </div>
                    </div>
                    
                    <div style={{ background: 'rgba(0,0,0,0.3)', padding: '16px', borderRadius: '4px', border: '1px solid var(--border-subtle)' }}>
                      <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '8px', textTransform: 'uppercase' }}>Principal Variation (PV)</div>
                      <div style={{ fontSize: '0.9rem', color: 'var(--text-main)', fontFamily: 'var(--font-mono)', lineHeight: '1.5' }}>
                        {gameState.mctsAnalysis.principal_variation?.length > 0 
                          ? gameState.mctsAnalysis.principal_variation.join(' → ')
                          : 'No PV available'
                        }
                      </div>
                    </div>
                  </div>

                  <div>
                    <h3 style={{ fontSize: '1rem', color: 'var(--text-main)', marginBottom: '12px', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>Top Move Evaluations</h3>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      {gameState.mctsAnalysis.evaluations?.slice(0, 10).map((evalItem: any, idx: number) => (
                        <div key={idx} style={{ 
                          display: 'grid', 
                          gridTemplateColumns: '1fr 100px 100px', 
                          gap: '16px',
                          background: 'rgba(0,240,255,0.05)',
                          padding: '12px',
                          borderRadius: '4px',
                          borderLeft: `2px solid ${idx === 0 ? 'var(--neon-cyan)' : 'var(--border-subtle)'}`
                        }}>
                          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9rem' }}>
                            <span style={{ color: idx === 0 ? 'var(--neon-cyan)' : 'var(--text-main)' }}>{evalItem.description}</span>
                          </div>
                          <div style={{ textAlign: 'right', fontSize: '0.85rem' }}>
                            <span style={{ color: 'var(--text-muted)' }}>VISITS:</span>
                            <div style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-main)' }}>{evalItem.visits}</div>
                          </div>
                          <div style={{ textAlign: 'right', fontSize: '0.85rem' }}>
                            <span style={{ color: 'var(--text-muted)' }}>WIN RATE:</span>
                            <div style={{ fontFamily: 'var(--font-mono)', color: 'var(--neon-purple)' }}>{(evalItem.win_rate * 100).toFixed(1)}%</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </section>
          } />

          <Route path="/settings" element={
          <section className="chart-grid">
            <div className="card" style={{ gridColumn: 'span 2' }}>
              <h2 className="card-title" style={{ fontSize: '1.2rem', marginBottom: '24px' }}>System & Hardware</h2>
              
              <div className="setting-row">
                <div className="setting-label">
                  <span>Available Parallelism (CPU Cores)</span>
                  <span className="setting-value">{cores} / 16</span>
                </div>
                <input 
                  type="range" 
                  min="1" 
                  max="16" 
                  value={cores}
                  onChange={async (e) => {
                    const val = parseInt(e.target.value);
                    setCores(val);
                    await updateConfig({ cores: val });
                  }}
                  className="cyber-slider"
                />

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginTop: '16px' }}>
                  <div style={{ background: 'rgba(0,0,0,0.3)', padding: '16px', borderRadius: '4px', border: '1px solid var(--border-subtle)' }}>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '8px', fontFamily: 'var(--font-mono)' }}>LIVE CPU LOAD (PER CORE)</div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(16, 1fr)', gap: '2px', height: '40px' }}>
                      {cpuUsage.length > 0 ? cpuUsage.slice(1).map((usage, i) => (
                        <div key={i} style={{ 
                          height: '100%', 
                          background: 'rgba(255, 255, 255, 0.05)',
                          display: 'flex',
                          alignItems: 'flex-end',
                          borderRadius: '2px',
                          overflow: 'hidden'
                        }} title={`Core ${i}: ${usage.toFixed(1)}%`}>
                          <div style={{ 
                            width: '100%', 
                            height: `${usage}%`, 
                            background: usage > 90 ? '#ef4444' : usage > 60 ? '#f59e0b' : 'var(--neon-cyan)',
                            transition: 'height 0.2s ease, background 0.2s ease'
                          }}></div>
                        </div>
                      )) : <span style={{ color: 'var(--text-muted)' }}>Loading...</span>}
                    </div>
                  </div>
                  
                  <div style={{ background: 'rgba(0,0,0,0.3)', padding: '16px', borderRadius: '4px', border: '1px solid var(--border-subtle)' }}>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '8px', fontFamily: 'var(--font-mono)' }}>ACTUAL THROUGHPUT</div>
                    <div style={{ color: 'var(--neon-cyan)', fontSize: '1.5rem', fontFamily: 'var(--font-mono)' }}>
                      {trainStatus?.isRunning ? `${gamesPerMinute.toFixed(1)}` : '0.0'} <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>GAMES/MIN</span>
                    </div>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem', marginTop: '4px' }}>
                      Optimize for throughput, not CPU load.
                    </div>
                  </div>
                </div>

                <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', fontFamily: 'var(--font-sans)', marginTop: '8px' }}>
                  Hardware limits may still forcibly cap this value at runtime based on OS quotas.
                </div>
              </div>
            </div>

            <div className="card" style={{ gridColumn: 'span 2' }}>
              <h2 className="card-title" style={{ fontSize: '1.2rem', marginBottom: '24px' }}>Training Hyperparameters</h2>

              <div className="setting-row">
                <div className="setting-label">
                  <span>Training Loop Iterations</span>
                  <span className="setting-value">{iterations}</span>
                </div>
                <input 
                  type="range" 
                  min="100" 
                  max="10000" 
                  step="100"
                  value={iterations}
                  onChange={async (e) => {
                    const val = parseInt(e.target.value);
                    setIterations(val);
                    await updateConfig({ iterations: val });
                  }}
                  className="cyber-slider"
                />
                <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', fontFamily: 'var(--font-sans)', marginTop: '8px' }}>
                  Number of epochs before the training script completes.
                </div>
              </div>

              <div className="setting-row">
                <div className="setting-label">
                  <span>Learning Rate</span>
                  <span className="setting-value">{learningRate}</span>
                </div>
                <input 
                  type="range" 
                  min="0.0001" 
                  max="0.01" 
                  step="0.0001"
                  value={learningRate}
                  onChange={async (e) => {
                    const val = parseFloat(e.target.value);
                    setLearningRate(val);
                    await updateConfig({ learningRate: val });
                  }}
                  className="cyber-slider"
                />
                <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', fontFamily: 'var(--font-sans)', marginTop: '8px' }}>
                  Step size for neural network weight updates.
                </div>
              </div>

              <div className="setting-row">
                <div className="setting-label">
                  <span>MCTS Iterations</span>
                  <span className="setting-value">{mctsIters}</span>
                </div>
                <input 
                  type="range" 
                  min="50" 
                  max="2000" 
                  step="50"
                  value={mctsIters}
                  onChange={async (e) => {
                    const val = parseInt(e.target.value);
                    setMctsIters(val);
                    await updateConfig({ mctsIters: val });
                  }}
                  className="cyber-slider"
                />
                <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', fontFamily: 'var(--font-sans)', marginTop: '8px' }}>
                  Number of Monte Carlo Tree Search rollouts per move.
                </div>
              </div>

              {/* Placeholder for future settings */}
              <div className="setting-row" style={{ opacity: 0.5, pointerEvents: 'none' }}>
                <div className="setting-label">
                  <span>MCTS Batch Size</span>
                  <span className="setting-value">256</span>
                </div>
                <input type="range" min="32" max="1024" value="256" readOnly className="cyber-slider" />
                <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', fontFamily: 'var(--font-sans)', marginTop: '8px' }}>
                  [LOCKED] Requires neural network initialization.
                </div>
              </div>
            </div>

            <div className="card" style={{ gridColumn: 'span 2' }}>
              <h2 className="card-title" style={{ fontSize: '1.2rem', marginBottom: '24px' }}>Game & Simulation</h2>

              <div className="setting-row">
                <div className="setting-label">
                  <span>Game Mode</span>
                  <span className="setting-value">{getModeName(gamemode)}</span>
                </div>
                <div style={{ display: 'flex', gap: '8px', marginTop: '12px' }}>
                  {[1, 2].map(mode => {
                    const isActive = gamemode === mode;
                    return (
                      <button
                        key={mode}
                        onClick={async () => {
                          setGamemode(mode);
                          await updateConfig({ gamemode: mode });
                        }}
                        style={{
                          background: isActive ? 'rgba(0, 240, 255, 0.2)' : 'rgba(0,0,0,0.5)',
                          border: `1px solid ${isActive ? 'var(--neon-cyan)' : 'var(--border-subtle)'}`,
                          color: isActive ? 'var(--neon-cyan)' : 'var(--text-muted)',
                          padding: '6px 12px',
                          borderRadius: '4px',
                          cursor: 'pointer',
                          fontFamily: 'var(--font-mono)',
                          fontSize: '0.9rem',
                          textTransform: 'uppercase',
                          transition: 'all 0.2s',
                          boxShadow: isActive ? '0 0 10px rgba(0,240,255,0.2) inset' : 'none'
                        }}
                      >
                        {getModeName(mode)}
                      </button>
                    );
                  })}
                </div>
                <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', fontFamily: 'var(--font-sans)', marginTop: '8px' }}>
                  Select the active rule set (Perfection = 30 turns, Domination = eliminate all).
                </div>
              </div>

              <div className="setting-row">
                <div className="setting-label">
                  <span>Active Tribes Pool</span>
                  <span className="setting-value" style={{ fontSize: '1rem' }}>{tribes.length} SELECTED</span>
                </div>
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginTop: '12px' }}>
                  {['Imperius', 'Bardur', 'Oumaji', 'Kickoo', 'XinXi', 'Zebasi', 'AiMo', 'Vengir', 'Quetzali', 'Hoodrick', 'Yadakk'].map(tribe => {
                    const isActive = tribes.includes(tribe);
                    return (
                      <button
                        key={tribe}
                        onClick={async () => {
                          let newTribes;
                          if (isActive) {
                            newTribes = tribes.filter(t => t !== tribe);
                            if (newTribes.length < 2) newTribes = [tribe, tribe]; // enforce min 2
                          } else {
                            newTribes = [...tribes, tribe];
                          }
                          setTribes(newTribes);
                          await updateConfig({ tribes: newTribes });
                        }}
                        style={{
                          background: isActive ? 'rgba(0, 240, 255, 0.2)' : 'rgba(0,0,0,0.5)',
                          border: `1px solid ${isActive ? 'var(--neon-cyan)' : 'var(--border-subtle)'}`,
                          color: isActive ? 'var(--neon-cyan)' : 'var(--text-muted)',
                          padding: '6px 12px',
                          borderRadius: '4px',
                          cursor: 'pointer',
                          fontFamily: 'var(--font-mono)',
                          fontSize: '0.9rem',
                          textTransform: 'uppercase',
                          transition: 'all 0.2s',
                          boxShadow: isActive ? '0 0 10px rgba(0,240,255,0.2) inset' : 'none'
                        }}
                      >
                        {tribe}
                      </button>
                    );
                  })}
                </div>
                <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', fontFamily: 'var(--font-sans)', marginTop: '8px' }}>
                  Select the tribes available during self-play league generation.
                </div>
              </div>
            </div>

            <div className="card" style={{ gridColumn: 'span 2' }}>
              <h2 className="card-title" style={{ fontSize: '1.2rem', marginBottom: '24px' }}>Milestones & Rewards</h2>

              <div className="setting-row">
                <div className="setting-label">
                  <span>Training Milestones</span>
                  <span className="setting-value" style={{ fontSize: '1rem' }}>{milestones.length} SET</span>
                </div>
                <div style={{ marginTop: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {milestones.sort((a,b) => a.targetIter - b.targetIter).map(m => (
                    <div key={m.id} style={{ display: 'flex', gap: '16px', alignItems: 'center', background: 'rgba(0,0,0,0.4)', padding: '12px', borderRadius: '4px', border: '1px solid var(--border-subtle)' }}>
                      <div style={{ flex: 1 }}>
                        <div style={{ color: 'var(--neon-cyan)', fontFamily: 'var(--font-mono)', fontWeight: 'bold', textShadow: '0 0 5px var(--neon-cyan)' }}>ITERATION {m.targetIter}</div>
                        <div style={{ color: 'var(--text-main)', fontSize: '1rem' }}>{m.message}</div>
                      </div>
                      <button 
                        className="cyber-button" 
                        style={{ padding: '6px 12px', borderColor: 'var(--neon-red)', color: 'var(--neon-red)', boxShadow: '0 0 10px rgba(255,42,42,0.2) inset' }}
                        onClick={async () => {
                          const newM = milestones.filter(x => x.id !== m.id);
                          setMilestones(newM);
                          await updateConfig({ milestones: newM });
                        }}
                      >Remove</button>
                    </div>
                  ))}
                  <div style={{ display: 'flex', gap: '8px', marginTop: '8px' }}>
                    <input id="new-m-iter" type="number" placeholder="Target Iter" className="cyber-input no-spinners" style={{ width: '120px' }} />
                    <input id="new-m-msg" type="text" placeholder="Reward Message (e.g. Early Phase Complete)" className="cyber-input" style={{ flex: 1 }} />
                    <button 
                      className="cyber-button purple"
                      onClick={async () => {
                        const iter = parseInt((document.getElementById('new-m-iter') as HTMLInputElement).value);
                        const msg = (document.getElementById('new-m-msg') as HTMLInputElement).value;
                        if (iter && msg) {
                          const newM = [...milestones, { id: Math.random().toString(36).slice(2), targetIter: iter, message: msg }];
                          setMilestones(newM);
                          await updateConfig({ milestones: newM });
                          (document.getElementById('new-m-iter') as HTMLInputElement).value = '';
                          (document.getElementById('new-m-msg') as HTMLInputElement).value = '';
                        }
                      }}
                    >Add</button>
                  </div>
                </div>
                <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', fontFamily: 'var(--font-sans)', marginTop: '8px' }}>
                  Define custom milestones to receive notifications and visual rewards during long training sessions.
                </div>
              </div>
            </div>
          </section>
          } />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
