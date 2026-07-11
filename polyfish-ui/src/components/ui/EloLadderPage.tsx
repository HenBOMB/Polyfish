import { useEffect, useState, useCallback, useMemo } from 'react';

interface LadderAnchor {
  name: string;
  path: string;
  elo: number;
  frozen_iteration: number | null;
  frozen_at: string | null;
}

interface LadderReading {
  at: string;
  run_id: string;
  iteration: number | null;
  kind: string;
  model: string;
  opponent: string;
  games: number;
  wins: number;
  losses: number;
  draws: number;
  win_rate: number;
  elo_est: number;
  avg_score_model: number;
  avg_score_opponent: number;
}

interface LadderData {
  anchors: LadderAnchor[];
  readings: LadderReading[];
}

const KIND_COLORS: Record<string, string> = {
  gauge: '#00f0ff',
  audit: '#fb923c',
  backfill: '#64748b',
  link: '#4ade80',
};

function shortModelLabel(model: string) {
  const base = model.split('/').pop() || model;
  return base.length > 28 ? base.slice(0, 25) + '…' : base;
}

function EloLadderPage() {
  const [data, setData] = useState<LadderData>({ anchors: [], readings: [] });
  const [error, setError] = useState<string | null>(null);
  const [hovered, setHovered] = useState<number | null>(null);

  const load = useCallback(async () => {
    try {
      const res = await fetch('http://localhost:3000/api/elo-ladder');
      if (!res.ok) throw new Error(await res.text());
      const json = await res.json();
      setData({ anchors: json.anchors || [], readings: json.readings || [] });
      setError(null);
    } catch {
      setError('Failed to load Elo ladder data.');
    }
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, 10000);
    return () => clearInterval(t);
  }, [load]);

  const readings = useMemo(
    () => [...data.readings].sort((a, b) => (a.at || '').localeCompare(b.at || '')),
    [data.readings]
  );

  const margin = { top: 16, right: 20, bottom: 16, left: 70 };
  const svgWidth = 1200;
  const height = 340;
  const chartWidth = svgWidth - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;

  const { yMin, yMax, yTicks } = useMemo(() => {
    if (readings.length === 0) return { yMin: -100, yMax: 100, yTicks: [-100, -50, 0, 50, 100] };
    let lo = Infinity;
    let hi = -Infinity;
    for (const r of readings) {
      if (r.elo_est < lo) lo = r.elo_est;
      if (r.elo_est > hi) hi = r.elo_est;
    }
    for (const a of data.anchors) {
      if (a.elo < lo) lo = a.elo;
      if (a.elo > hi) hi = a.elo;
    }
    const range = hi - lo || 100;
    const pad = range * 0.15;
    lo -= pad;
    hi += pad;
    const step = Math.pow(10, Math.floor(Math.log10((hi - lo) / 4 || 1)));
    const niceLo = Math.floor(lo / step) * step;
    const niceHi = Math.ceil(hi / step) * step;
    const ticks: number[] = [];
    for (let v = niceLo; v <= niceHi + step * 0.01; v += step) ticks.push(Math.round(v));
    return { yMin: niceLo, yMax: niceHi, yTicks: ticks };
  }, [readings, data.anchors]);

  const scaleX = (i: number) =>
    margin.left + (readings.length > 1 ? (i / (readings.length - 1)) * chartWidth : chartWidth / 2);
  const scaleY = (v: number) => margin.top + chartHeight - ((v - yMin) / (yMax - yMin || 1)) * chartHeight;

  return (
    <section>
      <div className="metrics-page-header">
        <div>
          <h2 className="card-title" style={{ margin: 0 }}>Elo Ladder</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '4px' }}>
            Head-to-head strength gauge results against frozen historical checkpoints ({readings.length} readings).
          </p>
        </div>
      </div>

      {error && (
        <div style={{ background: 'rgba(255, 42, 42, 0.1)', color: 'var(--neon-red)', padding: '12px', borderRadius: '4px', border: '1px solid var(--neon-red)', fontFamily: 'var(--font-mono)', marginBottom: '16px' }}>
          [ERROR] {error}
        </div>
      )}

      <div className="mc-card mc-full-width" style={{ marginBottom: '20px' }}>
        <div className="mc-header">
          <h3 className="mc-title">Elo estimate over time</h3>
        </div>
        <div className="mc-legend">
          {Object.entries(KIND_COLORS).map(([kind, color]) => (
            <span key={kind} className="mc-legend-item">
              <span className="mc-legend-swatch" style={{ background: color }} />
              {kind}
            </span>
          ))}
        </div>
        <div className="mc-body" style={{ minHeight: height }}>
          {readings.length === 0 ? (
            <span className="mc-empty">No data yet</span>
          ) : (
            <svg width="100%" height="100%" viewBox={`0 0 ${svgWidth} ${height}`} preserveAspectRatio="xMidYMid meet">
              {yTicks.map((tick, i) => (
                <g key={`yt-${i}`}>
                  <line x1={margin.left} y1={scaleY(tick)} x2={svgWidth - margin.right} y2={scaleY(tick)} stroke="rgba(255,255,255,0.07)" strokeWidth="1" strokeDasharray={tick === 0 ? undefined : '4 4'} />
                  <text x={margin.left - 10} y={scaleY(tick) + 8} textAnchor="end" fill="#6b7a90" fontSize="26" fontFamily="'Fira Code', monospace">{tick}</text>
                </g>
              ))}
              <polyline
                fill="none"
                stroke="#00f0ff"
                strokeWidth="2.5"
                strokeLinejoin="round"
                strokeLinecap="round"
                points={readings.map((r, i) => `${scaleX(i)},${scaleY(r.elo_est)}`).join(' ')}
              />
              {readings.map((r, i) => (
                <circle
                  key={i}
                  cx={scaleX(i)}
                  cy={scaleY(r.elo_est)}
                  r={hovered === i ? 8 : 6}
                  fill={hovered === i ? (KIND_COLORS[r.kind] || '#00f0ff') : '#0a0f19'}
                  stroke={KIND_COLORS[r.kind] || '#00f0ff'}
                  strokeWidth="2"
                />
              ))}
              {readings.map((_r, i) => {
                const x = scaleX(i);
                const halfStep = readings.length > 1 ? chartWidth / (readings.length - 1) / 2 : chartWidth / 2;
                return (
                  <rect
                    key={`hover-${i}`}
                    x={x - halfStep}
                    y={margin.top}
                    width={halfStep * 2}
                    height={chartHeight}
                    fill="transparent"
                    onMouseEnter={() => setHovered(i)}
                    onMouseLeave={() => setHovered(null)}
                  />
                );
              })}
              {hovered !== null && (
                <line x1={scaleX(hovered)} y1={margin.top} x2={scaleX(hovered)} y2={margin.top + chartHeight} stroke="rgba(255,255,255,0.2)" strokeWidth="1" strokeDasharray="4 4" pointerEvents="none" />
              )}
            </svg>
          )}
          {hovered !== null && readings[hovered] && (
            <div className="mc-tooltip" style={{ left: 20, top: 20, minWidth: '220px' }}>
              <div className="mc-tooltip-iter">{shortModelLabel(readings[hovered].model)} vs {readings[hovered].opponent}</div>
              <div className="mc-tooltip-val" style={{ color: KIND_COLORS[readings[hovered].kind] || '#00f0ff' }}>
                {Math.round(readings[hovered].elo_est)} elo ({readings[hovered].kind})
              </div>
              <div className="mc-tooltip-val" style={{ color: 'var(--text-muted)', fontSize: '0.78rem', fontWeight: 400 }}>
                {readings[hovered].wins}-{readings[hovered].losses}-{readings[hovered].draws} · {(readings[hovered].win_rate * 100).toFixed(1)}% win rate
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="mc-card mc-full-width">
        <div className="mc-header">
          <h3 className="mc-title">Frozen anchors</h3>
        </div>
        <div style={{ padding: '4px 16px 16px', overflowX: 'auto' }}>
          {data.anchors.length === 0 ? (
            <span className="mc-empty">No anchors yet</span>
          ) : (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--font-mono)', fontSize: '0.85rem' }}>
              <thead>
                <tr style={{ color: 'var(--text-muted)', textAlign: 'left', borderBottom: '1px solid var(--border-subtle)' }}>
                  <th style={{ padding: '6px 8px' }}>Name</th>
                  <th style={{ padding: '6px 8px' }}>Elo</th>
                  <th style={{ padding: '6px 8px' }}>Frozen iter</th>
                  <th style={{ padding: '6px 8px' }}>Frozen at</th>
                </tr>
              </thead>
              <tbody>
                {data.anchors.map((a) => (
                  <tr key={a.name} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                    <td style={{ padding: '6px 8px', color: 'var(--text-main)' }}>{a.name}</td>
                    <td style={{ padding: '6px 8px', color: 'var(--neon-cyan)' }}>{a.elo.toFixed(1)}</td>
                    <td style={{ padding: '6px 8px', color: 'var(--text-muted)' }}>{a.frozen_iteration ?? '—'}</td>
                    <td style={{ padding: '6px 8px', color: 'var(--text-muted)' }}>{a.frozen_at ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </section>
  );
}

export default EloLadderPage;
