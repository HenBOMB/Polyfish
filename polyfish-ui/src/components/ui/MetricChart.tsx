import { useRef, useState, useMemo, useCallback } from 'react';

interface SeriesConfig {
  key: string;
  label: string;
  color: string;
}

interface MetricChartProps {
  title: string;
  data: any[];
  series: SeriesConfig[];
  height?: number;
  iterationKey?: string;
}

function MetricChart({ title, data, series, height = 220, iterationKey = 'iteration' }: MetricChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredCol, setHoveredCol] = useState<{
    index: number;
    x: number;
    containerWidth: number;
  } | null>(null);
  const [hiddenKeys, setHiddenKeys] = useState<Record<string, boolean>>({});

  const toggleKey = (key: string) => {
    setHiddenKeys(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const visibleSeries = useMemo(() => series.filter(s => !hiddenKeys[s.key]), [series, hiddenKeys]);

  // Aggregate duplicate iterations
  const aggregatedData = useMemo(() => {
    if (!data || data.length === 0) return [];
    
    const groups = new Map<number, any[]>();
    for (const d of data) {
      const iter = d[iterationKey] ?? 0;
      if (!groups.has(iter)) groups.set(iter, []);
      groups.get(iter)!.push(d);
    }

    const result = [];
    for (const [iter, items] of groups.entries()) {
      const merged: any = { [iterationKey]: iter };
      for (const s of series) {
        let sum = 0;
        let count = 0;
        for (const item of items) {
          if (typeof item[s.key] === 'number') {
            sum += item[s.key];
            count++;
          }
        }
        merged[s.key] = count > 0 ? sum / count : 0;
      }
      result.push(merged);
    }
    
    // Sort by iteration just to be safe
    return result.sort((a, b) => a[iterationKey] - b[iterationKey]);
  }, [data, series, iterationKey]);

  // Chart layout
  const margin = { top: 12, right: 16, bottom: 44, left: 80 };
  const svgWidth = 1000;
  const chartWidth = svgWidth - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;

  // Y-axis auto-scale
  const { yMin, yMax, yTicks } = useMemo(() => {
    if (aggregatedData.length === 0 || visibleSeries.length === 0)
      return { yMin: 0, yMax: 10, yTicks: [0, 2.5, 5, 7.5, 10] };

    let rawMin = Infinity;
    let rawMax = -Infinity;
    for (const d of aggregatedData) {
      for (const s of visibleSeries) {
        const val = d[s.key];
        if (val !== undefined && val !== null) {
          if (val < rawMin) rawMin = val;
          if (val > rawMax) rawMax = val;
        }
      }
    }
    if (!isFinite(rawMin)) { rawMin = 0; rawMax = 10; }

    // Pad range
    const range = rawMax - rawMin || 1;
    const pad = range * 0.12;
    let lo = rawMin - pad;
    let hi = rawMax + pad;

    // Don't go below zero if all values are non-negative
    const allPositive = aggregatedData.every(d => visibleSeries.every(s => (d[s.key] ?? 0) >= 0));
    if (allPositive && lo < 0) lo = 0;

    // Nice tick generation
    const tickCount = 5;
    const rawStep = (hi - lo) / (tickCount - 1);

    // Round step to a "nice" number
    const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
    const residual = rawStep / mag;
    let niceStep: number;
    if (residual <= 1.5) niceStep = 1 * mag;
    else if (residual <= 3) niceStep = 2 * mag;
    else if (residual <= 7) niceStep = 5 * mag;
    else niceStep = 10 * mag;

    const niceLo = Math.floor(lo / niceStep) * niceStep;
    const niceHi = Math.ceil(hi / niceStep) * niceStep;

    const ticks: number[] = [];
    for (let v = niceLo; v <= niceHi + niceStep * 0.01; v += niceStep) {
      ticks.push(Math.round(v * 1e6) / 1e6);
    }

    return { yMin: niceLo, yMax: niceHi, yTicks: ticks };
  }, [aggregatedData, visibleSeries]);

  // X-axis ticks: show iteration labels at reasonable density, no duplicates
  const xTicks = useMemo(() => {
    if (aggregatedData.length === 0) return [];

    // Build unique iteration → first data index mapping
    const seen = new Map<string, number>();
    for (let i = 0; i < aggregatedData.length; i++) {
      const label = String(aggregatedData[i][iterationKey] ?? i + 1);
      if (!seen.has(label)) seen.set(label, i);
    }
    const uniqueEntries = Array.from(seen.entries()); // [label, firstIndex][]

    if (uniqueEntries.length <= 15) {
      return uniqueEntries.map(([label, index]) => ({ index, label }));
    }

    // Sample ~12 evenly-spaced unique labels
    const step = Math.ceil(uniqueEntries.length / 12);
    const result: { index: number; label: string }[] = [];
    for (let i = 0; i < uniqueEntries.length; i += step) {
      result.push({ index: uniqueEntries[i][1], label: uniqueEntries[i][0] });
    }
    // Always include last
    const last = uniqueEntries[uniqueEntries.length - 1];
    if (result[result.length - 1]?.label !== last[0]) {
      result.push({ index: last[1], label: last[0] });
    }
    return result;
  }, [aggregatedData, iterationKey]);

  const scaleX = useCallback(
    (index: number) => margin.left + (index / Math.max(1, aggregatedData.length - 1)) * chartWidth,
    [aggregatedData.length, chartWidth, margin.left]
  );

  const scaleY = useCallback(
    (value: number) => {
      const range = yMax - yMin || 1;
      return margin.top + chartHeight - ((value - yMin) / range) * chartHeight;
    },
    [yMin, yMax, chartHeight, margin.top]
  );

  // PNG export via canvas
  const exportPNG = useCallback(() => {
    if (!svgRef.current) return;

    const svgEl = svgRef.current;
    const serializer = new XMLSerializer();
    let svgString = serializer.serializeToString(svgEl);

    // Inline computed styles for export
    svgString = svgString.replace(/var\(--text-muted\)/g, '#6b7a90');
    svgString = svgString.replace(/var\(--bg-base\)/g, '#030305');
    svgString = svgString.replace(/var\(--font-mono\)/g, "'Fira Code', monospace");

    const canvas = document.createElement('canvas');
    const scale = 2;
    canvas.width = svgWidth * scale;
    canvas.height = height * scale;
    const ctx = canvas.getContext('2d')!;
    ctx.scale(scale, scale);

    const img = new Image();
    img.onload = () => {
      ctx.fillStyle = '#0a0f19';
      ctx.fillRect(0, 0, svgWidth, height);
      ctx.drawImage(img, 0, 0, svgWidth, height);
      const link = document.createElement('a');
      link.download = `${title.replace(/[^a-zA-Z0-9]/g, '_').toLowerCase()}.png`;
      link.href = canvas.toDataURL('image/png');
      link.click();
    };
    img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgString)));
  }, [title, height]);

  // Format tick label
  const fmtTick = (v: number) => {
    if (Math.abs(v) >= 100) return v.toFixed(0);
    if (v % 1 === 0) return v.toFixed(0);
    if (Math.abs(v) >= 10) return v.toFixed(1);
    return v.toFixed(1);
  };

  return (
    <div className="mc-card">
      <div className="mc-header">
        <h3 className="mc-title">{title}</h3>
        <div className="mc-actions">
          <button className="mc-png-btn" onClick={exportPNG} title="Export as PNG">PNG</button>
        </div>
      </div>

      {series.length > 1 && (
        <div className="mc-legend">
          {series.map(s => (
            <span
              key={s.key}
              className={`mc-legend-item ${hiddenKeys[s.key] ? 'mc-legend-hidden' : ''}`}
              onClick={() => toggleKey(s.key)}
            >
              <span className="mc-legend-swatch" style={{ background: hiddenKeys[s.key] ? '#444' : s.color }} />
              {s.label}
            </span>
          ))}
        </div>
      )}

      <div className="mc-body" ref={containerRef}>
        {aggregatedData.length === 0 ? (
          <span className="mc-empty">No data yet</span>
        ) : (
          <svg
            ref={svgRef}
            width="100%"
            height="100%"
            viewBox={`0 0 ${svgWidth} ${height}`}
            preserveAspectRatio="xMidYMid meet"
          >
            {/* Y-axis gridlines + labels */}
            {yTicks.map((tick, i) => (
              <g key={`yt-${i}`}>
                <line
                  x1={margin.left}
                  y1={scaleY(tick)}
                  x2={svgWidth - margin.right}
                  y2={scaleY(tick)}
                  stroke="rgba(255,255,255,0.07)"
                  strokeWidth="1"
                  strokeDasharray={i === 0 ? undefined : '4 4'}
                />
                <text
                  x={margin.left - 10}
                  y={scaleY(tick) + 8}
                  textAnchor="end"
                  fill="#6b7a90"
                  fontSize="26"
                  fontFamily="'Fira Code', monospace"
                >
                  {fmtTick(tick)}
                </text>
              </g>
            ))}

            {/* X-axis labels */}
            {xTicks.map((tick, i) => (
              <text
                key={`xt-${i}`}
                x={scaleX(tick.index)}
                y={height - 8}
                textAnchor="middle"
                fill="#6b7a90"
                fontSize="26"
                fontFamily="'Fira Code', monospace"
              >
                {tick.label}
              </text>
            ))}

            {/* Data lines + points */}
            {visibleSeries.map(s => {
              const points = aggregatedData.map((d, i) => ({
                x: scaleX(i),
                y: scaleY(d[s.key] ?? 0),
                val: d[s.key] ?? 0,
                iter: d[iterationKey] ?? i + 1,
              }));

              return (
                <g key={s.key} pointerEvents="none">
                  {/* Line */}
                  <polyline
                    fill="none"
                    stroke={s.color}
                    strokeWidth="2.5"
                    strokeLinejoin="round"
                    strokeLinecap="round"
                    points={points.map(p => `${p.x},${p.y}`).join(' ')}
                  />
                  {/* Points */}
                  {points.map((p, i) => (
                    <circle
                      key={i}
                      cx={p.x}
                      cy={p.y}
                      r={hoveredCol?.index === i ? "8" : "6"}
                      fill={hoveredCol?.index === i ? s.color : "#0a0f19"}
                      stroke={s.color}
                      strokeWidth="2"
                    />
                  ))}
                </g>
              );
            })}

            {/* Hover Column Catchers */}
            {aggregatedData.map((d, i) => {
              const x = scaleX(i);
              const halfStep = i > 0 ? (x - scaleX(i - 1)) / 2 : (aggregatedData.length > 1 ? (scaleX(1) - x) / 2 : 20);
              return (
                <rect
                  key={`hover-${i}`}
                  x={x - halfStep}
                  y={margin.top}
                  width={halfStep * 2}
                  height={chartHeight}
                  fill="transparent"
                  onMouseEnter={() => {
                    const container = containerRef.current;
                    const svg = svgRef.current;
                    if (!container || !svg) return;
                    const svgRect = svg.getBoundingClientRect();
                    const pxX = (x / svgWidth) * svgRect.width;
                    setHoveredCol({ index: i, x: pxX, containerWidth: svgRect.width });
                  }}
                  onMouseLeave={() => setHoveredCol(null)}
                />
              );
            })}

            {/* Vertical Hover Line */}
            {hoveredCol && (
              <line
                x1={scaleX(hoveredCol.index)}
                y1={margin.top}
                x2={scaleX(hoveredCol.index)}
                y2={margin.top + chartHeight}
                stroke="rgba(255,255,255,0.2)"
                strokeWidth="1"
                strokeDasharray="4 4"
                pointerEvents="none"
              />
            )}
          </svg>
        )}

        {/* Combined Tooltip */}
        {hoveredCol && (
          <div
            className="mc-tooltip"
            style={{
              left: hoveredCol.x > hoveredCol.containerWidth / 2 ? 'auto' : hoveredCol.x + 14,
              right: hoveredCol.x > hoveredCol.containerWidth / 2 ? hoveredCol.containerWidth - hoveredCol.x + 14 : 'auto',
              top: 20,
              display: 'flex',
              flexDirection: 'column',
              gap: '4px',
              minWidth: '120px'
            }}
          >
            <div className="mc-tooltip-iter" style={{ marginBottom: '4px', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '4px' }}>
              Iter {aggregatedData[hoveredCol.index][iterationKey] ?? hoveredCol.index + 1}
            </div>
            {visibleSeries.map(s => {
              const val = aggregatedData[hoveredCol.index][s.key] ?? 0;
              return (
                <div key={s.key} className="mc-tooltip-val" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: s.color }}>{s.label}</span>
                  <span style={{ color: '#fff', marginLeft: '12px', fontFamily: "'Fira Code', monospace" }}>{val.toFixed(2)}</span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

export default MetricChart;
