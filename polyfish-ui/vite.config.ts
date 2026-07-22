import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { spawn, ChildProcess } from 'child_process'

let backendProcess: ChildProcess | null = null;
let trainingProcess: ChildProcess | null = null;

const killProcess = (proc: ChildProcess | null, label: string) => {
  if (proc && proc.pid && !proc.killed) {
    console.log(`[process-manager] Killing ${label} (PID: ${proc.pid})`);
    proc.kill('SIGTERM');
    setTimeout(() => {
      if (!proc.killed) proc.kill('SIGKILL');
    }, 3000);
  }
};

const processManagerPlugin = () => ({
  name: 'process-manager',
  configureServer(server: any) {
    // Cleanup on Vite server close (ctrl+C, HMR restart, etc.)
    server.httpServer?.on('close', () => {
      killProcess(backendProcess, 'backend');
      killProcess(trainingProcess, 'training');
      backendProcess = null;
      trainingProcess = null;
    });

    // Also cleanup on process exit
    const cleanup = () => {
      killProcess(backendProcess, 'backend');
      killProcess(trainingProcess, 'training');
    };
    process.on('exit', cleanup);
    process.on('SIGINT', () => { cleanup(); process.exit(); });
    process.on('SIGTERM', () => { cleanup(); process.exit(); });

    server.middlewares.use(async (req: any, res: any, next: any) => {
      // Serve the tempo sidecar straight from disk so the dashboard gets it
      // without needing the (possibly older) Rust server to expose it. The
      // Rust /api/tempo-by-turn endpoint exists too and takes over in prod.
      if (req.url?.startsWith('/api/tempo-by-turn') && req.method === 'GET') {
        try {
          const { readFile } = await import('fs/promises');
          const raw = await readFile('../polyfish-rs/tempo_by_turn.json', 'utf-8');
          res.setHeader('Content-Type', 'application/json');
          res.end(raw);
        } catch {
          res.setHeader('Content-Type', 'application/json');
          res.end('{}');
        }
        return;
      }
      // CSV columns newer than the running Rust server's hardcoded /metrics
      // schema — served straight from training_log.csv so charts don't need
      // a backend restart to see them.
      if (req.url?.startsWith('/api/metrics-extra') && req.method === 'GET') {
        const EXTRA = ['avg_units_spawned', 'avg_units_granted', 'avg_units_lost', 'avg_giants_made',
          't2c_2nd_rate', 't2c_2nd_turn', 't2c_3rd_rate', 't2c_3rd_turn', 't2c_4th_rate', 't2c_4th_turn'];
        try {
          const { readFile } = await import('fs/promises');
          const raw = await readFile('../polyfish-rs/training_log.csv', 'utf-8');
          const lines = raw.split('\n').filter(l => l.trim());
          const headers = lines[0].split(',');
          const idx = (name: string) => headers.indexOf(name);
          const rows = lines.slice(1).map(l => {
            const parts = l.split(',');
            const row: any = {
              run_id: parts[idx('run_id')],
              iteration: Number(parts[idx('iteration')]) || 0,
            };
            for (const col of EXTRA) {
              const i = idx(col);
              const v = i >= 0 ? Number(parts[i]) : NaN;
              if (Number.isFinite(v)) row[col] = v;
            }
            return row;
          });
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify(rows.slice(-200)));
        } catch {
          res.setHeader('Content-Type', 'application/json');
          res.end('[]');
        }
        return;
      }
      if (req.url === '/api/start-server' && req.method === 'POST') {
        if (backendProcess && !backendProcess.killed) {
          res.statusCode = 400;
          res.end(JSON.stringify({ error: 'Server already running' }));
          return;
        }
        backendProcess = spawn('./target/release/polyfish', [], {
          cwd: '../polyfish-rs',
          stdio: 'ignore',
          detached: false
        });
        backendProcess.on('exit', () => { backendProcess = null; });
        res.end(JSON.stringify({ status: 'started', pid: backendProcess.pid }));
        return;
      }
      if (req.url === '/api/start-training' && req.method === 'POST') {
        if (trainingProcess && !trainingProcess.killed) {
          res.statusCode = 400;
          res.end(JSON.stringify({ error: 'Training already running' }));
          return;
        }
        // Start training without starting the server (-n flag)
        trainingProcess = spawn('./run_training_loop.sh', ['-n'], {
          cwd: '../polyfish-rs',
          stdio: 'ignore',
          detached: false
        });
        trainingProcess.on('exit', () => { trainingProcess = null; });
        res.end(JSON.stringify({ status: 'started', pid: trainingProcess.pid }));
        return;
      }
      next();
    });
  }
});

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), processManagerPlugin()],
  server: {
    proxy: {
      '^/(current|step|simulate|replay|autostep|eval|sequence|bestmoves|trainer|save|load|save_training_data|train|api|reset|config|metrics|system|rngstep|analyze).*': {
        target: 'http://localhost:3000',
        changeOrigin: true
      }
    }
  }
})
