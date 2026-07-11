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
