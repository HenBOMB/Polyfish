# polyfish-ui

React + TypeScript + Vite dashboard for watching Polyfish training runs. Polls the Rust
`polyfish` server at `http://localhost:3000` (`/api/runs`, `/api/training-metrics`,
`/api/moves-by-turn`, `/api/value-distribution`) and renders live charts.

## Dev

```bash
npm install
npm run dev      # Vite dev server with HMR
npm run build    # production build
```

Notes:
- The training server must be running (`./run-server.sh` from the repo root) for the dashboard to have data.
- `config.json`'s `iterations` field is decorative; filter metrics by `run_id`.
