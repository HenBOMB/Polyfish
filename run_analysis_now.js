const fs = require('fs');
const { exec } = require('child_process');
require('dotenv').config({ path: './polyfish-rs/.env' });

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
const TELEGRAM_CHAT_ID = process.env.TELEGRAM_CHAT_ID;

if (!SUPABASE_URL || !SUPABASE_KEY || !TELEGRAM_CHAT_ID) {
  console.error("Missing SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, or TELEGRAM_CHAT_ID in ./polyfish-rs/.env");
  process.exit(1);
}

async function sendTelegramUpdate(text) {
  try {
    const response = await fetch(`${SUPABASE_URL}/functions/v1/telegram-bot/send`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${SUPABASE_KEY}`
      },
      body: JSON.stringify({
        chat_id: TELEGRAM_CHAT_ID,
        text: text,
      })
    });

    if (response.ok) {
      console.log("Sent update to Telegram successfully.");
    } else {
      console.error("Failed to send telegram update:", await response.text());
    }
  } catch (e) {
    console.error("Exception while sending telegram update:", e);
  }
}

function runAgyAgent(payloadData) {
  return new Promise((resolve, reject) => {
    console.log("Triggering Antigravity CLI (agy) for immediate analysis...");

    const prompt = `Analyze this new training update from the database: ${JSON.stringify(payloadData)}. Keep it concise. Provide a short summary of how the training is going based on these metrics. Dashboard: http://localhost:3000/training.html`;

    const child = exec(`agy -p "${prompt}"`, async (error, stdout, stderr) => {
      console.log("agy stdout:", stdout);
      if (stderr) console.error("agy stderr:", stderr);

      if (error) {
        console.error(`Error running agy: ${error.message}`);
        await sendTelegramUpdate(`⚠️ Error generating agent insight: ${error.message}\n\nRaw Data:\n${JSON.stringify(payloadData, null, 2)}`);
        return reject(error);
      }

      const reportMsg = `🤖 *AGY On-Demand Training Insight*\n\n${stdout}\n\nhttp://localhost:3000/training.html`;
      await sendTelegramUpdate(reportMsg);
      resolve();
    });

    child.stdout.pipe(process.stdout);
    child.stderr.pipe(process.stderr);
  });
}

function parseCsvRow(line, headers) {
  const cols = line.split(',');
  const row = {};
  headers.forEach((h, i) => {
    const v = cols[i] ?? '';
    if (['iteration', 'run_id'].includes(h)) row[h] = parseInt(v, 10);
    else if (h === 'iter_started_at' || h === 'run_started_at' || h === 'games_file' || h === 'match_type') row[h] = v;
    else row[h] = parseFloat(v);
  });
  return row;
}

async function main() {
  const csvPath = './polyfish-rs/training_log.csv';

  if (!fs.existsSync(csvPath)) {
    console.error("No training_log.csv found at " + csvPath);
    process.exit(1);
  }

  const content = fs.readFileSync(csvPath, 'utf8');
  const lines = content.trim().split('\n').filter(l => l.length > 0);

  if (lines.length < 2) {
    console.error("The CSV file has no data rows.");
    process.exit(1);
  }

  const headers = lines[0].split(',');
  const lastLine = lines[lines.length - 1];
  const data = parseCsvRow(lastLine, headers);

  console.log("Extracted latest metrics:", data);
  const runLabel = data.iter_started_at || data.run_started_at || data.run_id;
  await sendTelegramUpdate(`🚀 **On-Demand Analysis — Run ${runLabel}, Iteration ${data.iteration}**\nRunning AGY analysis…\nhttp://localhost:3000/training.html`);
  await runAgyAgent(data);
}

main().catch(console.error);
