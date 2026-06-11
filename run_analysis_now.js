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

    const prompt = `Analyze this new training update from the database: ${JSON.stringify(payloadData)}. Keep it concise. Provide a short summary of how the training is going based on these metrics.`;

    const child = exec(`agy -p "${prompt}"`, async (error, stdout, stderr) => {
      console.log("agy stdout:", stdout);
      if (stderr) console.error("agy stderr:", stderr);

      if (error) {
        console.error(`Error running agy: ${error.message}`);
        await sendTelegramUpdate(`⚠️ Error generating agent insight: ${error.message}\n\nRaw Data:\n${JSON.stringify(payloadData, null, 2)}`);
        return reject(error);
      }

      const reportMsg = `🤖 *AGY On-Demand Training Insight*\n\n${stdout}`;
      await sendTelegramUpdate(reportMsg);
      resolve();
    });

    child.stdout.pipe(process.stdout);
    child.stderr.pipe(process.stderr);
  });
}

async function main() {
  const csvPath = './polyfish-rs/training_log.csv';

  if (!fs.existsSync(csvPath)) {
    console.error("No training_log.csv found at " + csvPath);
    process.exit(1);
  }

  const content = fs.readFileSync(csvPath, 'utf8');
  const lines = content.trim().split('\n').filter(l => l.length > 0);

  if (lines.length === 0) {
    console.error("The CSV file is empty.");
    process.exit(1);
  }

  const lastLine = lines[lines.length - 1];
  const columns = lastLine.split(',');

  // Format: $i,$Timestamp,$AvgScore,$MaxScore,$P1Avg,$P2Avg,$Loss,$AvgCaptures,$AvgHarvests,$AvgBuilds,$AvgResearch,$AvgAttacks
  if (columns.length >= 12) {
    const data = {
      iteration: parseInt(columns[0]),
      timestamp: parseFloat(columns[1]),
      avg_score: parseFloat(columns[2]),
      max_score: parseFloat(columns[3]),
      p1_avg: parseFloat(columns[4]),
      p2_avg: parseFloat(columns[5]),
      loss: parseFloat(columns[6]),
      avg_captures: parseFloat(columns[7]),
      avg_harvests: parseFloat(columns[8]),
      avg_builds: parseFloat(columns[9]),
      avg_research: parseFloat(columns[10]),
      avg_attacks: parseFloat(columns[11])
    };

    console.log("Extracted latest metrics:", data);
    await sendTelegramUpdate(`🚀 **On-Demand Analysis Triggered: Iteration ${data.iteration}**\nRunning AGY analysis now...`);
    await runAgyAgent(data);
  } else {
    console.error("Last line doesn't have enough columns:", lastLine);
    process.exit(1);
  }
}

main().catch(console.error);