const { createClient } = require('@supabase/supabase-js');
require('dotenv').config({ path: './polyfish-rs/.env' });
const { exec } = require('child_process');

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
const TELEGRAM_CHAT_ID = process.env.TELEGRAM_CHAT_ID;

if (!SUPABASE_URL || !SUPABASE_KEY) {
    console.error("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in ./polyfish-rs/.env");
    process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

async function sendTelegramUpdate(text) {
    if (!TELEGRAM_CHAT_ID) {
        console.error("No TELEGRAM_CHAT_ID set in .env! Cannot send message.");
        return;
    }

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
        console.log("Triggering Antigravity CLI (agy) for analysis...");

        const prompt = `Analyze this new training update from the database: ${JSON.stringify(payloadData)}. Keep it concise. Provide a short summary of how the training is going based on these metrics.`;

        const child = exec(`agy -p "${prompt}"`, async (error, stdout, stderr) => {
            console.log("agy stdout:", stdout);
            if (stderr) console.error("agy stderr:", stderr);

            if (error) {
                console.error(`Error running agy: ${error.message}`);
                await sendTelegramUpdate(`⚠️ Error generating agent insight: ${error.message}\n\nRaw Data:\n${JSON.stringify(payloadData, null, 2)}`);
                return reject(error);
            }

            const reportMsg = `🤖 *AGY Training Insight*\n\n${stdout}`;
            await sendTelegramUpdate(reportMsg);
            resolve();
        });

        child.stdout.pipe(process.stdout);
        child.stderr.pipe(process.stderr);
    });
}

const TABLE_NAME = 'training_metrics';

console.log(`Starting Training Agent. Listening to Supabase Realtime table: ${TABLE_NAME}`);

supabase
    .channel('public:' + TABLE_NAME)
    .on('postgres_changes', { event: 'INSERT', schema: 'public', table: TABLE_NAME }, async (payload) => {
        console.log('New update received:', payload.new);

        if (payload.new.iteration && payload.new.iteration % 100 === 0) {
            await sendTelegramUpdate(`🚀 *New Milestone Reached: Iteration ${payload.new.iteration}*\nTriggering analysis...`);
            await runAgyAgent(payload.new);
        } else {
            console.log(`Iteration ${payload.new.iteration} saved to Supabase, skipping agent trigger (waits for multiple of 100).`);
        }
    })
    .subscribe((status) => {
        if (status === 'SUBSCRIBED') {
            console.log('Successfully subscribed to realtime events!');
        } else {
            console.log('Subscription status:', status);
        }
    });