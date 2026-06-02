const { createClient } = require('@supabase/supabase-js');
require('dotenv').config({ path: './polyfish-rs/.env' });
const { exec } = require('child_process');
const fetch = require('node-fetch'); // Ensure you have node-fetch installed

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

// The user must provide their Chat ID in the .env file
const TELEGRAM_CHAT_ID = process.env.TELEGRAM_CHAT_ID;

if (!SUPABASE_URL || !SUPABASE_KEY) {
  console.error("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in ./polyfish-rs/.env");
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

// Sends a message using the Daddy-Agent telegram-bot edge function you requested
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

// Function to call Antigravity CLI (-agy-) to generate insights
function runAgyAgent(payloadData) {
  console.log("Triggering Antigravity CLI (agy) for analysis...");
  
  const prompt = `Analyze this new training update from the database: ${JSON.stringify(payloadData)}. Keep it concise. Provide a short summary of how the training is going based on these metrics.`;
  
  exec(`agy "${prompt}"`, (error, stdout, stderr) => {
    if (error) {
      console.error(`Error running agy: ${error.message}`);
      sendTelegramUpdate(`⚠️ Error generating agent insight: ${error.message}\n\nRaw Data:\n${JSON.stringify(payloadData, null, 2)}`);
      return;
    }
    if (stderr) {
      console.warn(`agy stderr: ${stderr}`);
    }
    
    // Send the generated report back to Telegram
    const reportMsg = `🤖 *AGY Training Insight*\n\n${stdout}`;
    sendTelegramUpdate(reportMsg);
  });
}

const TABLE_NAME = 'training_metrics';

console.log(`Starting Training Agent. Listening to Supabase Realtime table: ${TABLE_NAME}`);

supabase
  .channel('public:' + TABLE_NAME)
  .on('postgres_changes', { event: 'INSERT', schema: 'public', table: TABLE_NAME }, payload => {
    console.log('New update received:', payload.new);
    
    // Only trigger the agent every 100 iterations
    if (payload.new.iteration && payload.new.iteration % 100 === 0) {
      // Send raw notification immediately
      sendTelegramUpdate(`🚀 *New Milestone Reached: Iteration ${payload.new.iteration}*\nTriggering analysis...`);
      
      // Trigger AGY to process it and send an intelligent response
      runAgyAgent(payload.new);
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
