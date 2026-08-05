use serde_json::Value;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();

    let supabase_url = env::var("SUPABASE_URL").expect("SUPABASE_URL not set");
    let supabase_key = env::var("SUPABASE_SERVICE_ROLE_KEY")
        .or_else(|_| env::var("SUPABASE_PUBLIC_ANON_KEY"))
        .expect("Supabase key not set");
    let bucket_name = env::var("SUPABASE_STORAGE_BUCKET").unwrap_or_else(|_| "replays".to_string());

    let client = reqwest::Client::new();

    println!("🔍 Fetching all games from database...");
    let db_url = format!(
        "{}/rest/v1/games?select=id,game_name,storage_path,verified",
        supabase_url.trim_end_matches('/')
    );
    let res = client
        .get(&db_url)
        .header("apikey", &supabase_key)
        .header("Authorization", format!("Bearer {}", supabase_key))
        .send()
        .await?;

    if !res.status().is_success() {
        eprintln!("❌ Failed to fetch games: {}", res.text().await?);
        return Ok(());
    }

    let games: Vec<Value> = res.json().await?;
    println!("✅ Found {} games in database.", games.len());

    println!(
        "🔍 Fetching all files from storage bucket '{}'...",
        bucket_name
    );
    let list_url = format!(
        "{}/storage/v1/object/list/{}",
        supabase_url.trim_end_matches('/'),
        bucket_name
    );
    let list_res = client
        .post(&list_url)
        .header("apikey", &supabase_key)
        .header("Authorization", format!("Bearer {}", supabase_key))
        .header("Content-Type", "application/json")
        .json(&serde_json::json!({
            "prefix": "",
            "limit": 1000,
            "offset": 0,
            "sortBy": { "column": "name", "order": "asc" }
        }))
        .send()
        .await?;

    if !list_res.status().is_success() {
        eprintln!(
            "❌ Failed to list storage files: {}",
            list_res.text().await?
        );
        return Ok(());
    }

    let storage_files: Vec<Value> = list_res.json().await?;
    let storage_names: std::collections::HashSet<String> = storage_files
        .iter()
        .filter_map(|f| f["name"].as_str())
        .map(|s| s.to_string())
        .collect();

    println!("✅ Found {} files in storage.", storage_names.len());

    let mut missing_count = 0;
    let mut verified_count = 0;
    let mut failed_replay_count = 0;

    for game in &games {
        let game_id = game["id"].as_str().unwrap_or("0");
        let game_name = game["game_name"].as_str().unwrap_or("Unknown");
        let storage_path = game["storage_path"].as_str().unwrap_or("");
        let already_verified = game["verified"].as_bool().unwrap_or(false);

        if already_verified {
            println!("✅ Game {} is already verified.", game_name);
            continue;
        }

        if storage_names.contains(storage_path) {
            println!(
                "💾 Downloading game {}: {} (path: {})",
                game_id, game_name, storage_path
            );

            // Download the file
            let download_url = format!(
                "{}/storage/v1/object/authenticated/{}/{}",
                supabase_url.trim_end_matches('/'),
                bucket_name,
                storage_path
            );

            let download_res = client
                .get(&download_url)
                .header("apikey", &supabase_key)
                .header("Authorization", format!("Bearer {}", supabase_key))
                .send()
                .await?;

            if !download_res.status().is_success() {
                eprintln!(
                    "❌ Failed to download game {}: {}",
                    game_id,
                    download_res.text().await?
                );
                missing_count += 1;
                continue;
            }

            let replay_json: Value = download_res.json().await?;

            // Parse the canonical versioned replay.
            let replay: polyfish::replay::Replay = match serde_json::from_value(replay_json) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("❌ Failed to parse replay {}: {}", game_name, e);
                    failed_replay_count += 1;
                    continue;
                }
            };

            // Replay the game
            match polyfish::replay::ReplayExecutor::execute(&replay) {
                Ok(_) => {
                    println!("✨ Replay SUCCESS for game {}: {}", game_id, game_name);

                    // Update verified to true
                    let update_url = format!(
                        "{}/rest/v1/games?id=eq.{}",
                        supabase_url.trim_end_matches('/'),
                        game_id
                    );
                    let update_res = client
                        .patch(&update_url)
                        .header("apikey", &supabase_key)
                        .header("Authorization", format!("Bearer {}", supabase_key))
                        .header("Content-Type", "application/json")
                        .header("Prefer", "return=minimal")
                        .json(&serde_json::json!({ "verified": true }))
                        .send()
                        .await?;

                    if update_res.status().is_success() {
                        verified_count += 1;
                    } else {
                        eprintln!(
                            "❌ Failed to update verification for game {}: {}",
                            game_id,
                            update_res.text().await?
                        );
                    }
                }
                Err(e) => {
                    eprintln!(
                        "⚠️ Replay FAILED for game {}: {} -> {}",
                        game_id, game_name, e
                    );
                    failed_replay_count += 1;
                }
            }
        } else {
            println!(
                "⚠️ Missing file for game {}: {} (path: {})",
                game_id, game_name, storage_path
            );
            missing_count += 1;
        }
    }

    println!("\n--- Summary ---");
    println!("Total Games in DB: {}", games.len());
    println!("Successfully Verified: {}", verified_count);
    println!("Failed Replays:       {}", failed_replay_count);
    println!("Missing Files:        {}", missing_count);
    println!("----------------");

    Ok(())
}
