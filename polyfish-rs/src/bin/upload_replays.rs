use serde_json::Value;
use std::env;
use std::fs;

fn sanitize_storage_key(name: &str) -> String {
    let mut result = String::new();
    let mut last_was_dash = false;
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
            result.push(c.to_ascii_lowercase());
            last_was_dash = false;
        } else {
            if !last_was_dash && !result.is_empty() {
                result.push('-');
                last_was_dash = true;
            }
        }
    }
    result.trim_matches('-').to_string()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();

    let supabase_key = env::var("SUPABASE_SERVICE_ROLE_KEY")
        .or_else(|_| env::var("SUPABASE_PUBLIC_ANON_KEY"))
        .unwrap_or_default();
    let supabase_url = env::var("SUPABASE_URL").unwrap_or_default();

    if supabase_url.is_empty() || supabase_key.is_empty() {
        println!("Error: Supabase URL or Key not set in ENV.");
        return Ok(());
    }

    let bucket_name = env::var("SUPABASE_STORAGE_BUCKET").unwrap_or_else(|_| "games".to_string());
    let client = reqwest::Client::new();

    let entries = fs::read_dir("replays")?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        if !path.is_file() || !path.to_string_lossy().ends_with(".json") {
            continue;
        }

        let content = fs::read_to_string(&path)?;
        let body: Value = match serde_json::from_str(&content) {
            Ok(b) => b,
            Err(e) => {
                println!("Skipping {:?}, could not parse: {}", path, e);
                continue;
            }
        };

        // Fallback for different replay schemas you may have
        let is_mod_replay = body["turns"].is_array() && body["gameState"].is_object();
        let is_legacy_replay = body["settings"].is_object(); // direct state

        if !is_mod_replay && !is_legacy_replay {
            println!("Skipping {:?} (unrecognized replay format)", path);
            continue;
        }

        let game_state_obj = if is_mod_replay {
            &body["gameState"]
        } else {
            &body
        };

        let game_name = game_state_obj["settings"]["gameName"]
            .as_str()
            .unwrap_or("Unknown");

        let seed = game_state_obj["initial_seed"]
            .as_u64()
            .or_else(|| game_state_obj["settings"]["seed"].as_u64())
            .unwrap_or(0);

        let uuid_val = body["uuid"].as_str().unwrap_or("").to_string();

        let db_url = if !uuid_val.is_empty() {
            format!(
                "{}/rest/v1/games?uuid=eq.{}&select=id",
                supabase_url.trim_end_matches('/'),
                uuid_val
            )
        } else {
            let safe_game_name = game_name.replace(" ", "%20");
            format!(
                "{}/rest/v1/games?seed=eq.{}&game_name=eq.{}&select=id",
                supabase_url.trim_end_matches('/'),
                seed,
                safe_game_name
            )
        };

        // 1. Check if it already exists
        let check_req = client
            .get(&db_url)
            .header("apikey", &supabase_key)
            .header("Authorization", format!("Bearer {}", supabase_key))
            .send()
            .await?;

        if let Ok(json) = check_req.json::<serde_json::Value>().await {
            if let Some(arr) = json.as_array() {
                if !arr.is_empty() {
                    println!(
                        "⚠️ Rejected duplicate game (UUID or Seed/Name): {}",
                        game_name
                    );
                    continue;
                }
            }
        }

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let file_name = format!("{}_{}.json", sanitize_storage_key(game_name), timestamp);
        let storage_url = format!(
            "{}/storage/v1/object/{}/{}",
            supabase_url.trim_end_matches('/'),
            bucket_name,
            file_name
        );

        // 2. Upload to storage
        let upload_res = client
            .post(&storage_url)
            .header("apikey", &supabase_key)
            .header("Authorization", format!("Bearer {}", supabase_key))
            .header("Content-Type", "application/json")
            .body(content)
            .send()
            .await?;

        if !upload_res.status().is_success() {
            let err_text = upload_res.text().await.unwrap_or_default();
            println!(
                "❌ Supabase Storage Upload Failed for {}: {}",
                game_name, err_text
            );
            continue;
        }

        // 3. Insert record into games table
        let insert_url = format!("{}/rest/v1/games", supabase_url.trim_end_matches('/'));
        let mut insert_payload = serde_json::json!({
            "seed": seed,
            "game_name": game_name,
            "storage_path": file_name,
            "verified": false
        });
        if !uuid_val.is_empty() {
            insert_payload
                .as_object_mut()
                .unwrap()
                .insert("uuid".into(), serde_json::json!(uuid_val));
        }

        let insert_res = client
            .post(&insert_url)
            .header("apikey", &supabase_key)
            .header("Authorization", format!("Bearer {}", supabase_key))
            .header("Content-Type", "application/json")
            .header("Prefer", "return=minimal")
            .json(&insert_payload)
            .send()
            .await?;

        if !insert_res.status().is_success() {
            let err_text = insert_res.text().await.unwrap_or_default();
            println!(
                "❌ Supabase DB Insert Failed for {}: {}",
                game_name, err_text
            );
            continue;
        }

        println!("✅ Successfully uploaded {} ({})", game_name, file_name);
    }

    println!("Finished processing replays directory.");
    Ok(())
}
