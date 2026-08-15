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

fn percent_encode(s: &str) -> String {
    s.bytes()
        .map(|b| {
            if b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.' || b == b'~' {
                (b as char).to_string()
            } else {
                format!("%{:02X}", b)
            }
        })
        .collect()
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

        if !path.is_file() || !path.to_string_lossy().ends_with(".replay.json") {
            continue;
        }

        let content = fs::read_to_string(&path)?;
        let replay = match polyfish::replay::load_replay(&path) {
            Ok(replay) => replay,
            Err(e) => {
                println!("Skipping {:?}, invalid canonical replay: {}", path, e);
                continue;
            }
        };
        let game_name = if replay.initial_state.settings.game_name.is_empty() {
            replay.metadata.game_id.as_deref().unwrap_or("Unknown")
        } else {
            &replay.initial_state.settings.game_name
        };
        let seed = replay.initial_state.initial_seed;
        let uuid_val = replay.metadata.game_id.clone().unwrap_or_default();

        let db_url = if !uuid_val.is_empty() {
            format!(
                "{}/rest/v1/games?uuid=eq.{}&select=id",
                supabase_url.trim_end_matches('/'),
                percent_encode(&uuid_val)
            )
        } else {
            format!(
                "{}/rest/v1/games?seed=eq.{}&game_name=eq.{}&select=id",
                supabase_url.trim_end_matches('/'),
                seed,
                percent_encode(game_name)
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

        let file_name = format!(
            "{}_{}.replay.json",
            sanitize_storage_key(game_name),
            timestamp
        );
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
