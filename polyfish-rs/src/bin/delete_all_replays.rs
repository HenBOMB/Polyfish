use serde_json::Value;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();

    let supabase_url = env::var("SUPABASE_URL").expect("SUPABASE_URL not set");
    let supabase_key = env::var("SUPABASE_SERVICE_ROLE_KEY")
        .or_else(|_| env::var("SUPABASE_PUBLIC_ANON_KEY"))
        .expect("Supabase key not set");

    let bucket_name = env::var("SUPABASE_STORAGE_BUCKET").unwrap_or_else(|_| "games".to_string());

    let client = reqwest::Client::new();

    println!(
        "⚠️ WARNING: This will delete ALL data from the 'games' table AND empty the '{}' bucket.",
        bucket_name
    );
    println!("Waiting 3 seconds before proceeding...");
    tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;

    // --- 1. Empty Storage Bucket ---
    println!("\n🗑️  Emptying storage bucket '{}'...", bucket_name);
    let mut files_deleted = 0;

    loop {
        // List up to 1000 files
        let list_url = format!(
            "{}/storage/v1/object/list/{}",
            supabase_url.trim_end_matches('/'),
            bucket_name
        );

        let res = client
            .post(&list_url)
            .header("apikey", &supabase_key)
            .header("Authorization", format!("Bearer {}", supabase_key))
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "limit": 1000,
                "offset": 0
            }))
            .send()
            .await?;

        if !res.status().is_success() {
            eprintln!("❌ Failed to list files in bucket: {}", res.text().await?);
            break;
        }

        let files: Vec<Value> = res.json().await?;
        if files.is_empty() {
            println!("✅ Bucket is empty.");
            break;
        }

        // Extract filenames, ignoring folders like `.emptyFolderPlaceholder`
        let filenames: Vec<String> = files
            .into_iter()
            .filter_map(|f| f["name"].as_str().map(|s| s.to_string()))
            .filter(|name| !name.is_empty())
            .collect();

        if filenames.is_empty() {
            break;
        }

        let delete_url = format!(
            "{}/storage/v1/object/{}",
            supabase_url.trim_end_matches('/'),
            bucket_name
        );

        // Use bulk delete API
        let del_res = client
            .delete(&delete_url)
            .header("apikey", &supabase_key)
            .header("Authorization", format!("Bearer {}", supabase_key))
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "prefixes": filenames
            }))
            .send()
            .await?;

        if del_res.status().is_success() {
            let count = filenames.len();
            files_deleted += count;
            println!("Deleted {} files... (Total: {})", count, files_deleted);
        } else {
            eprintln!(
                "❌ Failed to delete files from bucket: {}",
                del_res.text().await?
            );
            break;
        }
    }

    // --- 2. Clear Database Table ---
    println!("\n🗑️  Deleting all rows from 'games' table...");
    // We can use a DELETE targeting all records where id is not null
    let table_url = format!(
        "{}/rest/v1/games?id=not.is.null",
        supabase_url.trim_end_matches('/')
    );

    let res = client
        .delete(&table_url)
        .header("apikey", &supabase_key)
        .header("Authorization", format!("Bearer {}", supabase_key))
        .send()
        .await?;

    if res.status().is_success() {
        println!("✅ Completely wiped all rows from 'games' table.");
    } else {
        eprintln!("❌ Failed to delete rows from table: {}", res.text().await?);
    }

    println!("\n🎉 Wipe complete. You are ready to re-scrape everything!");

    Ok(())
}
