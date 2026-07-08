use std::env;
use std::fs;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();

    let supabase_url = env::var("SUPABASE_URL").expect("SUPABASE_URL not set");
    let supabase_key = env::var("SUPABASE_SERVICE_ROLE_KEY")
        .or_else(|_| env::var("SUPABASE_PUBLIC_ANON_KEY"))
        .expect("Supabase key not set");
    let bucket_name = env::var("SUPABASE_STORAGE_BUCKET").expect("STORAGE_BUCKET not set");

    let file_to_download = "kimeusian-monsoon_1774234919.json";
    let client = reqwest::Client::new();

    let url = format!(
        "{}/storage/v1/object/authenticated/{}/{}",
        supabase_url.trim_end_matches('/'),
        bucket_name,
        file_to_download
    );

    println!("📥 Downloading {}...", file_to_download);
    let res = client
        .get(&url)
        .header("apikey", &supabase_key)
        .header("Authorization", format!("Bearer {}", supabase_key))
        .send()
        .await?;

    if res.status().is_success() {
        let content = res.text().await?;
        let _ = fs::create_dir_all("replays");
        fs::write(format!("replays/{}", file_to_download), content)?;
        println!("✅ Downloaded to replays/{}", file_to_download);
    } else {
        eprintln!("❌ Failed to download: {}", res.text().await?);
    }

    Ok(())
}
