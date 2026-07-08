use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();

    let supabase_url = env::var("SUPABASE_URL").expect("SUPABASE_URL not set");
    let supabase_key = env::var("SUPABASE_SERVICE_ROLE_KEY")
        .or_else(|_| env::var("SUPABASE_PUBLIC_ANON_KEY"))
        .expect("Supabase key not set");

    let client = reqwest::Client::new();
    let url = format!(
        "{}/rest/v1/games?id=not.is.null",
        supabase_url.trim_end_matches('/')
    );

    println!("🧹 Unverifying all games (resetting to false)...");
    let res = client
        .patch(&url)
        .header("apikey", &supabase_key)
        .header("Authorization", format!("Bearer {}", supabase_key))
        .header("Content-Type", "application/json")
        .json(&serde_json::json!({ "verified": false }))
        .send()
        .await?;

    if res.status().is_success() {
        println!("✅ All games unverified.");
    } else {
        eprintln!("❌ Failed to unverify games: {}", res.text().await?);
    }

    Ok(())
}
