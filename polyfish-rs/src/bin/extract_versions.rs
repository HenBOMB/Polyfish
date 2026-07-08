use clap::Parser;
use regex::Regex;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;
use std::collections::HashSet;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
/// A utility to recursively extract version numbers from files.
struct Args {
    /// Directory to scan
    #[arg(default_value = ".")]
    path: String,

    /// Regex pattern to find versions. Group 1 is captured as the version string.
    #[arg(short, long, default_value = r#"(?i)version\s*[:=]\s*["']?([0-9]+\.[0-9]+(?:\.[0-9]+)?(?:[a-zA-Z0-9.\-]+)?)["']?"#)]
    regex: String,

    /// Show files even if no version is found
    #[arg(short, long)]
    all: bool,

    /// Only show the first version found per file
    #[arg(short, long)]
    first: bool,
}

fn main() {
    let args = Args::parse();
    let root = Path::new(&args.path);
    
    let re = match Regex::new(&args.regex) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error: Invalid regex pattern: {}", e);
            std::process::exit(1);
        }
    };

    println!("🔍 Scanning {} for versions...", root.display());
    println!("--------------------------------------------------");

    let mut files_checked = 0;
    let mut files_with_version = 0;

    for entry in WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let path = entry.path();
        
        // Skip hidden directories (like .git)
        if path.components().any(|c| c.as_os_str().to_string_lossy().starts_with('.')) && !args.path.starts_with('.') {
            continue;
        }

        files_checked += 1;

        // Try reading as string. If it's binary, this will likely fail or return garbage.
        // We handle errors by skipping.
        if let Ok(content) = fs::read_to_string(path) {
            let mut versions = HashSet::new();
            
            for caps in re.captures_iter(&content) {
                let version = if let Some(m) = caps.get(1) {
                    m.as_str().to_string()
                } else {
                    caps.get(0).unwrap().as_str().to_string()
                };
                versions.insert(version);
                if args.first {
                    break;
                }
            }

            if !versions.is_empty() {
                files_with_version += 1;
                let mut v_list: Vec<_> = versions.into_iter().collect();
                v_list.sort();
                println!("{}: {}", path.display(), v_list.join(", "));
            } else if args.all {
                println!("{}: [no version found]", path.display());
            }
        }
    }

    println!("--------------------------------------------------");
    println!("✅ Done. Checked {} files, found versions in {}.", files_checked, files_with_version);
}
