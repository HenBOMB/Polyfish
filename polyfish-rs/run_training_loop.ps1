param (
    [switch]$ForceTrain,
    [switch]$Boost,
    [switch]$Chill,
    [switch]$RewardShaping
)

$ErrorActionPreference = "Stop"

$Iterations = 1000
$GamesPerIter = 10
$UseThreads = 4
$env:MCTS_ITERS = 200
$env:RUST_BACKTRACE = 1

$LogFile = "session.log"
# Start a transcript to log all console output to the session.log file
Start-Transcript -Path $LogFile -Append

Write-Host "Building binaries..." -ForegroundColor Cyan
cargo build --bin polyfish --bin self_play --release

$RewardFlag = ""
if ($RewardShaping) {
    $RewardFlag = "--reward-shaping"
    Write-Host "🎯 Reward shaping enabled!" -ForegroundColor Yellow
}

if ($Boost) {
    $UseThreads = $UseThreads * 2
    Write-Host "🚀 Boost mode enabled! Using $UseThreads threads" -ForegroundColor Yellow
}

if ($Chill) {
    $UseThreads = 4
    Write-Host "❄️ Chill mode! Using 4 threads" -ForegroundColor Cyan
}

$env:RAYON_NUM_THREADS = $UseThreads
$env:OMP_NUM_THREADS = $UseThreads

if ($ForceTrain) {
    Write-Host "Force training flag detected! Running training immediately..." -ForegroundColor Yellow
    Write-Host "[Training] Training model..." -ForegroundColor Cyan
    # Assuming python is in PATH or using standard 'python' instead of .venv/bin/python3 for Windows
    python train.py
}

# Determine starting iteration from log
$StartIter = 1
if (Test-Path "training_log.csv") {
    $LastLine = Get-Content "training_log.csv" | Select-Object -Last 1
    if ($LastLine -match "^(\d+),") {
        $StartIter = [int]$matches[1] + 1
        Write-Host "Resuming from iteration $StartIter" -ForegroundColor Green
    }
}

# 0. Initialize & Auto-Restore Model
Write-Host "Initializing/Checking model..." -ForegroundColor Cyan
if ($StartIter -gt 1 -and -not (Test-Path "model.safetensors")) {
    $LatestCp = Get-ChildItem -Path "checkpoints" -Filter "model_checkpoint_iter*.safetensors" | Sort-Object Name | Select-Object -Last 1
    if ($LatestCp) {
        Write-Host "🔄 Resuming: Restoring latest checkpoint $($LatestCp.Name) to model.safetensors" -ForegroundColor Green
        Copy-Item $LatestCp.FullName -Destination "model.safetensors" -Force
    }
}
python init_model.py

for ($i = $StartIter; $i -le ($Iterations + $StartIter); $i++) {
    Write-Host "=================================================="
    Write-Host "Starting Iteration $i" -ForegroundColor Cyan
    Write-Host "=================================================="
    
    # 1. League Training Logic (20% chance)
    if (-not (Test-Path "checkpoints")) { New-Item -ItemType Directory -Path "checkpoints" | Out-Null }
    
    $OpponentFlag = ""
    $MatchType = "Self-Play"
    
    $RandVal = Get-Random -Minimum 1 -Maximum 101
    
    if ($RandVal -le 20 -and (Test-Path "checkpoints") -and (Get-ChildItem "checkpoints" -Filter "model_checkpoint_iter*.safetensors" | Measure-Object).Count -gt 0) {
        # SMART LEAGUE SELECTION: 50% chance 'Fresh' (latest), 50% chance 'Historical' (diverse)
        $AllCps = Get-ChildItem -Path "checkpoints" -Filter "model_checkpoint_iter*.safetensors" | Sort-Object LastWriteTime -Descending
        if ($AllCps.Count -gt 0) {
            $FreshCps = $AllCps | Select-Object -First 5
            $HistCps = $AllCps | Select-Object -Skip 5
            
            $SelectedCp = $null
            if ($HistCps.Count -gt 0 -and (Get-Random -Minimum 0 -Maximum 2) -eq 0) {
                $SelectedCp = $HistCps | Get-Random
            } else {
                $SelectedCp = $FreshCps | Get-Random
            }
            
            if ($SelectedCp) {
                $OpponentFlag = "--opponent `"$($SelectedCp.FullName)`""
                $MatchType = "League Match vs $($SelectedCp.Name)"
            }
        }
    }

    # Pick 2 random tribes for this iteration
    $TribeList = @("Imperius", "Imperius")
    $SelectedTribes = $TribeList | Get-Random -Count 2
    $Tribe1 = $SelectedTribes[0]
    $Tribe2 = $SelectedTribes[1]
    
    Write-Host "[$MatchType] Generative games... Tribes: $Tribe1 vs $Tribe2" -ForegroundColor Green
    
    # Run self_play and capture output
    $Args = @("--num-games", $GamesPerIter, "--mcts-iters", $env:MCTS_ITERS)
    if ($RewardFlag) { $Args += $RewardFlag }
    if ($OpponentFlag) { 
        # Extract the path from the flag
        $Args += "--opponent"
        $Args += $SelectedCp.FullName
    }
    $Args += @("--tribe1", $Tribe1, "--tribe2", $Tribe2, "--iteration", $i)
    
    $SpOutput = & .\target\release\self_play.exe @Args | Out-String
    Write-Host $SpOutput
    
    # Extract metrics using RegEx
    $AvgScore = "0"; $MaxScore = "0"; $P1Avg = "0"; $P2Avg = "0"
    $AvgCaptures = "0"; $AvgHarvests = "0"; $AvgBuilds = "0"; $AvgResearch = "0"; $AvgAttacks = "0"
    
    if ($SpOutput -match '"avg_score":\s*([0-9.]+)') { $AvgScore = $matches[1] }
    if ($SpOutput -match '"max_score":\s*([0-9.]+)') { $MaxScore = $matches[1] }
    if ($SpOutput -match '"p1_avg":\s*([0-9.]+)') { $P1Avg = $matches[1] }
    if ($SpOutput -match '"p2_avg":\s*([0-9.]+)') { $P2Avg = $matches[1] }
    if ($SpOutput -match '"avg_captures":\s*([0-9.]+)') { $AvgCaptures = $matches[1] }
    if ($SpOutput -match '"avg_harvests":\s*([0-9.]+)') { $AvgHarvests = $matches[1] }
    if ($SpOutput -match '"avg_builds":\s*([0-9.]+)') { $AvgBuilds = $matches[1] }
    if ($SpOutput -match '"avg_research":\s*([0-9.]+)') { $AvgResearch = $matches[1] }
    if ($SpOutput -match '"avg_attacks":\s*([0-9.]+)') { $AvgAttacks = $matches[1] }
    
    # 2. Training
    Write-Host "[Training] Training model..." -ForegroundColor Cyan
    $TrainOutput = & python train.py | Out-String
    Write-Host $TrainOutput
    
    $Loss = "0"
    if ($TrainOutput -match '"loss":\s*([0-9.]+)') { $Loss = $matches[1] }
    
    # 3. Log
    $Timestamp = [int][double]::Parse((Get-Date -UFormat %s))
    "$i,$Timestamp,$AvgScore,$MaxScore,$P1Avg,$P2Avg,$Loss,$AvgCaptures,$AvgHarvests,$AvgBuilds,$AvgResearch,$AvgAttacks" | Out-File -FilePath "training_log.csv" -Append -Encoding ascii
    Write-Host "Iteration $i complete. Type: $MatchType | Avg: $AvgScore | Loss: $Loss" -ForegroundColor Green
    Write-Host "  -> STATS/GAME: Captures: $AvgCaptures | Harvests: $AvgHarvests | Builds: $AvgBuilds | Tech: $AvgResearch | Attacks: $AvgAttacks" -ForegroundColor DarkGray
    
    # 3.5 Push to Supabase Realtime Table
    try {
        if (Test-Path ".env") {
            $EnvContent = Get-Content ".env" -Raw
            $SupabaseUrl = ""
            $SupabaseKey = ""
            if ($EnvContent -match 'SUPABASE_URL="([^"]+)"') { $SupabaseUrl = $matches[1] }
            if ($EnvContent -match 'SUPABASE_SERVICE_ROLE_KEY="([^"]+)"') { $SupabaseKey = $matches[1] }
            
            if ($SupabaseUrl -and $SupabaseKey) {
                $Headers = @{
                    "apikey" = $SupabaseKey
                    "Authorization" = "Bearer $SupabaseKey"
                    "Content-Type" = "application/json"
                    "Prefer" = "return=minimal"
                }
                # Handle possible empty or non-numeric values gracefully
                function ParseNum { param($val) if ([string]::IsNullOrWhiteSpace($val)) { return 0.0 } try { return [double]$val } catch { return 0.0 } }
                
                $Body = @{
                    iteration = $i
                    timestamp = $Timestamp
                    avg_score = (ParseNum $AvgScore)
                    max_score = (ParseNum $MaxScore)
                    p1_avg = (ParseNum $P1Avg)
                    p2_avg = (ParseNum $P2Avg)
                    loss = (ParseNum $Loss)
                    avg_captures = (ParseNum $AvgCaptures)
                    avg_harvests = (ParseNum $AvgHarvests)
                    avg_builds = (ParseNum $AvgBuilds)
                    avg_research = (ParseNum $AvgResearch)
                    avg_attacks = (ParseNum $AvgAttacks)
                } | ConvertTo-Json
                
                Invoke-RestMethod -Uri "$SupabaseUrl/rest/v1/training_metrics" -Method Post -Headers $Headers -Body $Body | Out-Null
                Write-Host "Pushed metrics to Supabase training_metrics table." -ForegroundColor Magenta
            }
        }
    } catch {
        Write-Host "Failed to push to Supabase: $_" -ForegroundColor Red
    }
    
    # 4. Checkpoint (Every 50 iterations)
    if ($i % 50 -eq 0) {
        $Ts = (Get-Date).ToString("yyyyMMdd_HHmmss")
        Write-Host "Creating checkpoint for iteration $i (Timestamp: $Ts)..." -ForegroundColor Yellow
        Copy-Item "model.safetensors" -Destination "checkpoints/model_checkpoint_iter${i}_${Ts}.safetensors" -Force
    }
    
    # Smart Pruning: Keep recent density and historical milestones
    $AllFiles = Get-ChildItem -Path "checkpoints" -Filter "model_checkpoint_iter*.safetensors" | Sort-Object LastWriteTime -Descending
    if ($AllFiles.Count -gt 0) {
        $idx = 0
        foreach ($File in $AllFiles) {
            $idx++
            $Keep = $false
            if ($File.Name -match 'iter(\d+)_') {
                $IterVal = [int]$matches[1]
                if ($idx -le 50) {
                    $Keep = $true
                } elseif ($IterVal % 100 -eq 0 -or $IterVal -eq 1) {
                    $Keep = $true
                }
            }
            if (-not $Keep) {
                Remove-Item $File.FullName -Force
            }
        }
    }

    # 4. Cleanup (Fresh Games Only)
    if (-not (Test-Path "archive")) { New-Item -ItemType Directory -Path "archive" | Out-Null }
    Get-ChildItem -Filter "games_*.safetensors" -ErrorAction SilentlyContinue | Move-Item -Destination "archive/" -Force
    
    # Keep only the last 30 game files to match Linux script and replay buffer
    $ArchivedGames = Get-ChildItem -Path "archive" -Filter "games_*.safetensors" | Sort-Object LastWriteTime -Descending
    if ($ArchivedGames.Count -gt 30) {
        $ArchivedGames | Select-Object -Skip 30 | Remove-Item -Force
    }
}

Stop-Transcript
