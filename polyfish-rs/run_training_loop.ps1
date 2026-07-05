param (
    [switch]$ForceTrain,
    [switch]$Boost,
    [switch]$Chill,
    [switch]$RewardShaping,
    [switch]$Resume,
    [string]$ResumeRunId = "latest"
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

# Migrate CSV and resolve run (new run by default; -Resume to continue)
python training_log.py migrate
if ($Resume) {
    $RunInfoJson = python training_log.py resolve-run --resume $ResumeRunId
} else {
    $RunInfoJson = python training_log.py resolve-run
}
$RunInfo = $RunInfoJson | ConvertFrom-Json
$RunId = $RunInfo.run_id
$RunStartedAt = $RunInfo.run_started_at
$StartIter = [int]$RunInfo.start_iter
Write-Host "Training run_id=$RunId started_at=$RunStartedAt starting at iteration $StartIter" -ForegroundColor Green

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
    $MatchType = "selfplay"
    
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
                $MatchType = "league"
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

    $SpTemp = New-TemporaryFile
    Set-Content -Path $SpTemp -Value $SpOutput -Encoding utf8
    $GameJson = python training_log.py parse-self-play --input $SpTemp | ConvertFrom-Json
    Remove-Item $SpTemp -Force
    $GamesFile = $GameJson.games_file
    
    # 2. Training
    Write-Host "[Training] Training model..." -ForegroundColor Cyan
    $TrainOutput = & python train.py | Out-String
    Write-Host $TrainOutput

    $TrainTemp = New-TemporaryFile
    Set-Content -Path $TrainTemp -Value $TrainOutput -Encoding utf8
    $TrainJson = python training_log.py parse-train --input $TrainTemp | ConvertFrom-Json
    Remove-Item $TrainTemp -Force
    $Loss = $TrainJson.loss
    $PolicyLoss = $TrainJson.policy_loss
    $ValueLoss = $TrainJson.value_loss
    
    # 3. Log
    $Timestamp = [int][double]::Parse((Get-Date -UFormat %s))
    $GameJsonStr = ($GameJson | ConvertTo-Json -Compress)
    $TrainJsonStr = ($TrainJson | ConvertTo-Json -Compress)
    python training_log.py append-row --run-id $RunId --run-started-at $RunStartedAt --iteration $i --timestamp $Timestamp --games-file $GamesFile --game-json $GameJsonStr --train-json $TrainJsonStr --match-type $MatchType
    $AvgScore = $GameJson.avg_score
    $AvgCaptures = $GameJson.avg_captures
    $AvgHarvests = $GameJson.avg_harvests
    $AvgBuilds = $GameJson.avg_builds
    $AvgResearch = $GameJson.avg_research
    $AvgAttacks = $GameJson.avg_attacks
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
                    run_id = [int64]$RunId
                    run_started_at = $RunStartedAt
                    iteration = $i
                    timestamp = $Timestamp
                    games_file = if ($GamesFile) { "archive/$GamesFile" } else { "" }
                    avg_score = (ParseNum $AvgScore)
                    max_score = (ParseNum $GameJson.max_score)
                    p1_avg = (ParseNum $GameJson.p1_avg)
                    p2_avg = (ParseNum $GameJson.p2_avg)
                    loss = (ParseNum $Loss)
                    policy_loss = (ParseNum $PolicyLoss)
                    value_loss = (ParseNum $ValueLoss)
                    avg_captures = (ParseNum $AvgCaptures)
                    avg_harvests = (ParseNum $AvgHarvests)
                    avg_builds = (ParseNum $AvgBuilds)
                    avg_research = (ParseNum $AvgResearch)
                    avg_attacks = (ParseNum $AvgAttacks)
                    avg_moves = (ParseNum $GameJson.avg_moves)
                    match_type = $MatchType
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
    
    # Keep only the last 10 game files to match train.py's replay_buffer_size
    # (lowered from 30 on 2026-07-05 — see train.py's replay_buffer_size comment)
    $ArchivedGames = Get-ChildItem -Path "archive" -Filter "games_*.safetensors" | Sort-Object LastWriteTime -Descending
    if ($ArchivedGames.Count -gt 10) {
        $ArchivedGames | Select-Object -Skip 10 | Remove-Item -Force
    }
}

Stop-Transcript
