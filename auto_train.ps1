# Configuration
# Replace this with the actual path to your training script for Windows
$TrainCmd = ".\run-training.bat" 
$CheckInterval = 1 # seconds

# Embed a little bit of C# to call the native Windows API (GetLastInputInfo) 
# to flawlessly track global system idle time.
$IdleCode = @'
using System;
using System.Runtime.InteropServices;

public class Win32 {
    [StructLayout(LayoutKind.Sequential)]
    struct LASTINPUTINFO {
        public static readonly int SizeOf = Marshal.SizeOf(typeof(LASTINPUTINFO));
        [MarshalAs(UnmanagedType.U4)]
        public UInt32 cbSize;
        [MarshalAs(UnmanagedType.U4)]
        public UInt32 dwTime;
    }

    [DllImport("user32.dll")]
    static extern bool GetLastInputInfo(ref LASTINPUTINFO plii);

    public static uint GetIdleTime() {
        LASTINPUTINFO lastInPut = new LASTINPUTINFO();
        lastInPut.cbSize = (uint)Marshal.SizeOf(lastInPut);
        if (GetLastInputInfo(ref lastInPut)) {
            return (uint)Environment.TickCount - lastInPut.dwTime;
        }
        return 0;
    }
}
'@

Add-Type -TypeDefinition $IdleCode -Language CSharp

function IsAllowedTime {
    $day = (Get-Date).DayOfWeek
    $hour = (Get-Date).Hour

    # Weekend: All day (Saturday or Sunday)
    if ($day -eq [System.DayOfWeek]::Saturday -or $day -eq [System.DayOfWeek]::Sunday) {
        return $true
    }

    # Weekday: 20:00 (8 PM) to 08:00 (8 AM)
    if ($hour -ge 20 -or $hour -lt 8) {
        return $true
    }

    return $false
}

$TrainProcess = $null
$LastReportDay = ""

Write-Host "Starting auto-train monitor for Windows..." -ForegroundColor Cyan
Write-Host "Training allowed: Weekdays 20:00-08:00, Weekends all day."
Write-Host "Requires 60s of mouse/keyboard inactivity."
Write-Host "Checking every $CheckInterval seconds..."

while ($true) {
    $CurrentHour = (Get-Date).Hour
    $CurrentDay = (Get-Date).ToString("yyyy-MM-dd")

    # AGY daily report at 8 PM
    if ($CurrentHour -eq 20 -and $LastReportDay -ne $CurrentDay) {
        Write-Host "$(Get-Date): Triggering daily AGY training evaluation report..." -ForegroundColor Yellow
        
        $AgyPrompt = "Perform a rigorous evaluation of the Polyfish/PolyZero training data. Please follow these exact steps:`n1. Parse 'session.log' and 'training_log.csv' in the current directory to extract the latest training metrics.`n2. Compare Policy Loss vs. Value Loss. If Value Loss is near zero (e.g., ~0.03) while Policy Loss remains high, explicitly flag that the 'value head has collapsed' and the model is producing near-random value gradients.`n3. Calculate the timeout rate. Search 'session.log' for 'Decisive: true' vs 'Decisive: false'. If the vast majority of games are timing out instead of ending decisively, flag that the model is failing to learn long-term planning and closing strategies.`n4. Read 'src/bin/self_play.rs' to identify the current curriculum phase (e.g., Tiny maps progressing from 10 to 30 max turns). Account for these phases when analyzing score trends, as map size and turn limit increases artificially inflate scores.`nWrite a comprehensive Markdown report summarizing the health of the training run based on these specific metrics."

        # Start agy in the background and redirect output to a file
        Start-Process -FilePath "agy" -ArgumentList "`"$AgyPrompt`"" -RedirectStandardOutput "agy_report_$CurrentDay.txt" -RedirectStandardError "agy_report_$CurrentDay.txt" -NoNewWindow
        
        $LastReportDay = $CurrentDay
    }

    # Get system idle time in milliseconds and convert to seconds
    $IdleMs = [Win32]::GetIdleTime()
    $IdleSecs = [math]::Floor($IdleMs / 1000)
    
    #Write-Host "Current Idle: $IdleSecs seconds" -ForegroundColor DarkGray

    # Check if there was activity (idle less than 60s)
    if ($IdleSecs -lt 60) {
        if ($TrainProcess -ne $null -and !$TrainProcess.HasExited) {
            Write-Host "$(Get-Date): Activity detected. Halting training script (PID: $($TrainProcess.Id))." -ForegroundColor Red
            
            # taskkill /T /F gracefully kills the entire process tree on Windows
            Start-Process -FilePath "taskkill" -ArgumentList "/T /F /PID $($TrainProcess.Id)" -NoNewWindow -Wait
            
            $TrainProcess = $null
        }
    } else {
        # Idle for >= 60s, check if current time is allowed
        if (IsAllowedTime) {
            if ($TrainProcess -eq $null -or $TrainProcess.HasExited) {
                Write-Host "$(Get-Date): Conditions met (Idle for ${IdleSecs}s). Starting training script..." -ForegroundColor Green
                
                # Start the training script asynchronously
                $TrainProcess = Start-Process -FilePath "cmd.exe" -ArgumentList "/c $TrainCmd" -PassThru -NoNewWindow
                
                Write-Host "Training script started with PID: $($TrainProcess.Id)" -ForegroundColor DarkGray
            }
        } else {
            # Outside of the allowed time window
            if ($TrainProcess -ne $null -and !$TrainProcess.HasExited) {
                Write-Host "$(Get-Date): Time window ended. Halting training script (PID: $($TrainProcess.Id))." -ForegroundColor Red
                
                Start-Process -FilePath "taskkill" -ArgumentList "/T /F /PID $($TrainProcess.Id)" -NoNewWindow -Wait
                
                $TrainProcess = $null
            }
        }
    }

    Start-Sleep -Seconds $CheckInterval
}
