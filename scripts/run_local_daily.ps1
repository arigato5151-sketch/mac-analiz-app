param(
    [switch]$OddsRefreshOnly
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$LogDirectory = Join-Path $ProjectRoot 'logs\local_scheduler'
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$Mode = if ($OddsRefreshOnly) { 'odds-refresh' } else { 'daily' }
$Timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$LogPath = Join-Path $LogDirectory "$Mode-$Timestamp.log"
$Mutex = [System.Threading.Mutex]::new($false, 'MacAnalizLocalPipeline')
$HasMutex = $false
$ExitCode = 0

Set-Location $ProjectRoot
Start-Transcript -Path $LogPath -Append | Out-Null

try {
    $HasMutex = $Mutex.WaitOne(0)
    if (-not $HasMutex) {
        Write-Host 'Another local data import is already running; skipping this launch.'
    }
    elseif (-not (Test-Path -LiteralPath $Python)) {
        throw "Project virtual environment is missing: $Python"
    }
    elseif (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot '.env'))) {
        throw "Project .env file is missing: $ProjectRoot\.env"
    }
    else {
        if (-not $OddsRefreshOnly) {
            Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')] START Refresh fixtures"
            & $Python -m data_pipeline.fetch_fixtures --days 14
            if ($LASTEXITCODE -ne 0) { throw "Fixture refresh failed with exit code $LASTEXITCODE" }
            Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')] DONE  Refresh fixtures"

            Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')] START Refresh completed match results"
            & $Python -m data_pipeline.fetch_results --lookback-days 7
            if ($LASTEXITCODE -ne 0) { throw "Results refresh failed with exit code $LASTEXITCODE" }
            Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')] DONE  Refresh completed match results"
        }

        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')] START Import IdDaa fixtures and markets"
        & $Python -m data_pipeline.fetch_iddaa_odds --retries 3
        if ($LASTEXITCODE -ne 0) { throw "IdDaa odds import failed with exit code $LASTEXITCODE" }
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')] DONE  Import IdDaa fixtures and markets"
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')] PIPELINE SUCCESS"
    }
}
catch {
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')] PIPELINE FAILED: $($_.Exception.Message)" -ForegroundColor Red
    $ExitCode = 1
}
finally {
    if ($HasMutex) { $Mutex.ReleaseMutex() }
    $Mutex.Dispose()
    Stop-Transcript | Out-Null
}

if ($ExitCode -ne 0) { exit $ExitCode }
