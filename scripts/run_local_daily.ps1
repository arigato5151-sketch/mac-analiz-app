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
$PipelineExitCode = 0
$PipelineFailureMessage = ''

Set-Location $ProjectRoot
Start-Transcript -Path $LogPath -Append | Out-Null

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    Write-Host "`n[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')] START $Label"
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')] DONE  $Label"
}

function Send-CompletionStatus {
    param([Parameter(Mandatory = $true)][ValidateSet('success', 'failure')][string]$Status)

    $StatusArguments = @('-m', 'notifications.local_run_status', '--status', $Status, '--mode', $Mode)
    if ($PipelineFailureMessage) {
        $StatusArguments += @('--message', $PipelineFailureMessage)
    }
    try {
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')] Sending Telegram completion status ($Status)."
        & $Python @StatusArguments
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Telegram completion notification failed with exit code $LASTEXITCODE."
        }
    }
    catch {
        Write-Host "Telegram completion notification failed: $($_.Exception.Message)"
    }
}

try {
    $HasMutex = $Mutex.WaitOne(0)
    if (-not $HasMutex) {
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')] Another local pipeline is already running; skipping this launch."
        $PipelineFailureMessage = 'Another local pipeline is already running; retry this scheduled launch.'
        $PipelineExitCode = 1
    }
    if ($HasMutex -and -not (Test-Path -LiteralPath $Python)) {
        throw "Project virtual environment is missing: $Python"
    }
    if ($HasMutex -and -not (Test-Path -LiteralPath (Join-Path $ProjectRoot '.env'))) {
        throw "Project .env file is missing: $ProjectRoot\.env"
    }

    if ($HasMutex -and $OddsRefreshOnly) {
        Invoke-PythonStep 'Nesine daily fixtures and odds' @('-m', 'data_pipeline.nesine_odds', '--once-daily')
        Invoke-PythonStep 'Refresh predictions using latest odds' @('-m', 'models.predict', '--days', '1')
        Invoke-PythonStep 'Send eligible daily coupon' @('-m', 'notifications.daily_coupon')
    }
    elseif ($HasMutex) {
        Invoke-PythonStep 'Nesine daily fixtures and odds' @('-m', 'data_pipeline.nesine_odds')
        Invoke-PythonStep 'Sync API-Football fixtures' @('-m', 'data_pipeline.fetch_fixtures', '--days', '1')
        Invoke-PythonStep 'Sync daily bookmaker odds' @('-m', 'data_pipeline.sync_odds', '--days', '1', '--max-matches', '120')
        Invoke-PythonStep 'Refresh daily team context' @('-m', 'data_pipeline.refresh_context', '--days', '1')
        Invoke-PythonStep 'Generate daily predictions' @('-m', 'models.predict', '--days', '1')
        Invoke-PythonStep 'Send eligible daily coupon' @('-m', 'notifications.daily_coupon')
        Invoke-PythonStep 'Validate daily data quality' @('-m', 'data_pipeline.data_quality', '--days', '1')
    }

    if ($HasMutex) {
        Write-Host "`n[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')] PIPELINE SUCCESS"
        Send-CompletionStatus 'success'
    }
}
catch {
    $PipelineFailureMessage = $_.Exception.Message
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')] PIPELINE FAILED: $PipelineFailureMessage" -ForegroundColor Red
    $PipelineExitCode = 1
    Send-CompletionStatus 'failure'
}
finally {
    if ($HasMutex) {
        $Mutex.ReleaseMutex()
    }
    $Mutex.Dispose()
    Stop-Transcript | Out-Null
}

if ($PipelineExitCode -eq 1) {
    exit 1
}
