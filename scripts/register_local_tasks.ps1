$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Runner = Join-Path $PSScriptRoot 'run_local_daily.ps1'
if (-not (Test-Path -LiteralPath $Runner)) {
    throw "Local pipeline runner not found: $Runner"
}
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot '.venv\Scripts\python.exe'))) {
    throw 'The project virtual environment is missing. Create .venv first.'
}
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot '.env'))) {
    throw 'The project .env file is missing. Configure API/database/Telegram secrets first.'
}

$PowerShell = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
$DailyAction = New-ScheduledTaskAction -Execute $PowerShell -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$Runner`"" -WorkingDirectory $ProjectRoot
$RetryAction = New-ScheduledTaskAction -Execute $PowerShell -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$Runner`" -OddsRefreshOnly" -WorkingDirectory $ProjectRoot
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 2) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$Principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited

$DailyTrigger = New-ScheduledTaskTrigger -Daily -At '06:17'
Register-ScheduledTask -TaskName 'MacAnaliz-GunlukAkis' -TaskPath '\' -Action $DailyAction -Trigger $DailyTrigger -Settings $Settings -Principal $Principal -Description 'Nesine günlük fikstür/oranları, API-Football verileri, tahminler ve uygun günlük kupon bildirimi.' -Force | Out-Null

$RetryTimes = @('08:17', '10:17', '12:17', '14:17', '16:17', '18:17', '20:17', '22:17')
$RetryTriggers = @($RetryTimes | ForEach-Object { New-ScheduledTaskTrigger -Daily -At $_ })
Register-ScheduledTask -TaskName 'MacAnaliz-NesineOranYenileme' -TaskPath '\' -Action $RetryAction -Trigger $RetryTriggers -Settings $Settings -Principal $Principal -Description 'Nesine güncel oranlarını alır, tahminleri yeniler ve gün içinde henüz gönderilmediyse uygun kuponu iletir.' -Force | Out-Null

Get-ScheduledTask -TaskPath '\' | Where-Object TaskName -like 'MacAnaliz-*' | Select-Object TaskName, State, TaskPath
