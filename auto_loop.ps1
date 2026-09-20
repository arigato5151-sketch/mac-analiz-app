param(
    [int]$MaxIterations = 5,
    [string]$PytestArgs = "-q"
)

# Otomatik dongu: pytest -> basarisizsa OpenCode'a duzeltme talebi -> tekrar.
# Testler gecene kadar insan onayı beklemeden surer (opencode --auto).

$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
$logDir = Join-Path $root ".auto_loop"
if (-not (Test-Path -LiteralPath $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

if (-not (Get-Command opencode -ErrorAction SilentlyContinue)) {
    Write-Host "!!! HATA: 'opencode' komutu bulunamadi. OpenCode CLI kurulu mu ve PATH'te mi?"
    exit 2
}

for ($i = 1; $i -le $MaxIterations; $i++) {
    Write-Host ""
    Write-Host "=== [Dongu $i/$MaxIterations] Testler calistiriliyor: python -m pytest $PytestArgs ==="

    $logFile = Join-Path $logDir ("run_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date))
    Push-Location $root
    try {
        $output = cmd /c "python -m pytest $PytestArgs 2>&1" | Tee-Object -FilePath $logFile
    }
    finally {
        Pop-Location
    }
    $exitCode = $LASTEXITCODE

    if ($exitCode -eq 0) {
        $summary = ($output | Select-Object -Last 1)
        Write-Host ""
        Write-Host ">>> TUM TESTLER BASARILI: $summary"
        Write-Host ">>> Log: $logFile"
        exit 0
    }

    Write-Host ">>> Testler basarisiz (exit kodu $exitCode). OpenCode'a duzeltme talebi gonderiliyor..."
    $prompt = "Pytest basarisiz (exit kodu $exitCode). Tam hata ciktisi su dosyada: $logFile - bu dosyayi oku, kok nedeni bul, ilgili kodu duzelt ve kaydet. Ardindan 'python -m pytest $PytestArgs' komutunu calistirip tum testlerin gectigini dogrula. Duzeltme bittiginde kisa bir ozet yaz."

    Push-Location $root
    try {
        opencode run $prompt --auto
    }
    finally {
        Pop-Location
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host ">>> UYARI: opencode run sifir olmayan kodla dondu (kod: $LASTEXITCODE). Tekrar denenecek."
    }
}

Write-Host ""
Write-Host "!!! HATA: $MaxIterations denemeden sonra testler hala basarisiz."
Write-Host "!!! Tum test loglari: $logDir"
exit 1
