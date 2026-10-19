param([switch]$SkipBuild)
$ErrorActionPreference = 'Stop'
$taskRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $taskRoot
$env:PYTHONUTF8 = '1'
$taskRuntime = Join-Path $taskRoot 'runtime'
New-Item -ItemType Directory -Path $taskRuntime -Force | Out-Null
if (-not (Test-Path -LiteralPath '.env')) { Copy-Item -LiteralPath '.env.example' -Destination '.env' }
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    uv sync --frozen
    if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }
}
uv sync --frozen
if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }
if (-not (Test-Path -LiteralPath 'tools\qdrant\qdrant.exe') -or -not (Test-Path -LiteralPath 'runtime\models\bge-small-zh-v1.5\model.safetensors')) {
    & '.venv\Scripts\python.exe' 'scripts\setup_knowledge.py'
    if ($LASTEXITCODE -ne 0) { throw 'Vector resource installation failed.' }
}
if (-not (Test-Path -LiteralPath 'tools\livekit\livekit-server.exe')) {
    & '.venv\Scripts\python.exe' 'scripts\setup_resources.py'
    if ($LASTEXITCODE -ne 0) { throw 'Resource download failed.' }
}
if (-not $SkipBuild) {
    Push-Location -LiteralPath 'frontend'
    try {
        if (-not (Test-Path -LiteralPath 'node_modules')) { npm ci --no-audit --no-fund }
        npm run build
        if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
    } finally { Pop-Location }
}
$taskProcesses = @()
$taskPidFile = Join-Path $taskRuntime 'processes.json'
if (Test-Path -LiteralPath $taskPidFile) {
    # Windows PowerShell 5 defaults to ANSI; project paths can contain Chinese characters.
    $taskOldProcesses = Get-Content -LiteralPath $taskPidFile -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($taskOldProcess in $taskOldProcesses) {
        $taskExistingProcess = Get-Process -Id $taskOldProcess.pid -ErrorAction SilentlyContinue
        if ($null -ne $taskExistingProcess -and $taskExistingProcess.Path) {
            # Reboots can reuse PIDs. Only a matching executable and start time belong to this run.
            $taskRecordedStart = if ($taskOldProcess.started -is [datetime]) { $taskOldProcess.started.ToUniversalTime() } else { [datetime]::Parse($taskOldProcess.started).ToUniversalTime() }
            $taskSamePath = [System.IO.Path]::GetFullPath($taskExistingProcess.Path) -eq [System.IO.Path]::GetFullPath($taskOldProcess.executable)
            $taskSameStart = [math]::Abs(($taskExistingProcess.StartTime.ToUniversalTime() - $taskRecordedStart).TotalSeconds) -lt 0.01
            if ($taskSamePath -and $taskSameStart) {
                throw 'Services already running. Run .\scripts\stop.ps1 before restarting.'
            }
        }
    }
}
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
function Start-SupportProcess($taskName, $taskExecutable, $taskArguments) {
    $taskProcess = Start-Process -FilePath $taskExecutable -ArgumentList $taskArguments -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskRuntime "$taskName.log") -RedirectStandardError (Join-Path $taskRuntime "$taskName.error.log")
    return @{ name = $taskName; pid = $taskProcess.Id; executable = $taskExecutable; started = $taskProcess.StartTime.ToUniversalTime().ToString('o') }
}
try {
    $taskProcesses += Start-SupportProcess 'qdrant' (Join-Path $taskRoot 'tools\qdrant\qdrant.exe') '--config-path tools/qdrant/config.yaml'
    $taskVectorDeadline = [datetime]::UtcNow.AddSeconds(30)
    $taskVectorReady = $false
    while ([datetime]::UtcNow -lt $taskVectorDeadline) {
        try {
            Invoke-WebRequest 'http://127.0.0.1:6333/readyz' -UseBasicParsing -TimeoutSec 1 | Out-Null
            $taskVectorReady = $true
            break
        } catch { Start-Sleep -Milliseconds 300 }
    }
    if (-not $taskVectorReady) { throw 'Qdrant startup failed. Check runtime/qdrant*.log.' }
    $taskProcesses += Start-SupportProcess 'livekit' (Join-Path $taskRoot 'tools\livekit\livekit-server.exe') '--dev --bind 127.0.0.1'
    $taskProcesses += Start-SupportProcess 'api' $taskPython '-m uvicorn app.main:app --host 127.0.0.1 --port 8000'
    $taskProcesses += Start-SupportProcess 'agent' $taskPython '-m app.voice start'
    $taskProcesses | ConvertTo-Json | Set-Content -LiteralPath $taskPidFile -Encoding utf8
} catch {
    foreach ($taskProcess in $taskProcesses) { Stop-Process -Id $taskProcess.pid -ErrorAction SilentlyContinue }
    throw
}
Write-Host 'Services started: http://127.0.0.1:8000'
Write-Host 'Logs: runtime/*.log ; stop: .\scripts\stop.ps1'
Write-Host 'Add LLM_API_KEY, LLM_BASE_URL, LLM_MODEL to .env, then restart for model-backed voice.'
