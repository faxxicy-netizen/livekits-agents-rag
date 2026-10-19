$ErrorActionPreference = 'Stop'
$taskRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$taskPidFile = Join-Path $taskRoot 'runtime\processes.json'
if (-not (Test-Path -LiteralPath $taskPidFile)) { Write-Host 'No recorded services.'; exit 0 }
$taskProcesses = Get-Content -LiteralPath $taskPidFile -Raw -Encoding UTF8 | ConvertFrom-Json
$taskRemaining = @()
foreach ($taskEntry in $taskProcesses) {
    try {
        $taskProcess = Get-Process -Id $taskEntry.pid -ErrorAction SilentlyContinue
        if ($null -eq $taskProcess) {
            Write-Host "Already stopped $($taskEntry.name)"
            continue
        }
        # Check identity and start time so a reused process ID never stops an unrelated app.
        if (-not $taskProcess.Path) { throw 'Cannot read process identity.' }
        $taskActualPath = [System.IO.Path]::GetFullPath($taskProcess.Path)
        $taskExpectedPath = [System.IO.Path]::GetFullPath($taskEntry.executable)
        $taskExpectedStart = if ($taskEntry.started -is [datetime]) { $taskEntry.started.ToUniversalTime() } else { [datetime]::Parse($taskEntry.started).ToUniversalTime() }
        $taskStartDifference = [math]::Abs(($taskProcess.StartTime.ToUniversalTime() - $taskExpectedStart).TotalSeconds)
        if ($taskActualPath -ne $taskExpectedPath -or $taskStartDifference -ge 0.01) {
            Write-Host "Skipped stale record for $($taskEntry.name) (PID $($taskEntry.pid))."
            continue
        }
        # Python's Windows venv executable is a launcher with a real child process.
        $taskChildren = Get-CimInstance Win32_Process -Filter "ParentProcessId = $($taskEntry.pid)"
        foreach ($taskChild in $taskChildren) {
            if ($taskChild.CreationDate.ToUniversalTime() -ge $taskExpectedStart.AddSeconds(-1)) {
                $taskChildProcess = Get-Process -Id $taskChild.ProcessId -ErrorAction SilentlyContinue
                if ($null -ne $taskChildProcess -and -not $taskChildProcess.HasExited) {
                    $taskChildStartDifference = [math]::Abs(($taskChildProcess.StartTime.ToUniversalTime() - $taskChild.CreationDate.ToUniversalTime()).TotalSeconds)
                    if ($taskChildStartDifference -lt 0.01) {
                        Stop-Process -InputObject $taskChildProcess -ErrorAction Stop
                    }
                }
            }
        }
        if (-not $taskProcess.HasExited) { Stop-Process -InputObject $taskProcess -ErrorAction Stop }
        Write-Host "Stopped $($taskEntry.name)"
    } catch {
        # Continue stopping the other services; retain failed records for a later retry.
        $taskRemaining += $taskEntry
        Write-Warning "Could not stop $($taskEntry.name): $($_.Exception.Message)"
    }
}
if ($taskRemaining.Count -gt 0) {
    ConvertTo-Json -InputObject @($taskRemaining) | Set-Content -LiteralPath $taskPidFile -Encoding UTF8
    throw 'Some services could not be stopped. Their process records were preserved for retry.'
}
Remove-Item -LiteralPath $taskPidFile
