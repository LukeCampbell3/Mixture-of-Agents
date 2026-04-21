# =============================================================================
# Ollama model management helper — Windows PowerShell
# =============================================================================
# Usage:
#   .\scripts\setup_ollama.ps1              # pull auto-selected model
#   .\scripts\setup_ollama.ps1 -Size 7b     # pull qwen2.5:7b
#   .\scripts\setup_ollama.ps1 -Action list # list available models
#   .\scripts\setup_ollama.ps1 -Action status
# =============================================================================

param(
    [string]$Action = "auto",
    [string]$Size   = "",
    [string]$Model  = ""
)

$OllamaUrl = $env:OLLAMA_URL ?? "http://localhost:11434"

function Write-Ok   { param($m) Write-Host "  [OK] $m" -ForegroundColor Green }
function Write-Warn { param($m) Write-Host "  [!!] $m" -ForegroundColor Yellow }
function Write-Err  { param($m) Write-Host "  [XX] $m" -ForegroundColor Red; exit 1 }

function Test-Server {
    try {
        $r = Invoke-WebRequest -Uri "$OllamaUrl/api/tags" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
        return $r.StatusCode -eq 200
    } catch { return $false }
}

function Get-Models {
    try {
        $r = Invoke-WebRequest -Uri "$OllamaUrl/api/tags" -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
        $data = $r.Content | ConvertFrom-Json
        return $data.models
    } catch { return @() }
}

function Get-AutoModel {
    $RamGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
    if ($RamGB -ge 12) { return "qwen2.5:7b" }
    elseif ($RamGB -ge 6) { return "qwen2.5:3b" }
    else { return "qwen2.5:1.5b" }
}

function Invoke-Pull {
    param([string]$ModelName)
    Write-Host "  Pulling $ModelName..." -ForegroundColor Cyan
    ollama pull $ModelName
    if ($LASTEXITCODE -eq 0) { Write-Ok "Model ready: $ModelName" }
    else { Write-Err "Failed to pull $ModelName" }
}

# ── Resolve model name ────────────────────────────────────────────────────────
if ($Model) {
    $TargetModel = $Model
} elseif ($Size) {
    $TargetModel = if ($Size -match ":") { $Size } else { "qwen2.5:$Size" }
} else {
    $TargetModel = ""
}

# ── Main ──────────────────────────────────────────────────────────────────────
switch ($Action.ToLower()) {
    "status" {
        if (Test-Server) {
            Write-Ok "Ollama server running at $OllamaUrl"
            $models = Get-Models
            if ($models.Count -gt 0) {
                Write-Host "`n  Pulled models:"
                foreach ($m in $models) {
                    $mb = [math]::Round($m.size / 1MB)
                    Write-Host ("    {0,-40} {1,6} MB" -f $m.name, $mb)
                }
            } else {
                Write-Host "  No models pulled yet."
            }
        } else {
            Write-Warn "Ollama server not reachable at $OllamaUrl"
        }
    }
    "list" {
        if (-not (Test-Server)) { Write-Err "Ollama not running. Start with: ollama serve" }
        $models = Get-Models
        if ($models.Count -gt 0) {
            foreach ($m in $models) {
                $mb = [math]::Round($m.size / 1MB)
                Write-Host ("  {0,-40} {1,6} MB" -f $m.name, $mb)
            }
        } else {
            Write-Host "  No models pulled yet."
        }
    }
    "auto" {
        if (-not (Test-Server)) { Write-Err "Ollama not running. Start with: ollama serve" }
        if (-not $TargetModel) { $TargetModel = Get-AutoModel }
        Write-Host "  Auto-selected: $TargetModel" -ForegroundColor Cyan
        Invoke-Pull $TargetModel
    }
    default {
        if (-not (Test-Server)) { Write-Err "Ollama not running. Start with: ollama serve" }
        if (-not $TargetModel) { $TargetModel = $Action }
        if ($TargetModel -notmatch ":") { $TargetModel = "qwen2.5:$TargetModel" }
        Invoke-Pull $TargetModel
    }
}
