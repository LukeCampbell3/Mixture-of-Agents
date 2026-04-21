#Requires -Version 5.1
<#
.SYNOPSIS
    Mixture-of-Agents CLI — Windows installer

.DESCRIPTION
    Installs all prerequisites and sets up the CLI on Windows.
    Requires PowerShell 5.1+ (built into Windows 10/11).

.PARAMETER Model
    Ollama model to pull. Default: auto-selected based on RAM.
    Examples: qwen2.5:1.5b, qwen2.5:3b, qwen2.5:7b

.PARAMETER NoOllama
    Skip Ollama installation. Use when you have OpenAI/Anthropic API keys.

.PARAMETER Dev
    Install dev/test dependencies (pytest, ruff, mypy).

.EXAMPLE
    .\install.ps1
    .\install.ps1 -Model qwen2.5:1.5b
    .\install.ps1 -NoOllama
#>

param(
    [string]$Model = "auto",
    [switch]$NoOllama,
    [switch]$Dev
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Colours ───────────────────────────────────────────────────────────────────
function Write-Ok   { param($msg) Write-Host "  [OK] $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "  [!!] $msg" -ForegroundColor Yellow }
function Write-Err  { param($msg) Write-Host "  [XX] $msg" -ForegroundColor Red }
function Write-Info { param($msg) Write-Host "  --> $msg" -ForegroundColor Cyan }
function Write-Hdr  { param($msg) Write-Host "`n$msg" -ForegroundColor White; Write-Host ("-" * 60) }

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# ── Banner ────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║       Mixture-of-Agents CLI — Windows Installer          ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Platform : Windows $([System.Environment]::OSVersion.Version)"
Write-Host "  Directory: $ScriptDir"
Write-Host ""

# ── Helper: run command, throw on failure ─────────────────────────────────────
function Invoke-Cmd {
    param([string]$Cmd, [string[]]$Args, [string]$Desc = "")
    if ($Desc) { Write-Info $Desc }
    & $Cmd @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed (exit $LASTEXITCODE): $Cmd $($Args -join ' ')"
    }
}

# ── Helper: check if command exists ──────────────────────────────────────────
function Test-Cmd { param($Name) return [bool](Get-Command $Name -ErrorAction SilentlyContinue) }

# ── Helper: download file ─────────────────────────────────────────────────────
function Get-Download {
    param([string]$Url, [string]$Dest)
    Write-Info "Downloading $([System.IO.Path]::GetFileName($Dest))..."
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $wc = New-Object System.Net.WebClient
    $wc.DownloadFile($Url, $Dest)
}

# ── 1. Execution policy ───────────────────────────────────────────────────────
Write-Hdr "1. Execution Policy"
$policy = Get-ExecutionPolicy -Scope CurrentUser
if ($policy -eq "Restricted") {
    Write-Info "Setting execution policy to RemoteSigned for current user..."
    Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
    Write-Ok "Execution policy updated"
} else {
    Write-Ok "Execution policy: $policy"
}

# ── 2. Winget / Chocolatey detection ─────────────────────────────────────────
Write-Hdr "2. Package Manager"
$HasWinget = Test-Cmd "winget"
$HasChoco  = Test-Cmd "choco"

if ($HasWinget) {
    Write-Ok "winget available"
} elseif ($HasChoco) {
    Write-Ok "Chocolatey available"
} else {
    Write-Warn "No package manager found. Will use direct downloads."
}

# ── 3. Python ─────────────────────────────────────────────────────────────────
Write-Hdr "3. Python"

$MinPyMinor = 9
$PythonCmd  = $null

foreach ($cmd in @("python", "python3", "py")) {
    if (Test-Cmd $cmd) {
        try {
            $ver = & $cmd -c "import sys; print(sys.version_info.minor)" 2>$null
            if ([int]$ver -ge $MinPyMinor) {
                $PythonCmd = $cmd
                $PythonVer = & $cmd --version 2>&1
                Write-Ok "Found: $PythonVer ($cmd)"
                break
            }
        } catch {}
    }
}

if (-not $PythonCmd) {
    Write-Warn "Python 3.$MinPyMinor+ not found — installing..."

    if ($HasWinget) {
        winget install --id Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements
    } elseif ($HasChoco) {
        choco install python311 -y
    } else {
        # Direct download
        $PyInstaller = "$env:TEMP\python-installer.exe"
        Get-Download "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe" $PyInstaller
        Write-Info "Running Python installer (silent)..."
        Start-Process -FilePath $PyInstaller -ArgumentList "/quiet InstallAllUsers=0 PrependPath=1 Include_pip=1" -Wait
        Remove-Item $PyInstaller -Force -ErrorAction SilentlyContinue
    }

    # Refresh PATH
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("PATH", "User")

    foreach ($cmd in @("python", "python3", "py")) {
        if (Test-Cmd $cmd) {
            $PythonCmd = $cmd
            Write-Ok "Installed: $(& $cmd --version 2>&1)"
            break
        }
    }

    if (-not $PythonCmd) {
        Write-Err "Python installation failed. Please install manually from https://www.python.org/"
        Write-Err "Make sure to check 'Add Python to PATH' during installation."
        exit 1
    }
}

# ── 4. Virtual environment ────────────────────────────────────────────────────
Write-Hdr "4. Virtual Environment"

$VenvDir = Join-Path $ScriptDir ".venv"
if (-not (Test-Path $VenvDir)) {
    Write-Info "Creating virtual environment..."
    Invoke-Cmd $PythonCmd @("-m", "venv", $VenvDir) "Creating .venv"
    Write-Ok "Created .venv"
} else {
    Write-Ok "Existing .venv found"
}

$PythonVenv = Join-Path $VenvDir "Scripts\python.exe"
$PipVenv    = Join-Path $VenvDir "Scripts\pip.exe"

# Upgrade pip
& $PythonVenv -m pip install --quiet --upgrade pip wheel setuptools
Write-Ok "pip upgraded"

# ── 5. Python dependencies ────────────────────────────────────────────────────
Write-Hdr "5. Python Dependencies"

$ReqFile = Join-Path $ScriptDir "requirements.txt"
if (-not (Test-Path $ReqFile)) {
    Write-Err "requirements.txt not found at $ReqFile"
    exit 1
}

Write-Info "Installing from requirements.txt..."
Invoke-Cmd $PipVenv @("install", "--quiet", "-r", $ReqFile)
Write-Ok "Core dependencies installed"

if ($Dev) {
    Write-Info "Installing dev dependencies..."
    Invoke-Cmd $PipVenv @("install", "--quiet", "pytest", "pytest-asyncio", "black", "ruff", "mypy")
    Write-Ok "Dev dependencies installed"
}

# ── 6. Ollama ─────────────────────────────────────────────────────────────────
if (-not $NoOllama) {
    Write-Hdr "6. Ollama"

    if (Test-Cmd "ollama") {
        $OllamaVer = ollama --version 2>&1
        Write-Ok "Ollama already installed: $OllamaVer"
    } else {
        Write-Info "Installing Ollama..."

        if ($HasWinget) {
            winget install --id Ollama.Ollama --silent --accept-package-agreements --accept-source-agreements
        } else {
            $OllamaInstaller = "$env:TEMP\OllamaSetup.exe"
            Get-Download "https://ollama.com/download/OllamaSetup.exe" $OllamaInstaller
            Write-Info "Running Ollama installer (silent)..."
            Start-Process -FilePath $OllamaInstaller -ArgumentList "/S" -Wait
            Remove-Item $OllamaInstaller -Force -ErrorAction SilentlyContinue
        }

        # Refresh PATH
        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("PATH", "User")
        $OllamaDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
        if (Test-Path $OllamaDir) {
            $env:PATH = "$OllamaDir;$env:PATH"
        }

        if (Test-Cmd "ollama") {
            Write-Ok "Ollama installed"
        } else {
            Write-Warn "Ollama installed but not on PATH yet — a new terminal session may be needed"
        }
    }

    # Start Ollama server
    $OllamaRunning = $false
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
        $OllamaRunning = ($resp.StatusCode -eq 200)
    } catch {}

    if (-not $OllamaRunning) {
        Write-Info "Starting Ollama server..."
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden -ErrorAction SilentlyContinue

        Write-Info "Waiting for Ollama server..."
        $ready = $false
        for ($i = 0; $i -lt 20; $i++) {
            Start-Sleep -Seconds 1
            try {
                $r = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
                if ($r.StatusCode -eq 200) { $ready = $true; break }
            } catch {}
        }
        if ($ready) { Write-Ok "Ollama server ready" }
        else { Write-Warn "Ollama server not responding — run 'ollama serve' manually" }
    } else {
        Write-Ok "Ollama server already running"
    }

    # ── 7. Model ──────────────────────────────────────────────────────────────
    Write-Hdr "7. Model"

    if ($Model -eq "auto") {
        $RamGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
        if ($RamGB -ge 12) {
            $Model = "qwen2.5:7b"
        } elseif ($RamGB -ge 6) {
            $Model = "qwen2.5:3b"
        } else {
            $Model = "qwen2.5:1.5b"
        }
        Write-Info "Auto-selected model for ${RamGB}GB RAM: $Model"
    }

    # Check if already pulled
    $modelBase = $Model.Split(":")[0]
    $alreadyPulled = $false
    try {
        $tags = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
        if ($tags.Content -match [regex]::Escape($modelBase)) {
            $alreadyPulled = $true
        }
    } catch {}

    if ($alreadyPulled) {
        Write-Ok "Model already available: $Model"
    } else {
        Write-Info "Pulling $Model (this may take several minutes)..."
        try {
            ollama pull $Model
            Write-Ok "Model ready: $Model"
        } catch {
            Write-Warn "Could not pull model automatically. Run: ollama pull $Model"
        }
    }
} else {
    Write-Hdr "6. Ollama"
    Write-Warn "Skipped (-NoOllama). Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env"
}

# ── 8. Environment file ───────────────────────────────────────────────────────
Write-Hdr "8. Environment"

$EnvFile    = Join-Path $ScriptDir ".env"
$EnvExample = Join-Path $ScriptDir ".env.example"

if (-not (Test-Path $EnvFile)) {
    if (Test-Path $EnvExample) {
        Copy-Item $EnvExample $EnvFile
        Write-Ok "Created .env from .env.example"
    } else {
        $envLines = @(
            '# Mixture-of-Agents CLI — Environment',
            'OPENAI_API_KEY=',
            'ANTHROPIC_API_KEY=',
            'DEFAULT_LLM_PROVIDER=ollama',
            'DEFAULT_BUDGET_MODE=balanced',
            'DATA_DIR=data'
        )
        $envLines | Set-Content $EnvFile
        Write-Ok "Created .env"
    }
    Write-Info "Edit .env to add API keys if using OpenAI/Anthropic"
} else {
    Write-Ok ".env already exists"
}

# ── 9. Launcher script ────────────────────────────────────────────────────────
Write-Hdr "9. Launcher"

$LauncherPs1 = Join-Path $ScriptDir "run.ps1"
$runPs1Lines = @(
    '# Mixture-of-Agents CLI launcher',
    '$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path',
    '$VenvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"',
    '',
    'if (-not (Test-Path $VenvPython)) {',
    '    Write-Host "Error: .venv not found. Run .\install.ps1 first." -ForegroundColor Red',
    '    exit 1',
    '}',
    '',
    '# Load .env',
    '$EnvFile = Join-Path $ScriptDir ".env"',
    'if (Test-Path $EnvFile) {',
    '    Get-Content $EnvFile | ForEach-Object {',
    '        if ($_ -match "^\s*([^#=][^=]*)=(.*)$") {',
    '            [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")',
    '        }',
    '    }',
    '}',
    '',
    'Set-Location $ScriptDir',
    '& $VenvPython claude_integrated.py @args'
)
$runPs1Lines | Set-Content $LauncherPs1
Write-Ok "Created run.ps1"

# Also create a .bat for users who prefer cmd.exe
$LauncherBat = Join-Path $ScriptDir "run.bat"
$runBatLines = @(
    '@echo off',
    'set SCRIPT_DIR=%~dp0',
    'set VENV_PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe',
    '',
    'if not exist "%VENV_PYTHON%" (',
    '    echo Error: .venv not found. Run install.ps1 first.',
    '    exit /b 1',
    ')',
    '',
    'if exist "%SCRIPT_DIR%.env" (',
    '    for /f "usebackq tokens=1,2 delims==" %%A in ("%SCRIPT_DIR%.env") do (',
    '        if not "%%A"=="" if not "%%A:~0,1%"=="#" set "%%A=%%B"',
    '    )',
    ')',
    '',
    'cd /d "%SCRIPT_DIR%"',
    '"%VENV_PYTHON%" claude_integrated.py %*'
)
$runBatLines | Set-Content $LauncherBat
Write-Ok "Created run.bat"

# ── Summary ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║  Installation complete!                                   ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "  Start the CLI (PowerShell):  " -NoNewline; Write-Host ".\run.ps1" -ForegroundColor Cyan
Write-Host "  Start the CLI (cmd.exe):     " -NoNewline; Write-Host "run.bat"   -ForegroundColor Cyan
Write-Host ""
if (-not $NoOllama) {
    Write-Host "  Model:       " -NoNewline; Write-Host $Model -ForegroundColor Cyan
    Write-Host "  Ollama API:  " -NoNewline; Write-Host "http://localhost:11434" -ForegroundColor Cyan
}
Write-Host "  Config:      " -NoNewline; Write-Host ".env  (add API keys here)" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Tip: Inside the CLI, type /help to see all commands." -ForegroundColor Yellow
Write-Host ""
