#!/usr/bin/env python3
"""
Setup script for Claude Code + Agentic Network integration.
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check Python version."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

def install_requirements():
    """Install required packages."""
    print("\nInstalling requirements...")
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("Error: requirements.txt not found.")
        sys.exit(1)
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
    except subprocess.CalledProcessError:
        print("Error: Failed to install requirements")
        sys.exit(1)

def create_symlink():
    """Create a symlink/alias for easy access."""
    print("\nCreating CLI symlink...")
    
    # Determine the appropriate command name
    cli_name = "claude-agentic"
    
    # Create a simple wrapper script
    wrapper_content = f'''#!/usr/bin/env python3
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from claude_integrated import main

if __name__ == "__main__":
    main()
'''
    
    wrapper_path = f"{cli_name}.py"
    with open(wrapper_path, "w") as f:
        f.write(wrapper_content)
    
    # Make it executable
    if platform.system() != "Windows":
        os.chmod(wrapper_path, 0o755)
    
    print(f"✓ Created wrapper script: {wrapper_path}")
    print(f"\nTo use the CLI:")
    print(f"  python {wrapper_path}")
    
    if platform.system() != "Windows":
        print(f"\nOr make it globally available:")
        print(f"  sudo cp {wrapper_path} /usr/local/bin/{cli_name}")
        print(f"  sudo chmod +x /usr/local/bin/{cli_name}")
        print(f"  Then run: {cli_name}")

def check_ollama():
    """Check if Ollama is installed and running."""
    print("\nChecking Ollama...")
    
    try:
        # Try to check Ollama version
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Ollama found: {result.stdout.strip()}")
            
            # Check if Ollama is running
            try:
                import requests
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    print("✓ Ollama server is running")
                else:
                    print("⚠ Ollama server may not be running")
                    print("  Start with: ollama serve")
            except:
                print("⚠ Cannot connect to Ollama server")
                print("  Start with: ollama serve")
        else:
            print("⚠ Ollama not found or not in PATH")
            print("  Install from: https://ollama.com/")
    except FileNotFoundError:
        print("⚠ Ollama not found or not in PATH")
        print("  Install from: https://ollama.com/")

def pull_default_model():
    """Pull the default Qwen2.5 model."""
    print("\nChecking default model...")
    
    try:
        # Check if model is available
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            qwen_models = [m for m in models if "qwen2.5" in m.get("name", "").lower()]
            
            if qwen_models:
                print(f"✓ Qwen2.5 model found: {qwen_models[0]['name']}")
            else:
                print("⚠ Qwen2.5 model not found")
                print("  Pull with: ollama pull qwen2.5:7b")
        else:
            print("⚠ Cannot check models - Ollama may not be running")
    except:
        print("⚠ Cannot check models - Ollama may not be running")

def main():
    print("=" * 60)
    print("Claude Code + Agentic Network v2 Setup")
    print("=" * 60)
    
    # Check Python
    check_python_version()
    
    # Check Ollama
    check_ollama()
    
    # Install requirements
    install_requirements()
    
    # Check default model
    pull_default_model()
    
    # Create symlink
    create_symlink()
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("\nNext steps:")
    print("1. Start Ollama server (if not running):")
    print("   ollama serve")
    print("2. Pull Qwen2.5 model (if not available):")
    print("   ollama pull qwen2.5:7b")
    print("3. Start the CLI:")
    print("   python claude_integrated.py")
    print("4. Configure (optional):")
    print('   /config set llm_model "your-model"')
    print('   /config set llm_base_url "your-url"')
    print("=" * 60)

if __name__ == "__main__":
    main()