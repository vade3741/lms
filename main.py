import os
import sys
import subprocess

if __name__ == "__main__":
    app_path = os.path.join("app", "ui", "main.py")
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    print(f"Launching Streamlit application: {' '.join(cmd)}")
    subprocess.run(cmd)