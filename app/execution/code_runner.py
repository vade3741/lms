import tempfile
import subprocess
import os

import sys

def run_python_code(code: str, test_cases: str):
    """
    Runs the given python code along with test cases and returns (passed, output).
    """
    if not code.strip():
        return False, "No code provided."

    if isinstance(test_cases, list):
        test_str = "\n".join(str(tc) for tc in test_cases)
    elif isinstance(test_cases, dict):
        test_str = str(test_cases)
    else:
        test_str = str(test_cases or "")

    combined_code = code + "\n\n" + test_str

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(combined_code)
        temp_file_path = f.name

    try:
        # Run the temporary file using python
        result = subprocess.run(
            [sys.executable, temp_file_path],
            capture_output=True,
            text=True,
            timeout=5
        )

        passed = result.returncode == 0
        
        output = ""
        if result.stdout:
            output += "--- STDOUT ---\n" + result.stdout + "\n"
        if result.stderr:
            output += "--- STDERR ---\n" + result.stderr + "\n"
            
        if not output:
            output = "Execution completed successfully with no output."

        return passed, output
    except subprocess.TimeoutExpired:
        return False, "Error: Code execution timed out after 5 seconds."
    except Exception as e:
        return False, f"Error: {str(e)}"
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
