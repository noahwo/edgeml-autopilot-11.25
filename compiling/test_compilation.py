#! /home/han/anaconda3/envs/llmdev/bin/python
import glob
import os
import subprocess
import sys
from datetime import datetime

arduino_core = "arduino:mbed_nano:nano33ble"
arduino_core_list = ["arduino:mbed_nano:nano33ble", "arduino:mbed:nano33ble"]


def compile_arduino_code(file_path, arduino_core):
    """
    Compile Arduino code using arduino-cli for Arduino Nano 33 BLE.

    Args:
        file_path: Path to the .ino file
    Returns:
        tuple: (success: bool, message: str)
    """
    execute_command = [
        "arduino-cli",
        "compile",
        "--fqbn",
        arduino_core,
        file_path,
    ]

    try:
        result = subprocess.run(execute_command, capture_output=True, text=True)

        # Combine stdout and stderr for error checking
        combined_output = (result.stdout or "") + (result.stderr or "")
        error_indicators = ["error:", "exception:", "traceback:", "failed:"]
        has_error = any(
            indicator in combined_output.lower() for indicator in error_indicators
        )

        if result.returncode == 0 and not has_error:
            success_msg = "Compilation successful!"
            if result.stdout and result.stdout.strip():
                success_msg += f"\nOutput: {result.stdout.strip()}"
            return True, success_msg
        else:
            error_message = result.stderr if result.stderr else result.stdout
            return False, f"Compilation failed: {error_message}"

    except Exception as e:
        return False, f"Error during compilation: {str(e)}"


if __name__ == "__main__":
    if len(sys.argv) == 2:
        # Use the provided file path
        ino_file = sys.argv[1]
    else:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Look for .ino files in the script's directory
        ino_files = glob.glob(os.path.join(script_dir, "*.ino"))
        if not ino_files:
            print(f"Error: No .ino files found in script directory:\n{script_dir}")
            sys.exit(1)
        if len(ino_files) > 1:
            print("Multiple .ino files found. Please specify which one to compile:")
            for i, file in enumerate(ino_files, 1):
                print(f"{i}. {file}")
            try:
                choice = int(input("Enter the number of the file to compile: "))
                if 1 <= choice <= len(ino_files):
                    ino_file = ino_files[choice - 1]
                else:
                    print("Invalid choice")
                    sys.exit(1)
            except ValueError:
                print("Invalid input")
                sys.exit(1)
        else:
            ino_file = ino_files[0]
            print(f"Found single .ino file: {ino_file}")
    for arduino_core in arduino_core_list:
        start_time = datetime.now()
        success, message = compile_arduino_code(ino_file, arduino_core)
        end_time = datetime.now()
        print(message)
        print(f"Time taken: {end_time - start_time}, for {arduino_core}")

    sys.exit(0 if success else 1)
