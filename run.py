# This script is used as an entrypoint to forcibly uninstall the 
# GUI version of opencv-python that is transitively imported by ultralytics.
import subprocess
import sys
import os

def strip_gui_opencv():
    try:
        # Uninstall the bloated gui version pulled by YOLO
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python"])
    except Exception as e:
        print(f"Skipping uninstall: {e}")

if __name__ == "__main__":
    strip_gui_opencv()
    # Now that the bad package is gone, boot the main Streamlit script
    os.system("streamlit run app.py")
