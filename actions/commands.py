import subprocess
import shutil
import pyautogui


# ==========================================================
# VOLUME UP
# ==========================================================

def volume_up():

    print("ACTION: Volume Up")

    pyautogui.press(
        "volumeup"
    )


# ==========================================================
# VOLUME DOWN
# ==========================================================

def volume_down():

    print("ACTION: Volume Down")

    pyautogui.press(
        "volumedown"
    )


# ==========================================================
# PLAY / PAUSE
# ==========================================================

def play_pause():

    print("ACTION: Play / Pause")

    pyautogui.press(
        "playpause"
    )


# ==========================================================
# OPEN VS CODE
# ==========================================================

def open_vscode():

    print("ACTION: Open VS Code")

    # Try to find the VS Code CLI
    vscode = shutil.which(
        "code"
    )

    if vscode is None:

        print(
            "WARNING: VS Code 'code' command "
            "was not found in PATH."
        )

        return

    try:

        subprocess.Popen(
            [vscode]
        )

    except Exception as error:

        print(
            f"ERROR opening VS Code: "
            f"{error}"
        )