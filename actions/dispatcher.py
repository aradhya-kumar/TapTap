import time

from actions import commands


# ==========================================================
# ACTION MAP
# ==========================================================

ACTION_MAP = {

    # ------------------------------------------------------
    # SINGLE TAPS
    # ------------------------------------------------------

    ("top_left", "single"):
        commands.play_pause,

    ("top_right", "single"):
        commands.open_vscode,

    ("bottom_left", "single"):
        commands.volume_down,

    ("bottom_right", "single"):
        commands.volume_up,


    # ------------------------------------------------------
    # DOUBLE TAPS
    #
    # For now these use the same functions.
    # We'll change them to whatever actions you want.
    # ------------------------------------------------------

    ("top_left", "double"):
        commands.play_pause,

    ("top_right", "double"):
        commands.open_vscode,

    ("bottom_left", "double"):
        commands.volume_down,

    ("bottom_right", "double"):
        commands.volume_up,
}


# ==========================================================
# COOLDOWN
# ==========================================================

ACTION_COOLDOWN = 0.15

last_action_time = 0


# ==========================================================
# EXECUTE
# ==========================================================

def execute(
    zone,
    gesture
):

    global last_action_time

    zone = str(
        zone
    ).lower()

    gesture = str(
        gesture
    ).lower()

    current_time = (
        time.time()
    )

    # ------------------------------------------------------
    # Prevent accidental duplicate execution
    # ------------------------------------------------------

    if (
        current_time
        - last_action_time
        < ACTION_COOLDOWN
    ):

        print(
            "Action ignored "
            "(cooldown)"
        )

        return

    # ------------------------------------------------------
    # Find action
    # ------------------------------------------------------

    action = ACTION_MAP.get(
        (
            zone,
            gesture
        )
    )

    if action is None:

        print(
            f"No action assigned: "
            f"{zone} / {gesture}"
        )

        return

    last_action_time = (
        current_time
    )

    print(
        f"Executing: "
        f"{gesture.upper()} "
        f"{zone.upper()}"
    )

    action()