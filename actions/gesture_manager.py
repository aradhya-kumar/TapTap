import time
import threading


DOUBLE_TAP_WINDOW = 0.60


class GestureManager:

    def __init__(self, callback):

        self.callback = callback

        self.first_tap_zone = None
        self.first_tap_time = None

        self.single_timer = None

        self.lock = threading.Lock()


    # ======================================================
    # REGISTER TAP
    # ======================================================

    def register_tap(self, zone):

        zone = str(zone).lower()

        current_time = time.monotonic()

        print()
        print(
            f"[GESTURE] Tap received: "
            f"{zone.upper()}"
        )

        with self.lock:

            # ==================================================
            # CHECK FOR DOUBLE TAP
            # ==================================================

            if (
                self.first_tap_zone is not None
                and
                self.first_tap_time is not None
            ):

                elapsed = (
                    current_time
                    - self.first_tap_time
                )

                print(
                    f"[GESTURE] Time since first tap: "
                    f"{elapsed:.3f}s"
                )

                # Same zone + inside time window

                if (
                    zone == self.first_tap_zone
                    and
                    elapsed <= DOUBLE_TAP_WINDOW
                ):

                    if self.single_timer is not None:

                        self.single_timer.cancel()

                    self.single_timer = None

                    self.first_tap_zone = None

                    self.first_tap_time = None

                    print()
                    print("=" * 50)
                    print(
                        f"DOUBLE TAP -> "
                        f"{zone.upper()}"
                    )
                    print("=" * 50)

                    self.callback(
                        zone,
                        "double"
                    )

                    return

                # ==================================================
                # OLD TAP WAS NOT PART OF A DOUBLE
                # ==================================================

                previous_zone = (
                    self.first_tap_zone
                )

                if self.single_timer is not None:

                    self.single_timer.cancel()

                self.single_timer = None

                self.first_tap_zone = None

                self.first_tap_time = None

                print(
                    f"[GESTURE] Previous tap completed "
                    f"as SINGLE: "
                    f"{previous_zone.upper()}"
                )

                self.callback(
                    previous_zone,
                    "single"
                )

            # ==================================================
            # STORE NEW FIRST TAP
            # ==================================================

            self.first_tap_zone = zone

            self.first_tap_time = (
                current_time
            )

            print(
                f"[GESTURE] Waiting "
                f"{DOUBLE_TAP_WINDOW:.2f}s "
                f"for second {zone.upper()} tap..."
            )

            self.single_timer = threading.Timer(

                DOUBLE_TAP_WINDOW,

                self._complete_single,

                args=(
                    zone,
                    current_time
                )

            )

            self.single_timer.daemon = True

            self.single_timer.start()


    # ======================================================
    # COMPLETE SINGLE TAP
    # ======================================================

    def _complete_single(
        self,
        zone,
        tap_time
    ):

        with self.lock:

            # Make sure this timer still belongs
            # to the currently pending tap.

            if (
                self.first_tap_zone != zone
                or
                self.first_tap_time != tap_time
            ):

                return

            self.first_tap_zone = None

            self.first_tap_time = None

            self.single_timer = None

            print()
            print("=" * 50)
            print(
                f"SINGLE TAP -> "
                f"{zone.upper()}"
            )
            print("=" * 50)

            self.callback(
                zone,
                "single"
            )