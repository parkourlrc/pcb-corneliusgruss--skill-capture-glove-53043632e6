"""Enable external trigger mode on Arducam OV9782 USB cameras.

Sets UVC Camera Terminal controls:
  auto-exposure-mode = 8 (aperture priority — AEC stays active)
  auto-exposure-priority = 1 (external trigger via FSIN/F pad)

AEC remains on so the sensor adapts exposure to the scene between triggers.
This is critical for VIO — fixed manual exposure produces low-contrast images
(grayscale std 13-17 vs the 40+ needed for feature detection).

AWB handling: In triggered mode the ISP gets too few frames to converge AWB,
producing a yellow tint. The --lock-awb flag (default) lets cameras free-run
for 3 seconds so AWB converges, reads the settled value, locks it, then
enables trigger mode.

On macOS: uses uvc-util (IOKit, no kernel driver detach)
On Linux: uses v4l2-ctl

Usage:
  sudo python -m glove.scripts.enable_trigger                          # enable + lock AWB
  sudo python -m glove.scripts.enable_trigger --no-lock-awb            # enable without AWB lock
  sudo python -m glove.scripts.enable_trigger --off                    # disable (free-running)
  sudo python -m glove.scripts.enable_trigger --isp                    # show all ISP controls
  sudo python -m glove.scripts.enable_trigger --set wb=5500 gamma=120  # set ISP params

Reference: https://docs.arducam.com/UVC-Camera/Appilcation-Note/External-Trigger-Mode/
"""

import os
import platform
import subprocess
import sys
import tempfile
import time

ARDUCAM_VID = "0x0c45"
ARDUCAM_PID = "0x6366"
UVC_UTIL_PATH = os.path.expanduser("~/.local/bin/uvc-util")

# ISP control names and their UVC identifiers
ISP_CONTROLS = {
    "wb": "white-balance-temp",
    "awb": "auto-white-balance-temp",
    "exposure": "exposure-time-abs",
    "gain": "gain",
    "brightness": "brightness",
    "contrast": "contrast",
    "saturation": "saturation",
    "sharpness": "sharpness",
    "gamma": "gamma",
    "hue": "hue",
}

AWB_SETTLE_SECONDS = 3  # how long to free-run for AWB convergence


def run_uvc_util(device_index: int, controls: list[tuple[str, str]]) -> bool:
    """Set UVC controls on a device via uvc-util (macOS IOKit)."""
    cmd = [UVC_UTIL_PATH, "-I", str(device_index)]
    for name, value in controls:
        cmd.extend(["-s", f"{name}={value}"])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_device_count() -> int:
    """Count UVC devices via uvc-util."""
    try:
        result = subprocess.run(
            [UVC_UTIL_PATH, "--list-devices"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.count("locationID")
    except Exception:
        return 0


def read_uvc_control(device_index: int, control_name: str) -> str | None:
    """Read a single UVC control value. Returns the value string or None."""
    try:
        result = subprocess.run(
            [UVC_UTIL_PATH, "-I", str(device_index), "-S", control_name],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if "current-value" in line:
                # Parse "  current-value: 4600" → "4600"
                return line.split(":")[-1].strip()
    except Exception:
        pass
    return None


def show_status(device_index: int) -> None:
    """Show current trigger and image quality control values."""
    try:
        result = subprocess.run(
            [UVC_UTIL_PATH, "-I", str(device_index), "-S", "*"],
            capture_output=True, text=True, timeout=5,
        )
        current_control = ""
        for line in result.stdout.splitlines():
            # Top-level control name: not indented, ends with " {"
            if not line.startswith(" ") and line.rstrip().endswith("{"):
                current_control = line.strip().rstrip(" {")
            elif "current-value:" in line and current_control:
                val = line.split("current-value:")[-1].strip()
                print(f"    {current_control}: {val}")
    except Exception:
        pass


def build_uvc_util() -> None:
    """Clone and build uvc-util into ~/.local/bin/ if not already present."""
    print(f"uvc-util not found at {UVC_UTIL_PATH} — building from source...")
    os.makedirs(os.path.dirname(UVC_UTIL_PATH), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        clone_dir = os.path.join(tmpdir, "uvc-util")
        subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/jtfrey/uvc-util.git", clone_dir],
            check=True,
        )
        subprocess.run(
            ["clang", "-o", UVC_UTIL_PATH,
             "-framework", "IOKit",
             "-framework", "Foundation",
             "-framework", "CoreFoundation",
             "-fno-objc-arc",
             *[os.path.join(clone_dir, "src", f)
               for f in os.listdir(os.path.join(clone_dir, "src"))
               if f.endswith(".m")]],
            check=True,
        )
    print(f"Built and installed to {UVC_UTIL_PATH}")


def ensure_uvc_util() -> None:
    """Auto-build uvc-util if missing."""
    if os.path.isfile(UVC_UTIL_PATH):
        return
    try:
        build_uvc_util()
    except Exception as e:
        print(f"ERROR: failed to build uvc-util: {e}")
        print("Manual build:")
        print("  git clone https://github.com/jtfrey/uvc-util.git /tmp/uvc-util")
        print("  clang -o ~/.local/bin/uvc-util -framework IOKit -framework Foundation "
              "-framework CoreFoundation -fno-objc-arc /tmp/uvc-util/src/*.m")
        sys.exit(1)


def find_cameras() -> list[int]:
    """Return device indices of connected Arducam OV9782 cameras."""
    indices = []
    for i in range(8):
        # Probe by trying to read a control
        val = read_uvc_control(i, "auto-exposure-mode")
        if val is not None:
            indices.append(i)
        elif indices:
            break  # Stop after gap
    return indices


def lock_isp(camera_indices: list[int]) -> dict[str, str]:
    """Let cameras free-run for ISP convergence, then lock all ISP params.

    Returns {control_name: locked_value} dict.
    """
    print(f"\nAuto ISP: free-running for {AWB_SETTLE_SECONDS}s to let ISP converge...")

    # Disable trigger, enable all auto controls
    for i in camera_indices:
        run_uvc_util(i, [
            ("auto-exposure-mode", "8"),
            ("auto-exposure-priority", "0"),  # free-running
            ("auto-white-balance-temp", "1"),  # AWB on
        ])

    # Wait for ISP to converge
    for t in range(AWB_SETTLE_SECONDS, 0, -1):
        print(f"  ISP settling... {t}s", end="\r", flush=True)
        time.sleep(1)
    print(f"  ISP settled after {AWB_SETTLE_SECONDS}s      ")

    # Read all converged ISP values
    isp_keys = [
        "white-balance-temp", "exposure-time-abs", "gain",
        "gamma", "sharpness", "contrast", "saturation", "brightness", "hue",
    ]
    locked = {}
    for key in isp_keys:
        val = read_uvc_control(camera_indices[0], key)
        if val is not None:
            locked[key] = val

    # Lock: disable AWB, write all values to all cameras
    lock_controls = [("auto-white-balance-temp", "0")]
    for key, val in locked.items():
        lock_controls.append((key, val))

    for i in camera_indices:
        run_uvc_util(i, lock_controls)

    wb = locked.get("white-balance-temp", "?")
    exp = locked.get("exposure-time-abs", "?")
    gain = locked.get("gain", "?")
    print(f"  Locked: WB={wb}K  exposure={exp}  gain={gain}")
    return locked


def show_isp(camera_indices: list[int]) -> None:
    """Show all ISP controls for each camera."""
    for i in camera_indices:
        print(f"\n=== Camera {i} ISP controls ===")
        show_status(i)


def set_isp(camera_indices: list[int], params: dict[str, str]) -> None:
    """Set ISP parameters on all cameras.

    params: dict like {"wb": "5500", "gamma": "120"}
    """
    controls = []
    for short_name, value in params.items():
        uvc_name = ISP_CONTROLS.get(short_name)
        if uvc_name is None:
            print(f"WARNING: unknown ISP control '{short_name}'. "
                  f"Known: {', '.join(ISP_CONTROLS.keys())}")
            continue
        # If setting WB manually, disable AWB first
        if short_name == "wb":
            controls.append(("auto-white-balance-temp", "0"))
        controls.append((uvc_name, value))

    if not controls:
        return

    for i in camera_indices:
        if run_uvc_util(i, controls):
            print(f"Camera {i}: set {params}")
        else:
            print(f"Camera {i}: FAILED to set {params}")


def enable_trigger_macos(disable: bool = False, lock_awb_flag: bool = True) -> None:
    """Enable/disable trigger mode via uvc-util (macOS)."""
    ensure_uvc_util()

    # List devices
    result = subprocess.run(
        [UVC_UTIL_PATH, "--list-devices"],
        capture_output=True, text=True, timeout=5,
    )
    print(result.stdout.strip())

    camera_indices = find_cameras()
    if not camera_indices:
        print("No cameras found. Are they plugged in?")
        return

    if disable:
        for i in camera_indices:
            run_uvc_util(i, [
                ("auto-exposure-mode", "8"),
                ("auto-exposure-priority", "0"),
                ("auto-white-balance-temp", "1"),  # Re-enable AWB
            ])
            print(f"\nCamera {i}: trigger DISABLED (free-running, AWB on)")
            show_status(i)
    else:
        # Lock all ISP params before enabling trigger
        if lock_awb_flag:
            lock_isp(camera_indices)
            time.sleep(0.5)  # let USB settle before next batch

        # Enable trigger
        for i in camera_indices:
            run_uvc_util(i, [
                ("auto-exposure-mode", "8"),
                ("auto-exposure-priority", "1"),
            ])
            print(f"\nCamera {i}: trigger ENABLED (AEC on + external trigger)")
            show_status(i)

    print(f"\n{len(camera_indices)} camera(s) configured.")


def enable_trigger_v4l2(disable: bool = False) -> None:
    """Enable/disable trigger mode via v4l2-ctl (Linux)."""
    import glob

    if disable:
        ae_val, trigger_val = "3", "0"
    else:
        ae_val, trigger_val = "1", "1"

    for dev_path in sorted(glob.glob("/dev/video*")):
        try:
            result = subprocess.run(
                ["v4l2-ctl", "-d", dev_path, "--info"],
                capture_output=True, text=True, timeout=5,
            )
            if "OV9782" not in result.stdout and "Arducam" not in result.stdout:
                continue

            subprocess.run(
                ["v4l2-ctl", "-d", dev_path,
                 "-c", f"auto_exposure={ae_val}",
                 "-c", f"exposure_dynamic_framerate={trigger_val}"],
                check=True, timeout=5,
            )
            label = "ENABLED" if not disable else "DISABLED"
            print(f"{dev_path}: trigger mode {label}")
        except Exception:
            continue


def main() -> None:
    args = sys.argv[1:]
    disable = "--off" in args
    no_lock_awb = "--no-lock-awb" in args
    show_isp_flag = "--isp" in args
    set_flag = "--set" in args

    if platform.system() == "Darwin":
        ensure_uvc_util()

    if show_isp_flag:
        camera_indices = find_cameras()
        if not camera_indices:
            print("No cameras found.")
            return
        show_isp(camera_indices)
        return

    if set_flag:
        camera_indices = find_cameras()
        if not camera_indices:
            print("No cameras found.")
            return
        # Parse key=value pairs after --set
        params = {}
        for arg in args:
            if "=" in arg and not arg.startswith("-"):
                k, v = arg.split("=", 1)
                params[k] = v
        if params:
            set_isp(camera_indices, params)
            print("\nCurrent values:")
            show_isp(camera_indices)
        else:
            print("Usage: --set wb=5500 gamma=120 sharpness=4")
        return

    if platform.system() == "Linux":
        enable_trigger_v4l2(disable)
    else:
        enable_trigger_macos(disable, lock_awb_flag=not no_lock_awb)


if __name__ == "__main__":
    main()
