"""Display depth PNGs written by play.py without loading OpenCV into Isaac Sim."""

import argparse
import glob
import os
import time

import cv2


def _parent_is_alive(parent_pid: int) -> bool:
    if parent_pid <= 0:
        return True
    try:
        os.kill(parent_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a directory and display its latest depth PNG.")
    parser.add_argument("directory", help="Directory containing depth_*.png frames.")
    parser.add_argument("--parent-pid", type=int, default=0, help="Exit when this process exits.")
    parser.add_argument("--poll-interval", type=float, default=0.01, help="Directory polling interval in seconds.")
    args = parser.parse_args()

    title = "Depth Image"
    last_mtime_ns = -1
    window_created = False

    try:
        while _parent_is_alive(args.parent_pid):
            paths = glob.glob(os.path.join(args.directory, "depth_*.png"))
            if paths:
                latest = max(paths, key=lambda path: os.stat(path).st_mtime_ns)
                mtime_ns = os.stat(latest).st_mtime_ns
                if mtime_ns != last_mtime_ns:
                    image = cv2.imread(latest, cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        if not window_created:
                            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
                            cv2.resizeWindow(title, 320, 320)
                            window_created = True
                        cv2.imshow(title, image)
                        last_mtime_ns = mtime_ns

            if window_created:
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
            time.sleep(max(0.001, args.poll_interval))
    except KeyboardInterrupt:
        pass
    finally:
        if window_created:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
