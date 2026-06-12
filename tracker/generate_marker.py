"""
generate_marker.py — Generate printable ArUco marker images.

Usage:
  python generate_marker.py              → generates marker ID 0
  python generate_marker.py 0 1 2 3      → generates markers 0-3
  python generate_marker.py --size 400   → larger image (default 300px)

Output: aruco_marker_<id>.png in the current directory.
Print the PNG and measure the black square's side length — that's your MARKER_SIZE_CM.
"""

import sys
import cv2


def generate(marker_id: int, image_size: int = 300, dict_name: str = "DICT_4X4_50") -> str:
    dict_id = getattr(cv2.aruco, dict_name)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

    # Generate marker with a white border (1 cell padding)
    img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, image_size)

    # Add white margin for printing
    margin = image_size // 6
    bordered = cv2.copyMakeBorder(
        img, margin, margin, margin, margin,
        cv2.BORDER_CONSTANT, value=255,
    )

    filename = f"aruco_marker_{marker_id}.png"
    cv2.imwrite(filename, bordered)
    return filename


def main() -> None:
    args = sys.argv[1:]

    size = 300
    ids = [0]

    # Parse --size flag
    if "--size" in args:
        idx = args.index("--size")
        size = int(args[idx + 1])
        args = args[:idx] + args[idx + 2:]

    if args:
        ids = [int(a) for a in args]

    for marker_id in ids:
        filename = generate(marker_id, size)
        print(f"[generate] saved {filename} (marker ID {marker_id}, {size}px)")

    print(f"\nPrint these PNGs and measure the black square's side length in cm.")
    print(f"Set MARKER_SIZE_CM to that measurement when running the tracker.")


if __name__ == "__main__":
    main()
