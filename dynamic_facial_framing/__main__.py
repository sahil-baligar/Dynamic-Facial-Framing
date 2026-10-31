import argparse, json, sys
from . import reframe, classify_content, __version__


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dynamic_facial_framing", description="Horizontal video to vertical 9:16 with dynamic face framing.")
    ap.add_argument("input"); ap.add_argument("output", nargs="?")
    ap.add_argument("--start", type=float, default=0.0); ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--mode", default="auto", choices=["auto", "talking_head", "podcast", "gaming", "other"])
    ap.add_argument("--classify", action="store_true", help="only print the content classification as JSON")
    ap.add_argument("--size", default="1080x1920"); ap.add_argument("--crf", type=int, default=17)
    ap.add_argument("--version", action="version", version=__version__)
    a = ap.parse_args(argv)
    if a.classify:
        r = classify_content(a.input, a.start, a.duration)
        r["webcam"] = r["webcam"].as_list() if r["webcam"] else None
        print(json.dumps(r)); return 0
    if not a.output: ap.error("output is required")
    w, h = [int(v) for v in a.size.lower().split("x")]
    done = lambda x: print(f"\r{x * 100:5.1f}%", end="", file=sys.stderr, flush=True)
    r = reframe(a.input, a.output, a.start, a.duration, a.mode, (w, h), progress=done, crf=a.crf)
    print(file=sys.stderr); print(json.dumps(r)); return 0


if __name__ == "__main__":
    sys.exit(main())
