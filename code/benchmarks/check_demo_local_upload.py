from pathlib import Path


DEMO_PATH = Path(__file__).resolve().parents[1] / "demos" / "medical_sam_click_app.py"


def main() -> int:
    source = DEMO_PATH.read_text(encoding="utf-8")
    checks = {
        "UploadFile import is present": "UploadFile" in source,
        "upload endpoint is present": '@app.post("/upload_image")' in source,
        "uploaded examples are tracked": "uploaded_samples" in source,
        "upload directory is under demo outputs": "demo_outputs" in source and "uploads" in source,
        "upload button exists": 'id="uploadLocalImage"' in source,
        "local upload request is wired": 'fetch("/upload_image"' in source,
        "uploaded sample loads into canvas": "loadUploadedImage" in source,
        "examples include uploaded samples": "uploaded_examples" in source,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        print("Demo local upload checks failed:")
        for name in failed:
            print(f"- {name}")
        return 1
    print("Demo local upload checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
