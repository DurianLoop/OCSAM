from pathlib import Path


DEMO_PATH = Path(__file__).resolve().parents[1] / "demos" / "medical_sam_click_app.py"


def main() -> int:
    source = DEMO_PATH.read_text(encoding="utf-8")
    checks = {
        "point run mode select is present": 'id="pointRunMode"' in source,
        "point scope select is present": 'id="pointScope"' in source,
        "run mode input is wired": "pointRunModeInput" in source,
        "scope input is wired": "pointScopeInput" in source,
        "manual click mode does not auto-run": 'pointRunModeInput.value === "auto"' in source,
        "latest-only scope can trim request points": "points.slice(-1)" in source,
        "SAM3 defaults to manual cumulative": 'pointRunModeInput.value = "manual"' in source
        and 'pointScopeInput.value = "cumulative"' in source,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        print("Demo prompt control checks failed:")
        for name in failed:
            print(f"- {name}")
        return 1
    print("Demo prompt control checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
