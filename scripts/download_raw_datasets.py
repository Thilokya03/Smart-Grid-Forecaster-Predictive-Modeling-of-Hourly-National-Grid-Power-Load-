from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import kagglehub


# =============================================================================
# Configuration
# =============================================================================

GITHUB_REPOSITORY = "Thilokya03/Sri-Lanka-Holiday-Dataset"
HOLIDAY_ASSET_NAME = "Sri_Lanka_all_dates.csv"

KAGGLE_DATASET = "ziya07/hourly-power-load-and-climate-data"

# This script must be inside the project's scripts/ directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

HOLIDAY_FILE = RAW_DATA_DIR / HOLIDAY_ASSET_NAME
HOLIDAY_VERSION_FILE = RAW_DATA_DIR / ".holiday_release.json"

POWER_DATA_DIR = RAW_DATA_DIR / "hourly_power_load_and_climate"

GITHUB_API_URL = (
    f"https://api.github.com/repos/"
    f"{GITHUB_REPOSITORY}/releases/latest"
)


# =============================================================================
# Common helper functions
# =============================================================================

def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def github_headers() -> dict[str, str]:
    """Create request headers for the GitHub API."""

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "smart-grid-dataset-updater",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # Optional when running inside GitHub Actions.
    github_token = os.getenv("GITHUB_TOKEN")

    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    return headers


# =============================================================================
# Holiday dataset
# =============================================================================

def get_latest_github_release() -> dict:
    """Get information about the latest GitHub Release."""

    request = Request(
        GITHUB_API_URL,
        headers=github_headers(),
    )

    with urlopen(request, timeout=30) as response:
        return json.load(response)


def find_holiday_asset(release: dict) -> dict:
    """Find the required holiday CSV in the latest release."""

    for asset in release.get("assets", []):
        if asset.get("name") == HOLIDAY_ASSET_NAME:
            return asset

    raise RuntimeError(
        f"{HOLIDAY_ASSET_NAME} was not found in the latest release."
    )


def load_holiday_version() -> dict:
    """Read information about the locally downloaded release."""

    if not HOLIDAY_VERSION_FILE.exists():
        return {}

    try:
        with HOLIDAY_VERSION_FILE.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}


def holiday_dataset_is_latest(
    release: dict,
    asset: dict,
    local_version: dict,
) -> bool:
    """Check whether the existing holiday CSV is the latest release."""

    if not HOLIDAY_FILE.exists():
        return False

    if local_version.get("tag_name") != release.get("tag_name"):
        return False

    if local_version.get("asset_id") != asset.get("id"):
        return False

    saved_hash = local_version.get("sha256")

    if not saved_hash:
        return False

    return calculate_sha256(HOLIDAY_FILE) == saved_hash


def download_holiday_asset(asset: dict) -> None:
    """Download and safely replace the holiday CSV."""

    download_url = asset.get("browser_download_url")

    if not download_url:
        raise RuntimeError("The release asset has no download URL.")

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    temporary_file = HOLIDAY_FILE.with_suffix(".csv.download")

    request = Request(
        download_url,
        headers=github_headers(),
    )

    try:
        with urlopen(request, timeout=120) as response:
            with temporary_file.open("wb") as output:
                shutil.copyfileobj(response, output)

        expected_size = asset.get("size")

        if (
            expected_size is not None
            and temporary_file.stat().st_size != expected_size
        ):
            raise RuntimeError(
                "The downloaded holiday CSV has an unexpected size."
            )

        remote_digest = asset.get("digest")

        if (
            isinstance(remote_digest, str)
            and remote_digest.startswith("sha256:")
        ):
            expected_hash = remote_digest.removeprefix("sha256:")
            downloaded_hash = calculate_sha256(temporary_file)

            if downloaded_hash != expected_hash:
                raise RuntimeError(
                    "The holiday CSV failed SHA-256 verification."
                )

        temporary_file.replace(HOLIDAY_FILE)

    except Exception:
        temporary_file.unlink(missing_ok=True)
        raise


def save_holiday_version(release: dict, asset: dict) -> None:
    """Save information about the downloaded holiday release."""

    version_data = {
        "repository": GITHUB_REPOSITORY,
        "asset_name": HOLIDAY_ASSET_NAME,
        "tag_name": release.get("tag_name"),
        "release_name": release.get("name"),
        "published_at": release.get("published_at"),
        "asset_id": asset.get("id"),
        "asset_updated_at": asset.get("updated_at"),
        "sha256": calculate_sha256(HOLIDAY_FILE),
    }

    temporary_file = HOLIDAY_VERSION_FILE.with_suffix(".json.tmp")

    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(version_data, file, indent=2)
        file.write("\n")

    temporary_file.replace(HOLIDAY_VERSION_FILE)


def update_holiday_dataset() -> None:
    """Check and download the latest holiday dataset release."""

    print("\n[1/2] Checking Sri Lanka holiday dataset...")

    release = get_latest_github_release()
    asset = find_holiday_asset(release)
    local_version = load_holiday_version()

    latest_tag = release.get("tag_name", "unknown")

    if holiday_dataset_is_latest(
        release,
        asset,
        local_version,
    ):
        print(f"Holiday dataset is already latest: {latest_tag}")
        print(f"Location: {HOLIDAY_FILE}")
        return

    print(f"New holiday release found: {latest_tag}")
    print("Downloading holiday CSV...")

    download_holiday_asset(asset)
    save_holiday_version(release, asset)

    print("Holiday dataset updated successfully.")
    print(f"Location: {HOLIDAY_FILE}")


# =============================================================================
# Kaggle power-load dataset
# =============================================================================

def update_power_load_dataset() -> None:
    """Check and download the latest Kaggle dataset version."""

    print("\n[2/2] Checking Kaggle power-load dataset...")

    POWER_DATA_DIR.mkdir(parents=True, exist_ok=True)

    downloaded_path = kagglehub.dataset_download(
        KAGGLE_DATASET,
        output_dir=str(POWER_DATA_DIR),
        force_download=False,
    )

    print("Kaggle dataset check completed.")
    print(f"Location: {downloaded_path}")


# =============================================================================
# Main function
# =============================================================================

def main() -> int:
    """Update both raw datasets."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    errors = []

    try:
        update_holiday_dataset()
    except (HTTPError, URLError, OSError, RuntimeError) as error:
        errors.append(f"Holiday dataset: {error}")
        print(
            f"Holiday dataset update failed: {error}",
            file=sys.stderr,
        )

    try:
        update_power_load_dataset()
    except Exception as error:
        errors.append(f"Kaggle dataset: {error}")
        print(
            f"Kaggle dataset update failed: {error}",
            file=sys.stderr,
        )

    print("\nDataset update process completed.")

    if errors:
        print("\nSome datasets could not be updated:", file=sys.stderr)

        for error in errors:
            print(f"- {error}", file=sys.stderr)

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())