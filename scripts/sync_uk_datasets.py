import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests


# ============================================================
# Configuration
# ============================================================

OWNER = "Thilokya03"
REPO = "UK-Calendar-Holiday-Events-Dataset"

LATEST_RELEASE_API = (
    f"https://api.github.com/repos/{OWNER}/{REPO}/releases/latest"
)

DATASET_NAMES = [
    "full_calendar_features.csv",
    "uk_economic_features_daily.csv",
]

CHECKSUM_FILE = "SHA256SUMS.txt"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOWNLOAD_DIR = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "uk_features"
)

METADATA_FILE = DOWNLOAD_DIR / "release_metadata.json"


# ============================================================
# GitHub headers
# ============================================================

def get_headers():
    """
    Create headers for GitHub API requests.

    GITHUB_TOKEN is optional because the repository is public.
    It can still be supplied to increase API rate limits.
    """

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "Smart-Grid-Forecaster",
    }

    github_token = os.getenv("GITHUB_TOKEN")

    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    return headers


# ============================================================
# SHA-256
# ============================================================

def calculate_sha256(file_path: Path) -> str:
    """
    Calculate the SHA-256 checksum of a local file.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# Latest release
# ============================================================

def get_latest_release():
    """
    Get information about the latest GitHub release.
    """

    print("\nChecking latest GitHub release...")

    response = requests.get(
        LATEST_RELEASE_API,
        headers=get_headers(),
        timeout=30,
    )

    response.raise_for_status()

    release = response.json()

    print(f"Latest release : {release['tag_name']}")
    print(f"Release name   : {release.get('name', 'N/A')}")

    return release


# ============================================================
# Find release asset
# ============================================================

def find_asset(release, asset_name):
    """
    Find a specific asset inside the GitHub release.
    """

    for asset in release.get("assets", []):
        if asset["name"] == asset_name:
            return asset

    return None


# ============================================================
# Download text asset
# ============================================================

def download_text_asset(asset):
    """
    Download a small text release asset such as SHA256SUMS.txt.
    """

    response = requests.get(
        asset["browser_download_url"],
        headers=get_headers(),
        timeout=30,
    )

    response.raise_for_status()

    return response.text


# ============================================================
# Parse SHA256SUMS
# ============================================================

def parse_checksums(content):
    """
    Parse SHA256SUMS.txt.

    Expected format:

    hash  full_calendar_features.csv
    hash  uk_economic_features_daily.csv
    """

    checksums = {}

    for line in content.splitlines():

        line = line.strip()

        if not line:
            continue

        parts = line.split(maxsplit=1)

        if len(parts) != 2:
            continue

        checksum = parts[0].strip()

        filename = parts[1].strip()

        # Handles formats such as:
        # hash  file.csv
        # hash *file.csv

        filename = filename.lstrip("*")

        checksums[filename] = checksum

    return checksums


# ============================================================
# Download dataset safely
# ============================================================

def download_dataset(asset, destination, expected_checksum):
    """
    Download a dataset to a temporary file.

    The old file is replaced only after checksum verification.
    """

    temp_file = destination.with_suffix(
        destination.suffix + ".download"
    )

    print(f"Downloading {destination.name}...")

    try:

        with requests.get(
            asset["browser_download_url"],
            headers=get_headers(),
            stream=True,
            timeout=120,
        ) as response:

            response.raise_for_status()

            with open(temp_file, "wb") as file:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if chunk:
                        file.write(chunk)

        # ----------------------------------------------------
        # Verify downloaded file
        # ----------------------------------------------------

        downloaded_checksum = calculate_sha256(temp_file)

        if downloaded_checksum.lower() != expected_checksum.lower():

            raise ValueError(
                f"Checksum verification failed for "
                f"{destination.name}\n"
                f"Expected: {expected_checksum}\n"
                f"Actual:   {downloaded_checksum}"
            )

        # ----------------------------------------------------
        # Safe replacement
        # ----------------------------------------------------

        os.replace(temp_file, destination)

        print(
            f"Updated successfully: {destination.name}"
        )

        return True

    except Exception:

        if temp_file.exists():
            temp_file.unlink()

        raise


# ============================================================
# Save release metadata
# ============================================================

def save_metadata(release, checksums):
    """
    Save information about the release currently being used.
    """

    metadata = {
        "repository": f"{OWNER}/{REPO}",
        "release_tag": release["tag_name"],
        "release_name": release.get("name"),
        "published_at": release.get("published_at"),
        "checked_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "datasets": {
            filename: {
                "sha256": checksums.get(filename)
            }
            for filename in DATASET_NAMES
        },
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )


# ============================================================
# Main synchronization function
# ============================================================

def sync_datasets():
    """
    Synchronize local datasets with the latest GitHub release.
    """

    print("=" * 60)
    print("UK DATASET RELEASE SYNCHRONIZER")
    print("=" * 60)

    DOWNLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Get latest release
    # --------------------------------------------------------

    release = get_latest_release()

    # --------------------------------------------------------
    # 2. Find checksum asset
    # --------------------------------------------------------

    checksum_asset = find_asset(
        release,
        CHECKSUM_FILE,
    )

    if checksum_asset is None:

        raise RuntimeError(
            f"{CHECKSUM_FILE} was not found in "
            f"release {release['tag_name']}."
        )

    # --------------------------------------------------------
    # 3. Download checksum list
    # --------------------------------------------------------

    checksum_content = download_text_asset(
        checksum_asset
    )

    checksums = parse_checksums(
        checksum_content
    )

    updated_files = []

    # --------------------------------------------------------
    # 4. Check each required dataset
    # --------------------------------------------------------

    for dataset_name in DATASET_NAMES:

        print("\n" + "-" * 60)
        print(f"Checking: {dataset_name}")

        expected_checksum = checksums.get(
            dataset_name
        )

        if not expected_checksum:

            raise RuntimeError(
                f"No SHA-256 checksum found for "
                f"{dataset_name}"
            )

        destination = (
            DOWNLOAD_DIR / dataset_name
        )

        # ----------------------------------------------------
        # Check local file
        # ----------------------------------------------------

        if destination.exists():

            local_checksum = calculate_sha256(
                destination
            )

            print(
                f"Local SHA256  : {local_checksum}"
            )

            print(
                f"Latest SHA256 : {expected_checksum}"
            )

            if (
                local_checksum.lower()
                == expected_checksum.lower()
            ):

                print(
                    "File is already the latest version."
                )

                continue

            print(
                "Local file is outdated or modified."
            )

        else:

            print(
                "Local file does not exist."
            )

        # ----------------------------------------------------
        # Find dataset asset
        # ----------------------------------------------------

        asset = find_asset(
            release,
            dataset_name,
        )

        if asset is None:

            raise RuntimeError(
                f"{dataset_name} was not found "
                f"in release {release['tag_name']}."
            )

        # ----------------------------------------------------
        # Download and replace
        # ----------------------------------------------------

        download_dataset(
            asset=asset,
            destination=destination,
            expected_checksum=expected_checksum,
        )

        updated_files.append(
            dataset_name
        )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    save_metadata(
        release,
        checksums,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)

    print(
        f"Release in use: {release['tag_name']}"
    )

    if updated_files:

        print("Updated files:")

        for file_name in updated_files:
            print(f"  - {file_name}")

    else:

        print(
            "All datasets are already up to date."
        )

    print("=" * 60)

    return {
        "release": release["tag_name"],
        "updated_files": updated_files,
        "download_directory": str(
            DOWNLOAD_DIR
        ),
    }


# ============================================================
# Run directly
# ============================================================

if __name__ == "__main__":

    try:

        sync_datasets()

    except requests.RequestException as error:

        print(
            "\nGitHub/network request failed:"
        )

        print(error)

        raise SystemExit(1)

    except Exception as error:

        print("\nDataset synchronization failed:")

        print(error)

        raise SystemExit(1)