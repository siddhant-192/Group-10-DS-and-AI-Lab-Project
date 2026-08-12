from pathlib import Path
import hashlib
import json
import gdown

# Official BIRD Mini-Dev package
FILE_ID = "13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG"

SOURCE_URL = (
    "https://drive.google.com/file/d/"
    "13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG/view"
)

# Google Drive location
OUTPUT_DIR = Path("/content/drive/MyDrive/bird_minidev")
OUTPUT_FILE = OUTPUT_DIR / "bird_minidev.zip"
MANIFEST_FILE = OUTPUT_DIR / "source_manifest.json"


def sha256_file(path):
    sha = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)

    return sha.hexdigest()


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # If already downloaded, verify it
    # --------------------------------------------------

    if OUTPUT_FILE.exists():

        print(f"Dataset already exists:")
        print(OUTPUT_FILE)

        if not MANIFEST_FILE.exists():
            raise RuntimeError(
                "Dataset exists but source_manifest.json is missing."
            )

        with open(MANIFEST_FILE) as f:
            manifest = json.load(f)

        expected_sha = manifest["sha256"]
        actual_sha = sha256_file(OUTPUT_FILE)

        print(f"\nExpected SHA-256: {expected_sha}")
        print(f"Actual SHA-256:   {actual_sha}")

        if actual_sha != expected_sha:
            raise RuntimeError(
                "SHA-256 mismatch! The dataset file has changed."
            )

        print("\nSHA-256 verification: PASS")
        return

    # --------------------------------------------------
    # Download
    # --------------------------------------------------

    print("Downloading official BIRD Mini-Dev package...")
    print(f"Source: {SOURCE_URL}")
    print(f"Output: {OUTPUT_FILE}")

    gdown.download(
        id=FILE_ID,
        output=str(OUTPUT_FILE),
        quiet=False
    )

    # --------------------------------------------------
    # Verify download actually produced a file
    # --------------------------------------------------

    if not OUTPUT_FILE.exists():
        raise RuntimeError("Download failed: file was not created.")

    file_size = OUTPUT_FILE.stat().st_size

    print(f"\nDownloaded file size: {file_size / (1024 * 1024):.2f} MB")

    if file_size < 10000:
        raise RuntimeError(
            "Downloaded file is suspiciously small. "
            "The Google Drive download probably failed."
        )

    # --------------------------------------------------
    # Calculate SHA-256
    # --------------------------------------------------

    checksum = sha256_file(OUTPUT_FILE)

    print(f"SHA-256: {checksum}")

    # --------------------------------------------------
    # Create provenance manifest
    # --------------------------------------------------

    manifest = {
        "dataset": "BIRD Mini-Dev",
        "release": "Original Mini-Dev",
        "examples": 500,
        "databases": 11,
        "database_format_used": "SQLite",
        "source_url": SOURCE_URL,
        "google_drive_file_id": FILE_ID,
        "file": OUTPUT_FILE.name,
        "sha256": checksum
    }

    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nSource manifest created:")
    print(MANIFEST_FILE)


if __name__ == "__main__":
    main()