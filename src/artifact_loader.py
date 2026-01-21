"""
Google Drive artifact loader for checkpoints and results.

This module provides functions to download and extract checkpoints.zip and results.zip
from a shared Google Drive folder. Uses gdown for downloading with automatic retry
and SSL fallback support.

Usage:
    from src.artifact_loader import ensure_artifacts
    
    # Download and extract if folders don't exist
    ensure_artifacts()
    
    # Force re-download even if folders exist
    ensure_artifacts(force=True)
    
    # Download only specific artifacts
    from src.artifact_loader import ensure_checkpoints, ensure_results
    ensure_checkpoints()
    ensure_results()
"""

import warnings
import zipfile
from pathlib import Path

try:
    import gdown
except ImportError:
    gdown = None

from src.paths import (
    PROJECT_ROOT,
    CHECKPOINTS_ROOT,
    RESULTS_ROOT,
    CHECKPOINTS_ZIP,
    RESULTS_ZIP,
    GDRIVE_CHECKPOINTS_ZIP_ID,
    GDRIVE_RESULTS_ZIP_ID,
    GDRIVE_FOLDER_URL,
)


class DownloadError(Exception):
    """Raised when a download fails."""
    pass


def _check_gdown_installed() -> None:
    """Check if gdown is installed and raise helpful error if not."""
    if gdown is None:
        raise ImportError(
            "gdown is not installed. Install it with:\n"
            "  pip install 'gdown>=5.0.0'\n"
            "Or update your environment:\n"
            "  conda env update -f environment_cpu.yml"
        )


def _get_download_url(file_id: str) -> str:
    """Get the direct download URL for a Google Drive file."""
    return f"https://drive.google.com/uc?id={file_id}"


def download_file(
    file_id: str,
    output_path: Path,
    description: str = "file",
    verify_ssl: bool = True,
) -> Path:
    """
    Download a file from Google Drive.
    
    Args:
        file_id: Google Drive file ID
        output_path: Local path to save the file
        description: Human-readable description for progress messages
        verify_ssl: Whether to verify SSL certificates (try True first, fallback to False)
    
    Returns:
        Path to the downloaded file
    
    Raises:
        DownloadError: If download fails after all retry attempts
    """
    _check_gdown_installed()
    
    url = _get_download_url(file_id)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {description} from Google Drive...")
    print(f"  URL: {url}")
    print(f"  Destination: {output_path}")
    
    # Try with SSL verification first
    if verify_ssl:
        try:
            result = gdown.download(url, str(output_path), quiet=False, verify=True)
            if result and Path(result).exists():
                print(f"  Successfully downloaded {description}")
                return Path(result)
        except Exception as e:
            if "SSL" in str(e) or "certificate" in str(e).lower():
                print(f"  SSL verification failed, retrying without verification...")
                verify_ssl = False
            else:
                raise DownloadError(f"Failed to download {description}: {e}") from e
    
    # Retry without SSL verification if needed
    if not verify_ssl:
        warnings.warn(
            "SSL certificate verification disabled. This is usually due to missing "
            "system certificates. Consider running:\n"
            "  /Applications/Python\\ 3.x/Install\\ Certificates.command\n"
            "or setting REQUESTS_CA_BUNDLE environment variable.",
            UserWarning,
        )
        try:
            # Suppress the InsecureRequestWarning
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Unverified HTTPS request")
                result = gdown.download(url, str(output_path), quiet=False, verify=False)
            if result and Path(result).exists():
                print(f"  Successfully downloaded {description}")
                return Path(result)
        except Exception as e:
            raise DownloadError(f"Failed to download {description}: {e}") from e
    
    raise DownloadError(
        f"Failed to download {description}.\n\n"
        f"Manual download instructions:\n"
        f"  1. Open: {GDRIVE_FOLDER_URL}\n"
        f"  2. Download the file manually\n"
        f"  3. Place it at: {output_path}"
    )


def extract_zip(zip_path: Path, extract_to: Path, description: str = "archive") -> Path:
    """
    Extract a zip file to a directory.
    
    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract to (will be created if needed)
        description: Human-readable description for progress messages
    
    Returns:
        Path to the extraction directory
    """
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    
    print(f"Extracting {description}...")
    print(f"  Source: {zip_path}")
    print(f"  Destination: {extract_to}")
    
    # Create parent directory if needed
    extract_to.parent.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Get total size for progress
        total_size = sum(info.file_size for info in zf.infolist())
        print(f"  Total size: {total_size / (1024*1024):.1f} MB")
        
        # Extract all files
        zf.extractall(extract_to.parent)
    
    print(f"  Successfully extracted {description}")
    return extract_to


def ensure_checkpoints(force: bool = False, verify_ssl: bool = True) -> Path:
    """
    Ensure checkpoints directory exists, downloading if necessary.
    
    Args:
        force: If True, re-download even if checkpoints/ exists
        verify_ssl: Whether to verify SSL certificates
    
    Returns:
        Path to checkpoints directory
    """
    if CHECKPOINTS_ROOT.exists() and not force:
        print(f"Checkpoints directory already exists: {CHECKPOINTS_ROOT}")
        return CHECKPOINTS_ROOT
    
    # Download checkpoints.zip if not present or force
    if not CHECKPOINTS_ZIP.exists() or force:
        download_file(
            GDRIVE_CHECKPOINTS_ZIP_ID,
            CHECKPOINTS_ZIP,
            description="checkpoints.zip",
            verify_ssl=verify_ssl,
        )
    
    # Extract checkpoints.zip
    extract_zip(CHECKPOINTS_ZIP, CHECKPOINTS_ROOT, description="checkpoints")
    
    # Optionally remove zip after extraction
    # CHECKPOINTS_ZIP.unlink()
    
    return CHECKPOINTS_ROOT


def ensure_results(force: bool = False, verify_ssl: bool = True) -> Path:
    """
    Ensure results directory exists, downloading if necessary.
    
    Args:
        force: If True, re-download even if results/ exists
        verify_ssl: Whether to verify SSL certificates
    
    Returns:
        Path to results directory
    """
    if RESULTS_ROOT.exists() and not force:
        print(f"Results directory already exists: {RESULTS_ROOT}")
        return RESULTS_ROOT
    
    # Download results.zip if not present or force
    if not RESULTS_ZIP.exists() or force:
        download_file(
            GDRIVE_RESULTS_ZIP_ID,
            RESULTS_ZIP,
            description="results.zip",
            verify_ssl=verify_ssl,
        )
    
    # Extract results.zip
    extract_zip(RESULTS_ZIP, RESULTS_ROOT, description="results")
    
    # Optionally remove zip after extraction
    # RESULTS_ZIP.unlink()
    
    return RESULTS_ROOT


def ensure_artifacts(force: bool = False, verify_ssl: bool = True) -> tuple[Path, Path]:
    """
    Ensure both checkpoints and results directories exist.
    
    This is the main entry point for ensuring all required artifacts are available.
    Downloads and extracts from Google Drive if necessary.
    
    Args:
        force: If True, re-download even if directories exist
        verify_ssl: Whether to verify SSL certificates (set False if you have SSL issues)
    
    Returns:
        Tuple of (checkpoints_path, results_path)
    
    Example:
        >>> from src.artifact_loader import ensure_artifacts
        >>> checkpoints, results = ensure_artifacts()
        >>> # Now checkpoints/ and results/ are available
    """
    print("=" * 60)
    print("Ensuring artifact directories are available...")
    print("=" * 60)
    
    checkpoints_path = ensure_checkpoints(force=force, verify_ssl=verify_ssl)
    print()
    results_path = ensure_results(force=force, verify_ssl=verify_ssl)
    
    print()
    print("=" * 60)
    print("Artifacts ready!")
    print(f"  Checkpoints: {checkpoints_path}")
    print(f"  Results: {results_path}")
    print("=" * 60)
    
    return checkpoints_path, results_path


def print_manual_download_instructions() -> None:
    """Print manual download instructions for when automatic download fails."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        MANUAL DOWNLOAD INSTRUCTIONS                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  If automatic download fails, you can download manually:                     ║
║                                                                              ║
║  1. Open the shared Google Drive folder:                                     ║
║     {url:<66} ║
║                                                                              ║
║  2. Download 'checkpoints.zip' and 'results.zip'                             ║
║                                                                              ║
║  3. Place the zip files in the project root:                                 ║
║     {root:<66} ║
║                                                                              ║
║  4. Extract the files:                                                       ║
║     python -c "from src.artifact_loader import extract_zip; \\               ║
║                extract_zip('checkpoints.zip', 'checkpoints'); \\             ║
║                extract_zip('results.zip', 'results')"                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""".format(url=GDRIVE_FOLDER_URL, root=str(PROJECT_ROOT)))


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download checkpoints and results from Google Drive"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if directories exist",
    )
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="Disable SSL certificate verification",
    )
    parser.add_argument(
        "--checkpoints-only",
        action="store_true",
        help="Only download checkpoints",
    )
    parser.add_argument(
        "--results-only",
        action="store_true",
        help="Only download results",
    )
    parser.add_argument(
        "--instructions",
        action="store_true",
        help="Print manual download instructions",
    )
    
    args = parser.parse_args()
    
    if args.instructions:
        print_manual_download_instructions()
    elif args.checkpoints_only:
        ensure_checkpoints(force=args.force, verify_ssl=not args.no_verify_ssl)
    elif args.results_only:
        ensure_results(force=args.force, verify_ssl=not args.no_verify_ssl)
    else:
        ensure_artifacts(force=args.force, verify_ssl=not args.no_verify_ssl)
