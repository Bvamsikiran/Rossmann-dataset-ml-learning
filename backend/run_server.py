"""
Rossmann Sales ML Learning Lab - Server CLI Runner
Launches the FastAPI server using Uvicorn with port verification,
environment validation, and startup logging.
"""

import sys
import uvicorn
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.config import HOST, PORT, DATASET_DIR
from backend.data_loader import verify_dataset_directory


def main():
    print("=" * 70)
    print("  ROSSMANN SALES MACHINE LEARNING LAB - FASTAPI BACKEND")
    print("=" * 70)
    print(f"  Host:        {HOST}")
    print(f"  Port:        {PORT}")
    print(f"  Dataset Dir: {DATASET_DIR}")

    valid, msg = verify_dataset_directory()
    if valid:
        print(f"  Dataset Status: [READY] {msg}")
    else:
        print(f"  Dataset Status: [WARNING] {msg}")
    print("=" * 70)
    print(f"  Interactive API Docs: http://{HOST}:{PORT}/docs")
    print("=" * 70)

    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=True)


if __name__ == "__main__":
    main()
