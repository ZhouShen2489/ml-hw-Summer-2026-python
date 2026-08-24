"""Download the private Sofia ML regression competition files."""

from pathlib import Path

import kagglehub
from kagglehub.exceptions import UnauthenticatedError
from dotenv import load_dotenv


COMPETITION = "sofia-ml-regression-2026-summer-private"
DATA_DIR = Path(__file__).resolve().parent / "data"


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # This is the original KaggleHub download call, with a local output folder.
    try:
        path = kagglehub.competition_download(
            COMPETITION,
            output_dir=str(DATA_DIR),
        )
    except UnauthenticatedError as error:
        raise SystemExit(
            "Kaggle authentication is required. Run `kagglehub.login()` once, "
            "paste your Kaggle API token, then run this script again."
        ) from error

    print("Path to competition files:", path)


if __name__ == "__main__":
    main()
