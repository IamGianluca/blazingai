from pathlib import Path
from typing import Protocol, Type

from loguru import logger


class KaggleAPITemplate(Protocol):
    def competition_download_cli(self, competition, path) -> None:
        pass


def download_comp_data(
    comp_name: str, path: Path, api: Type[KaggleAPITemplate]
) -> Path:
    """Download competition data from Kaggle, and save it to `path`
    destination.

    Args:
        api: Kaggle API client.
        comp_name: Short name of Kaggle competition.
        path: Destination.
    """
    api.competition_download_cli(competition=comp_name, path=path)
    return path / f"{comp_name}.zip"


def extract_comp_data(fname: Path, path: Path) -> None:
    """Inflate zip archive to `path` Destination.

    Args:
        fname: Zip archive to inflate.
        path: Destination.
    """
    if not path.exists():
        logger.info(f"Extracting {fname} to {path}")

        import zipfile

        from tqdm import tqdm

        with zipfile.ZipFile(fname) as zf:
            for member in tqdm(zf.infolist()):
                try:
                    zf.extract(member, path)
                except zipfile.error as e:
                    logger.warning(f"{member}: {e}")
    else:
        # NOTE: does not check that the content of `path` and the `fname`
        # zip archive matches
        logger.info(f"{path}: Skipping. Found local copy of extract archive")
