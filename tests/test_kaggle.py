import zipfile
from pathlib import Path

from blazingai.kaggle import download_comp_data, extract_comp_data


class KaggleAPIFake:
    def __init__(self) -> None:
        pass

    def competition_download_cli(self, competition, path) -> None:
        pass


def test_download_from_kaggle():
    result: Path = download_comp_data(
        api=KaggleAPIFake(), comp_name="test", path=Path("tmp")
    )
    assert result == Path("tmp/test.zip")


def test_extract_comp_data(tmp_path):
    # create empty folder
    p: Path = tmp_path / "sub"
    p.mkdir()

    assert len(list(p.iterdir())) == 0

    # create one file
    fname: Path = p / "file.txt"
    fname.write_text("content")

    assert len(list(p.iterdir())) == 1

    # create zip archive
    with zipfile.ZipFile(p / "file.zip", "w") as zf:
        zf.write(fname)

    assert len(list(p.iterdir())) == 2

    # extract zip archive
    extract_comp_data(fname=p / "file.zip", path=p / "plus.txt")

    assert len(list(p.iterdir())) == 3
