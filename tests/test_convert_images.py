from pathlib import Path

import pytest
from pydicom import dcmread

from ml.vision.convert import convert_dicom2jpg


@pytest.fixture(scope="session")
def dicom_file(tmpdir_factory):
    p = Path.cwd() / "tests/data/1.dcm"
    dcm = dcmread(p)

    fpath = tmpdir_factory.mktemp("data").join("test.dcm")
    dcm.save_as(fpath)
    return Path(fpath)


def test_happy_case(dicom_file, tmp_path):
    # given
    assert dicom_file.exists()

    # when
    in_path = dicom_file.parent
    convert_dicom2jpg(in_path=in_path, out_path=tmp_path)

    # then
    fname = dicom_file.stem + ".jpg"
    assert (tmp_path / fname).exists()


def test_create_out_path_if_not_exists(dicom_file, tmp_path):
    # given
    out_path = tmp_path / "new"
    assert not out_path.exists()

    # when
    in_path = dicom_file.parent
    convert_dicom2jpg(in_path=in_path, out_path=out_path)

    # then
    fname = dicom_file.stem + ".jpg"
    assert (out_path / fname).exists()


def test_in_path_recursive(dicom_file, tmp_path):
    # when
    in_path = dicom_file.parents[1]
    convert_dicom2jpg(in_path=in_path, out_path=tmp_path)

    # then
    fname = dicom_file.stem + ".jpg"
    assert (tmp_path / f"data0/{fname}").exists()


def test_out_path_not_exists(dicom_file, tmp_path):
    # when
    in_path = dicom_file.parents[1]
    out_path = tmp_path / "one"
    convert_dicom2jpg(in_path=in_path, out_path=out_path)

    # then
    fname = dicom_file.stem + ".jpg"
    assert (out_path / f"data0/{fname}").exists()
