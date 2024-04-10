from pathlib import Path

import pytest
from pydicom import dcmread

from blazingai.vision.convert import convert_dicom2jpg


@pytest.fixture(scope="session")
def dicom_fpath(tmpdir_factory):
    p = Path.cwd() / "tests/data/1.dcm"
    dcm = dcmread(p)

    fpath = tmpdir_factory.mktemp("data").join("test.dcm")
    dcm.save_as(fpath)
    return Path(fpath)


@pytest.mark.slow
def test_happy_case(dicom_fpath, tmp_path):
    # given
    assert dicom_fpath.exists()

    # when
    in_path = dicom_fpath.parent
    convert_dicom2jpg(in_path=in_path, out_path=tmp_path)

    # then
    fname = dicom_fpath.stem + ".jpg"
    assert (tmp_path / fname).exists()


def test_create_out_path_if_not_exists(dicom_fpath, tmp_path):
    # given
    out_path = tmp_path / "new"
    assert not out_path.exists()

    # when
    in_path = dicom_fpath.parent
    convert_dicom2jpg(in_path=in_path, out_path=out_path)

    # then
    fname = dicom_fpath.stem + ".jpg"
    assert (out_path / fname).exists()


def test_in_path_recursive(dicom_fpath, tmp_path):
    # when
    in_path = dicom_fpath.parents[1]
    convert_dicom2jpg(in_path=in_path, out_path=tmp_path)

    # then
    fname = dicom_fpath.stem + ".jpg"
    assert (tmp_path / f"data0/{fname}").exists()


def test_out_path_not_exists(dicom_fpath, tmp_path):
    # when
    in_path = dicom_fpath.parents[1]
    out_path = tmp_path / "one"
    convert_dicom2jpg(in_path=in_path, out_path=out_path)

    # then
    fname = dicom_fpath.stem + ".jpg"
    assert (out_path / f"data0/{fname}").exists()
