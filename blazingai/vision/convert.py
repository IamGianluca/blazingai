from functools import partial
from pathlib import Path
from typing import List

import numpy as np
import pydicom
from joblib import delayed, Parallel
from loguru import logger
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm import tqdm


def convert_dicom2jpg(in_path: Path, out_path: Path) -> None:
    dirlist = list(in_path.iterdir())
    dirlist = [d for d in dirlist if d.is_dir()]
    if dirlist:
        _parallel_by_dir(dirs=dirlist, in_path=in_path, out_path=out_path)
    else:
        _parallel_by_file(in_path=in_path, out_path=out_path)


def _parallel_by_dir(dirs: List, in_path: Path, out_path: Path) -> None:
    f = partial(_convert_one_dir, in_path=in_path, out_path=out_path)
    Parallel(n_jobs=-1, prefer="threads")(
        delayed(f)(d) for d in tqdm(dirs, desc="Processing folders")
    )


def _convert_one_dir(dir_path: Path, in_path: Path, out_path: Path):
    img_paths = list(dir_path.rglob("*.dcm"))
    for img_path in img_paths:
        _convert_one_dicom_img(
            in_path=in_path, out_path=out_path, img_path=img_path
        )


def _parallel_by_file(in_path: Path, out_path: Path) -> None:
    img_paths = list(in_path.rglob("*.dcm"))
    logger.info(
        f"Converting {len(img_paths):,} DICOM files in {in_path} to JPG format in {out_path}"
    )

    f = partial(_convert_one_dicom_img, in_path=in_path, out_path=out_path)
    Parallel(n_jobs=-1)(
        delayed(f)(p) for p in tqdm(img_paths, desc="Processing Images")
    )


def _convert_one_dicom_img(
    img_path: Path, in_path: Path, out_path: Path
) -> None:
    img_array = dicom_to_numpy(
        img_path=img_path, voi_lut=True, fix_monochrome=True
    )
    img = Image.fromarray(img_array)

    fpath = Path(
        str(img_path)
        .replace(str(in_path), str(out_path))
        .replace("dcm", "jpg")
    )
    parent = fpath.parents[0]
    if not parent.exists():
        parent.mkdir(parents=True)

    img.save(fpath)


def dicom_to_numpy(
    img_path: Path, voi_lut: bool = True, fix_monochrome: bool = True
) -> np.ndarray:
    # credits: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = pydicom.read_file(img_path)

    # VOI LUT (if available by DICOM device) is used to transform
    # raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    return data
