import argparse
import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt
from ome_types import from_xml
from skimage import img_as_float
from skimage.transform import downscale_local_mean
from skimage.util.dtype import _convert
from tifffile import TiffReader, tifffile

from broadside.utils.arrays import square_concat
from broadside.utils.io import read_paths


def compute_inter_percentile_range(im: npt.NDArray, mid_chunk_size: int) -> float:
    assert im.ndim == 2
    h, w = im.shape

    h0 = max(0, h // 2 - mid_chunk_size // 2)
    h1 = min(h, h // 2 + mid_chunk_size // 2)
    w0 = max(0, w // 2 - mid_chunk_size // 2)
    w1 = min(w, w // 2 + mid_chunk_size // 2)

    im = im[h0:h1, w0:w1]
    p_lo, p_hi = np.percentile(im, (10, 90))
    return (p_hi - p_lo).item()


def downsample_with_mask(
    *,
    im: npt.NDArray,
    downsample: int,
    max_val: int,
    dtype: npt.DTypeLike,
    threshold_factor: float = 0.8,
    mid_chunk_size: int,
):
    # format image and mask
    assert im.ndim == 3
    h, w = im.shape[1:3]

    h0 = max(0, h // 2 - mid_chunk_size // 2)
    h1 = min(h, h // 2 + mid_chunk_size // 2)
    w0 = max(0, w // 2 - mid_chunk_size // 2)
    w1 = min(w, w // 2 + mid_chunk_size // 2)

    im = im[:, h0:h1, w0:w1]
    im = img_as_float(im)

    mask = np.logical_or.reduce(im >= max_val, axis=0)
    assert mask.ndim == 2
    mask = mask.astype(float)

    # transform image and mask
    im_scaled = downscale_local_mean(im, (1, downsample, downsample))
    mask = downscale_local_mean(mask, (downsample, downsample))

    threshold = threshold_factor / downsample
    im_scaled[:, mask >= threshold] = 0
    im_scaled.clip(0.0, 1.0, out=im_scaled)
    im_scaled = _convert(im_scaled, dtype)
    return im_scaled


def compute_iprs(stack_path: Path, *, ref_channel: int, mid_chunk_size: int):
    with TiffReader(stack_path) as reader:
        # get metadata
        n_pages = len(reader.pages)

        ome = from_xml(reader.ome_metadata, parser="lxml")
        n_channels = len(ome.images[0].pixels.channels)
        assert n_pages % n_channels == 0
        n_tiles_tot = n_pages // n_channels

        # compute page indexes
        page_inds = [(i * n_channels + ref_channel) for i in range(n_tiles_tot)]

        iprs = []
        for ind in page_inds:
            ipr = compute_inter_percentile_range(
                reader.pages[ind].asarray(), mid_chunk_size=mid_chunk_size
            )
            iprs.append(ipr)
        iprs = np.array(iprs)
    return iprs


def find_inds_of_k_maxima_across_lists(
    lists: list[list[float]], k: int
) -> list[list[int]]:
    lists_as_np = [np.array(l) for l in lists]
    flattened = np.concatenate(lists_as_np)

    if k > flattened.size:
        warnings.warn(
            f"Array too small: attempting to pick {k} elements from array with size {flattened.size}"
        )
        max_inds = np.arange(flattened.size).astype(int)
    elif k == flattened.size:
        max_inds = np.arange(flattened.size).astype(int)
    else:
        max_inds = np.argpartition(flattened, -k)[-k:]

    inds = []
    for i, l in enumerate(lists):
        offset = sum(len(_l) for _l in lists[:i])
        n = len(l)
        ind = max_inds[(offset <= max_inds) & (max_inds < (offset + n))] - offset
        inds.append(ind)

    return inds


def make_unmixing_mosaic(
    *,
    stack_paths: list[Path],
    ref_channel: int,
    downsample: int,
    n_tiles: int,
    mid_chunk_size: int,
    dst: Path,
):
    iprs = [
        compute_iprs(path, ref_channel=ref_channel, mid_chunk_size=mid_chunk_size)
        for path in stack_paths
    ]
    inds = find_inds_of_k_maxima_across_lists(iprs, n_tiles)

    downsampleds = []
    for stack_path, ind in zip(stack_paths, inds):
        with TiffReader(stack_path) as reader:
            ome = from_xml(reader.ome_metadata, parser="lxml")
            n_channels = len(ome.images[0].pixels.channels)
            max_val = 2 ** (ome.images[0].pixels.significant_bits) - 1
            dtype = reader.pages[0].dtype

            for i in ind:
                pages = reader.pages[i * n_channels : (i + 1) * n_channels]
                pages = [p.asarray() for p in pages]
                image = np.stack(pages)
                assert image.ndim == 3
                downsampled = downsample_with_mask(
                    im=image,
                    downsample=downsample,
                    max_val=max_val,
                    dtype=dtype,
                    mid_chunk_size=mid_chunk_size,
                )
                downsampleds.append(downsampled)

    mosaic = square_concat(downsampleds)
    tifffile.imwrite(dst, mosaic, compression="zlib")


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stacks-path", type=str, required=True)
    parser.add_argument("--ref-channel", type=int, required=True)
    parser.add_argument("--downsample", type=int, required=True)
    parser.add_argument("--n-tiles", type=int, required=True)
    parser.add_argument("--mid-chunk-size", type=int, required=True)
    parser.add_argument("--dst", type=str, required=True)

    args = parser.parse_args()

    stack_paths = read_paths(Path(args.stacks_path))
    make_unmixing_mosaic(
        stack_paths=stack_paths,
        ref_channel=args.ref_channel,
        downsample=args.downsample,
        n_tiles=args.n_tiles,
        mid_chunk_size=args.mid_chunk_size,
        dst=Path(args.dst),
    )


if __name__ == "__main__":
    make_unmixing_mosaic(
        stack_paths=[
            Path(r"/home/titanium/Downloads/nextflow-temp/PROSTATE-stack.tiff")
        ],
        ref_channel=0,
        downsample=4,
        n_tiles=49,
        mid_chunk_size=512,
        dst=Path(r"/home/titanium/Downloads/nextflow-temp/PROSTATE-mosaic.tiff"),
    )
