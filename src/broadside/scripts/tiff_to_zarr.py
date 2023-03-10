import argparse
from pathlib import Path

import dask.array as da
import zarr
from numcodecs import Blosc
from ome_zarr.io import parse_url
from ome_zarr.writer import write_multiscale
from tifffile import tifffile


def tiff_to_zarr(*, src: Path, dst: Path, tile_size: int):
    zgroup = zarr.open(tifffile.imread(src, aszarr=True), mode="r")
    pyramid_src = [
        zgroup[int(dataset["path"])]
        for dataset in zgroup.attrs["multiscales"][0]["datasets"]
    ]

    # create destination zarr full of empty arrays first, using write_multiscale's
    # metadata writing for convenience
    # the downside though is that the contents of the array have to be loaded in RAM
    store_dst = parse_url(dst, mode="w").store
    root_dst = zarr.group(store=store_dst, overwrite=True)
    root_dst.attrs.clear()

    empty_pyramid = [
        da.zeros(shape=layer.shape, dtype=layer.dtype) for layer in pyramid_src
    ]

    write_multiscale(
        pyramid=empty_pyramid,
        group=root_dst,
        axes=["c", "y", "x"],
        chunks=(1, tile_size, tile_size),
        storage_options=dict(compressor=Blosc()),
    )

    layer_src: zarr.Array
    for layer_src, key in zip(pyramid_src, root_dst.keys()):
        # FIXME note that this loads the entire pyramid layer into memory
        root_dst[key][:] = layer_src


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    parser.add_argument("--tile-size", type=int, required=True)

    args = parser.parse_args()

    tiff_to_zarr(
        src=Path(args.src),
        dst=Path(args.dst),
        tile_size=args.tile_size,
    )
