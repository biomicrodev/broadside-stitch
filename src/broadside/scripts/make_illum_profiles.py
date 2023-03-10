import argparse
from pathlib import Path

import tifffile
from ome_types import from_xml
from pybasic.compute import compute
from tifffile import TiffReader

from broadside.adjustments.hot_pixels import get_remove_hot_pixels_func
from broadside.utils.io import read_paths
from broadside.utils.parallel import dask_session


def _get_remove_hot_pixels_func(path: Path, *, dark_dir: Path):
    with TiffReader(path) as reader:
        ome = from_xml(reader.ome_metadata, parser="lxml")
        ts = ome.images[0].acquisition_date.timestamp()

    return get_remove_hot_pixels_func(ts, dark_dir=dark_dir)


def make_illumination_profiles(
    tile_paths: list[Path],
    *,
    flatfield_dst: Path,
    darkfield_dst: Path,
    compute_darkfield: bool,
    dark_dir: Path,
):
    mid_path = tile_paths[len(tile_paths) // 2]
    remove_hot_pixels = _get_remove_hot_pixels_func(mid_path, dark_dir=dark_dir)

    flatfield, darkfield = compute(
        tile_paths,
        iter_axes=[0],
        compute_darkfield=compute_darkfield,
        sort=compute_darkfield,
        verbose=False,
        func=remove_hot_pixels,
    )

    flatfield_dst.parent.mkdir(exist_ok=True, parents=True)
    tifffile.imwrite(flatfield_dst, flatfield, compression="zlib")

    darkfield_dst.parent.mkdir(exist_ok=True, parents=True)
    tifffile.imwrite(darkfield_dst, darkfield, compression="zlib")


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles-path", type=str, required=True)
    parser.add_argument("--flatfield-path", type=str, required=True)
    parser.add_argument("--darkfield-path", type=str, required=True)
    parser.add_argument("--darkfield", action="store_true")
    parser.add_argument("--no-darkfield", dest="darkfield", action="store_false")
    parser.set_defaults(darkfield=False)
    parser.add_argument("--dark-dir", type=str, required=True)

    parser.add_argument("--n-cpus", type=int, default=None)
    parser.add_argument("--memory-limit", type=str, default=None)
    parser.add_argument("--dask-report-filename", type=str, default=None)

    args = parser.parse_args()

    # parse nextflow-specific arguments
    tile_paths = read_paths(Path(args.tiles_path))

    # dask config
    with dask_session(
        memory_limit=args.memory_limit,
        n_cpus=args.n_cpus,
        dask_report_filename=args.dask_report_filename,
    ):
        make_illumination_profiles(
            tile_paths=tile_paths,
            flatfield_dst=Path(args.flatfield_path),
            darkfield_dst=Path(args.darkfield_path),
            compute_darkfield=args.darkfield,
            dark_dir=Path(args.dark_dir),
        )
