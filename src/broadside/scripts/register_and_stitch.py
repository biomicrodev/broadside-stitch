import argparse
from pathlib import Path

from ashlar.scripts.ashlar import process_single

from broadside.utils.io import read_paths


def register_and_stitch(
    *,
    output: Path,
    stacks_path: Path,
    align_channel: int,
    filter_sigma: float,
    max_shift_um: float,
    tile_size: int,
):
    stack_paths = read_paths(stacks_path)
    aligner_args = dict(
        filter_sigma=filter_sigma,
        max_shift=max_shift_um,
        channel=align_channel,
    )
    mosaic_args = dict(tile_size=tile_size)

    process_single(
        filepaths=[str(p) for p in stack_paths],
        output_path_format=str(output),
        flip_x=False,
        flip_y=False,
        ffp_paths=None,
        dfp_paths=None,
        aligner_args=aligner_args,
        mosaic_args=mosaic_args,
        pyramid=True,
        quiet=True,
    )


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--stacks-path", type=str, required=True)
    parser.add_argument("--align-channel", type=int, required=True)
    parser.add_argument("--filter-sigma", type=float, required=True)
    parser.add_argument("--maximum-shift", type=float, required=True)
    parser.add_argument("--tile-size", type=int, required=True)

    args = parser.parse_args()

    register_and_stitch(
        output=Path(args.output),
        stacks_path=Path(args.stacks_path),
        align_channel=args.align_channel,
        filter_sigma=args.filter_sigma,
        max_shift_um=args.maximum_shift,
        tile_size=args.tile_size,
    )
