import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import dask.array as da
import numpy as np
import numpy.typing as npt
import zarr
from PIL import ImageFont, Image, ImageDraw
from ome_types import from_xml
from ome_types.model import Pixels
from pint import Quantity
from skimage import img_as_float, img_as_ubyte
from skimage.exposure import rescale_intensity
from skimage.filters.thresholding import threshold_otsu
from skimage.transform import downscale_local_mean
from tifffile import tifffile, TiffReader

from broadside.utils.geoms import Point2D
from broadside.utils.io import read_paths
from broadside.utils.units import default_distance_units, ureg


@dataclass(frozen=True)
class Thumbnail:
    image: npt.NDArray
    cycle: str

    @property
    def downscaled(self):
        return downscale_local_mean(self.image, factors=8)

    @property
    def threshold(self):
        return threshold_otsu(self.downscaled)

    @property
    def tissue_mask(self):
        return self.downscaled > self.threshold


def _get_pixel_size(pixels: Pixels) -> Quantity:
    x_units = getattr(ureg, pixels.physical_size_x_unit.name.lower())
    y_units = getattr(ureg, pixels.physical_size_y_unit.name.lower())

    pixel_size_x: Quantity = pixels.physical_size_x * x_units
    pixel_size_y: Quantity = pixels.physical_size_y * y_units

    if not np.allclose(pixel_size_x, pixel_size_y):
        warnings.warn("Non-square pixel size not supported")

    avg_pixel_size = (pixel_size_x / 2) + (pixel_size_y / 2)
    return avg_pixel_size


def _get_position(pixels: Pixels):
    first_plane = pixels.planes[0]
    x = first_plane.position_x * getattr(ureg, first_plane.position_x_unit.name.lower())
    y = first_plane.position_y * getattr(ureg, first_plane.position_y_unit.name.lower())
    return Point2D(x=x, y=y)


@dataclass
class SceneStack:
    thumbnails: list[Thumbnail]
    offset_px: Point2D

    @classmethod
    def from_path(cls, path: Path, ref_channel: int, vis_level: int):
        with TiffReader(path) as reader:
            ome = from_xml(reader.ome_metadata, parser="lxml")
            assert len(ome.images) >= 1
            pixels = ome.images[0].pixels
            pixel_size = _get_pixel_size(pixels)
            base_position = _get_position(pixels)

        # FIXME: measure difference in ome.tiff vs ome.zarr
        # why is reading ome.tiff so dang slow?
        zgroup = zarr.open(store=tifffile.imread(path, aszarr=True), mode="r")
        pyramid = [
            da.from_zarr(zgroup[int(dataset["path"])])
            for dataset in zgroup.attrs["multiscales"][0]["datasets"]
        ]

        channel_params = [
            {**json.loads(c.name), "index": i} for i, c in enumerate(pixels.channels)
        ]
        channel_params = [c for c in channel_params if c["channel"] == ref_channel]

        # collect thumbnails
        thumbnails = []
        for ch in channel_params:
            image = pyramid[vis_level][ch["index"]].compute()
            image = img_as_float(image)
            thumbnails.append(Thumbnail(image=image, cycle=ch["cycle"]))

        # make sure all thumbnails are of the same shape
        # this should be guaranteed by ashlar, but we check anyway so we can sleep at night
        tnail_shapes = [tnail.image.shape for tnail in thumbnails]
        assert len(set(tnail_shapes)) == 1

        # compute scales and translations
        base_level = 0
        downscale = int(
            round(
                pyramid[base_level][ref_channel].shape[0]
                / pyramid[vis_level][ref_channel].shape[0]
            )
        )
        vis_shape = Point2D(
            y=pyramid[vis_level][ref_channel].shape[0],
            x=pyramid[vis_level][ref_channel].shape[1],
        )

        mpp = pixel_size.to(default_distance_units).magnitude
        base_center_px = Point2D(
            y=(base_position.y.to(default_distance_units)).magnitude / mpp,
            x=(base_position.x.to(default_distance_units)).magnitude / mpp,
        )
        vis_center_px = base_center_px / downscale
        offset_px = vis_center_px - vis_shape / 2
        offset_px = Point2D(x=int(round(offset_px.x)), y=int(round(offset_px.y)))

        return cls(thumbnails=thumbnails, offset_px=offset_px)

    @property
    def n_rounds(self):
        return len(self.thumbnails)

    @property
    def shape(self):
        shapes = [t.image.shape for t in self.thumbnails]
        assert len(set(shapes)) == 1
        exemplar_shape = shapes[0]
        return Point2D(y=exemplar_shape[0], x=exemplar_shape[1])


def _get_highest_common_vis_level(image_paths: list[Path]):
    vis_levels = []
    for path in image_paths:
        zgroup = zarr.open(store=tifffile.imread(path, aszarr=True), mode="r")
        datasets = zgroup.attrs["multiscales"][0]["datasets"]
        levels = [int(d["path"]) for d in datasets]
        vis_levels.append(max(levels))

    return min(vis_levels)


def _get_dtype(path: Path):
    with TiffReader(path) as reader:
        dtype = reader.pages[0].dtype
    # zgroup = zarr.open(store=tifffile.imread(path, aszarr=True), mode="r")
    # layer = -1
    # dtype = da.from_zarr(
    #     zgroup[int(zgroup.attrs["multiscales"][0]["datasets"][layer]["path"])]
    # ).dtype
    return dtype


def _write_avi(frames: list[npt.NDArray], dst: Path, fps: float):
    # input is list of numpy arrays in YXC!
    height, width = frames[0].shape[0:2]

    # opencv accepts arrays in BGR, not RGB, so we flip along axis==2
    frames = (np.flip(f, axis=2) for f in frames)

    video = cv2.VideoWriter(
        str(dst), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
    )
    for frame in frames:
        video.write(frame)
    cv2.destroyAllWindows()
    video.release()


@dataclass
class SlideStack:
    scene_stacks: list[SceneStack]
    orig_dtype: npt.DTypeLike

    stitched: npt.NDArray | None = None
    labeled: npt.NDArray | None = None

    total_areas_by_round: dict[int, float] | None = None
    disturbances_by_round: dict[int, float] | None = None

    @classmethod
    def from_paths(cls, image_paths: list[Path], ref_channel: int):
        vis_level = _get_highest_common_vis_level(image_paths)
        dtype = _get_dtype(image_paths[0])

        scene_stacks = [
            SceneStack.from_path(path, ref_channel=ref_channel, vis_level=vis_level)
            for path in image_paths
        ]

        return cls(scene_stacks=scene_stacks, orig_dtype=dtype)

    @property
    def n_rounds(self) -> int:
        return max([len(s.thumbnails) for s in self.scene_stacks])

    @property
    def round_names(self):
        exemplar_scene: SceneStack = next(
            s for s in self.scene_stacks if s.n_rounds == self.n_rounds
        )
        round_names = [t.cycle for t in exemplar_scene.thumbnails]
        return round_names

    def make_stitched(self) -> None:
        extents_ul = [s.offset_px for s in self.scene_stacks]
        extents_lr = [s.offset_px + s.shape for s in self.scene_stacks]

        min_corner = Point2D(
            y=min([p.y for p in extents_ul]),
            x=min([p.x for p in extents_ul]),
        )
        max_corner = Point2D(
            y=max([p.y for p in extents_lr]),
            x=max([p.x for p in extents_lr]),
        )

        stitched_shape = max_corner - min_corner
        stitched = np.zeros(
            shape=(self.n_rounds, stitched_shape.y, stitched_shape.x), dtype=float
        )

        # take each scene's stack and max it with the stitched array, with that scene's
        # offset
        for scene in self.scene_stacks:
            shape = scene.shape
            yr = slice(
                scene.offset_px.y - min_corner.y,
                scene.offset_px.y - min_corner.y + shape.y,
            )
            xr = slice(
                scene.offset_px.x - min_corner.x,
                scene.offset_px.x - min_corner.x + shape.x,
            )

            stack = []
            for t in scene.thumbnails:
                im = t.image

                in_range = tuple(np.percentile(im, (30, 99)))
                stack.append(rescale_intensity(im, in_range=in_range))

            stack = np.stack(stack)
            temp = np.zeros_like(stitched)
            temp[: len(stack), yr, xr] = stack
            np.maximum(stitched, temp, out=stitched)

        self.stitched = img_as_ubyte(stitched)

    def compute_disturbances(self):
        total_areas = {}
        for i in range(self.n_rounds):
            areas = []
            for s in self.scene_stacks:
                try:
                    areas.append(s.thumbnails[i].tissue_mask.sum())
                except IndexError:
                    pass
            total_areas[i] = sum(areas)
        # print(total_areas)
        self.total_areas_by_round = total_areas

        disturbances = {}
        for s in self.scene_stacks:
            for i in range(self.n_rounds - 1):
                mask_prev = s.thumbnails[i].tissue_mask
                mask_next = s.thumbnails[i + 1].tissue_mask

                disturbance = np.sum(np.logical_xor(mask_prev, mask_next))
                try:
                    disturbances[i].append(disturbance)
                except KeyError:
                    disturbances[i] = [disturbance]

        disturbances = {i: sum(d) for i, d in disturbances.items()}
        # print(disturbances)
        self.disturbances_by_round = disturbances

    def make_labeled(self, dst: Path) -> None:
        font = ImageFont.truetype("Arial.ttf", size=64)

        magenta = [1, 0, 1]
        cyan = [0, 1, 1]
        white = [1, 1, 1]

        # min and max vals for dtype
        min_val = np.iinfo(self.stitched.dtype).min
        max_val = np.iinfo(self.stitched.dtype).max

        frames = []
        for i in range(self.n_rounds - 1):
            round_prev = self.round_names[i]
            round_next = self.round_names[i + 1]

            # convert from grayscale to colored
            stitched_prev = self.stitched[i]
            stitched_next = self.stitched[i + 1]

            colored_prev = np.dstack([v * stitched_prev for v in magenta])
            colored_next = np.dstack([v * stitched_next for v in cyan])

            # mix prev and next
            blended = colored_prev + colored_next
            np.clip(blended, min_val, max_val, out=blended)
            blended = blended.astype(np.uint8)

            pil_im = Image.fromarray(blended, mode="RGB")
            # previous round
            ImageDraw.Draw(pil_im).text(
                (0, 0),
                round_prev,
                font=font,
                fill=tuple(v * 255 for v in magenta),
            )
            # next round
            ImageDraw.Draw(pil_im).text(
                (150, 0),
                round_next,
                font=font,
                fill=tuple(v * 255 for v in cyan),
            )
            # tissue 'coverage'?
            area_prev = self.total_areas_by_round[i]
            area_next = self.total_areas_by_round[i + 1]
            area_lost_pct = abs(area_next - area_prev) / area_prev * 100

            ImageDraw.Draw(pil_im).text(
                (0, 80),
                f"Cov. lost (rel): {area_lost_pct:,.1f}%",
                font=font,
                fill=tuple(v * 255 for v in white),
            )
            # disturbed
            dist_prev = self.disturbances_by_round[i]
            dist_rel_to_area = dist_prev / area_prev * 100
            ImageDraw.Draw(pil_im).text(
                (0, 160),
                f"Disturbed: {dist_rel_to_area:,.1f}%",
                font=font,
                fill=tuple(v * 255 for v in white),
            )

            frame = np.array(pil_im)
            frames.append(frame)

        # double last frame to mark the end
        frames.append(frames[-1])

        _write_avi(frames, dst, fps=1)


def make_tissue_loss_gif(image_paths: list[Path], dst: Path, ref_channel: int):
    slide_stack = SlideStack.from_paths(image_paths, ref_channel=ref_channel)
    slide_stack.make_stitched()
    slide_stack.compute_disturbances()
    slide_stack.make_labeled(dst)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-path", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    parser.add_argument("--ref-channel", type=int, required=True)

    args = parser.parse_args()

    make_tissue_loss_gif(
        image_paths=read_paths(Path(args.images_path)),
        dst=Path(args.dst),
        ref_channel=args.ref_channel,
    )
