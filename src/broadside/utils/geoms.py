from dataclasses import dataclass


@dataclass(frozen=True)
class Range:
    min: float
    max: float

    def __post_init__(self):
        if self.min > self.max:
            raise ValueError(f"{self.min} greater than {self.max}")

    def __contains__(self, value: float):
        return self.min <= value <= self.max


@dataclass
class Point2D:
    """
    I like being strict about the dunder methods.
    """

    x: float
    y: float

    def __add__(self, other: "Point2D"):
        return Point2D(x=self.x + other.x, y=self.y + other.y)

    def __sub__(self, other: "Point2D"):
        return Point2D(x=self.x - other.x, y=self.y - other.y)

    def __truediv__(self, other):
        if isinstance(other, Point2D):
            return Point2D(x=self.x / other.x, y=self.y / other.y)
        else:
            return Point2D(x=self.x / other, y=self.y / other)

    def as_dict(self):
        return dict(x=self.x, y=self.y)


@dataclass
class Point3D:
    x: float
    y: float
    z: float

    def __add__(self, other: "Point3D"):
        return Point3D(
            x=self.x + other.x,
            y=self.y + other.y,
            z=self.z + other.z,
        )

    def __sub__(self, other: "Point3D"):
        return Point3D(
            x=self.x - other.x,
            y=self.y - other.y,
            z=self.z - other.z,
        )

    def as_dict(self):
        return dict(x=self.x, y=self.y, z=self.z)


@dataclass(frozen=True)
class Size:
    w: float
    h: float


@dataclass(frozen=True)
class Rect:
    origin: Point2D
    size: Size


def get_center_of_points(x: list[float], y: list[float]) -> Point2D:
    cx = min(x) / 2 + max(x) / 2
    cy = min(y) / 2 + max(y) / 2
    return Point2D(x=cx, y=cy)
