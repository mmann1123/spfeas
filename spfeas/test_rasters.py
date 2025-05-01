import numpy as np
import rasterio
from rasterio.transform import from_origin


def create_random_noise_raster(width, height, output_path):
    """Create a raster with random noise."""
    data = np.random.random((height, width)).astype(np.float32)
    transform = from_origin(0, 0, 1, 1)  # Example transform
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="+proj=latlong",
        transform=transform,
    ) as dst:
        dst.write(data, 1)


def create_oriented_stripes_raster(
    width, height, output_path, orientation="horizontal"
):
    """Create a raster with oriented stripes."""
    data = np.zeros((height, width), dtype=np.float32)
    if orientation == "horizontal":
        data[::2, :] = 1  # Horizontal stripes
    elif orientation == "vertical":
        data[:, ::2] = 1  # Vertical stripes
    transform = from_origin(0, 0, 1, 1)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="+proj=latlong",
        transform=transform,
    ) as dst:
        dst.write(data, 1)


def create_random_stripes_raster(width, height, output_path):
    """Create a raster with random stripes."""
    data = np.random.choice([0, 1], size=(height, width), p=[0.5, 0.5]).astype(
        np.float32
    )
    transform = from_origin(0, 0, 1, 1)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="+proj=latlong",
        transform=transform,
    ) as dst:
        dst.write(data, 1)


def create_blobs_raster(width, height, output_path, blob_size="small"):
    """Create a raster with small or large blobs."""
    data = np.zeros((height, width), dtype=np.float32)
    if blob_size == "small":
        for _ in range(50):
            x, y = np.random.randint(0, width), np.random.randint(0, height)
            data[y, x] = 1
    elif blob_size == "large":
        for _ in range(10):
            x, y = np.random.randint(0, width), np.random.randint(0, height)
            data[
                max(0, y - 5) : min(height, y + 5), max(0, x - 5) : min(width, x + 5)
            ] = 1
    transform = from_origin(0, 0, 1, 1)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="+proj=latlong",
        transform=transform,
    ) as dst:
        dst.write(data, 1)
