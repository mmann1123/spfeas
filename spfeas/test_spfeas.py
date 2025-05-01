import os
import shutil
import unittest
import rasterio

from .errors import logger
from .spfeas import spatial_features
from .paths import get_path
from spfeas.test_rasters import (
    create_random_noise_raster,
    create_oriented_stripes_raster,
    create_random_stripes_raster,
    create_blobs_raster,
)

import mpglue as gl

import numpy as np


SPFEAS_PATH = get_path()


def test_features():
    """
    Test SpFeas features
    """

    data_dir = os.path.join(SPFEAS_PATH, "data")
    good_features_dir = os.path.join(data_dir, "_features")
    test_features_dir = os.path.join(data_dir, "features")

    good_features_mean = os.path.join(
        good_features_dir, "test_image__BD1_BK4_SC8_TRmean.vrt"
    )
    test_features_mean = os.path.join(
        test_features_dir, "test_image__BD1_BK4_SC8_TRmean.vrt"
    )

    assert os.path.isfile(good_features_mean)

    if os.path.isdir(test_features_dir):
        os.remove(test_features_dir)

    image = os.path.join(data_dir, "test_image.tif")

    spatial_features(
        image,
        test_features_dir,
        band_positions=[1],
        block=4,
        scales=[8],
        triggers=["mean"],
    )

    with gl.ropen(good_features_mean) as good_info:

        good_bands = good_info.bands

        good_band1 = good_info.read(bands2open=1, d_type="float32")

        good_band2 = good_info.read(bands2open=2, d_type="float32")

    del good_info

    with gl.ropen(test_features_mean) as test_info:

        test_bands = test_info.bands

        test_band1 = test_info.read(bands2open=1, d_type="float32")

        test_band2 = test_info.read(bands2open=2, d_type="float32")

    del test_info

    if test_bands != good_bands:
        logger.error("  The output band number did not match the test.")

    if not np.allclose(test_band1, good_band1):
        logger.error("  Band 1 did not match the test.")

    if not np.allclose(test_band2, good_band2):
        logger.error("  Band 2 did not match the test.")

    logger.info("")
    logger.info("  SpFeas tests were OK.")

    shutil.rmtree(test_features_dir)


class TestSpFeas(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Generate test rasters before running tests."""
        cls.test_dir = "test_rasters"
        os.makedirs(cls.test_dir, exist_ok=True)

        cls.random_noise_path = os.path.join(cls.test_dir, "random_noise.tif")
        cls.horizontal_stripes_path = os.path.join(
            cls.test_dir, "horizontal_stripes.tif"
        )
        cls.vertical_stripes_path = os.path.join(cls.test_dir, "vertical_stripes.tif")
        cls.random_stripes_path = os.path.join(cls.test_dir, "random_stripes.tif")
        cls.small_blobs_path = os.path.join(cls.test_dir, "small_blobs.tif")
        cls.large_blobs_path = os.path.join(cls.test_dir, "large_blobs.tif")

        create_random_noise_raster(100, 100, cls.random_noise_path)
        create_oriented_stripes_raster(
            100, 100, cls.horizontal_stripes_path, orientation="horizontal"
        )
        create_oriented_stripes_raster(
            100, 100, cls.vertical_stripes_path, orientation="vertical"
        )
        create_random_stripes_raster(100, 100, cls.random_stripes_path)
        create_blobs_raster(100, 100, cls.small_blobs_path, blob_size="small")
        create_blobs_raster(100, 100, cls.large_blobs_path, blob_size="large")

    @classmethod
    def tearDownClass(cls):
        """Clean up test rasters after tests."""
        for file in os.listdir(cls.test_dir):
            os.remove(os.path.join(cls.test_dir, file))
        os.rmdir(cls.test_dir)

    def test_random_noise_raster(self):
        """Test if random noise raster values are within valid range."""
        with rasterio.open(self.random_noise_path) as src:
            data = src.read(1)
            self.assertTrue((data >= 0).all() and (data <= 1).all())

    def test_oriented_stripes_raster(self):
        """Test if oriented stripes raster contains only 0 and 1 values."""
        with rasterio.open(self.horizontal_stripes_path) as src:
            data = src.read(1)
            self.assertTrue(set(data.flatten()).issubset({0, 1}))

        with rasterio.open(self.vertical_stripes_path) as src:
            data = src.read(1)
            self.assertTrue(set(data.flatten()).issubset({0, 1}))

    def test_random_stripes_raster(self):
        """Test if random stripes raster contains only 0 and 1 values."""
        with rasterio.open(self.random_stripes_path) as src:
            data = src.read(1)
            self.assertTrue(set(data.flatten()).issubset({0, 1}))

    def test_blobs_raster(self):
        """Test if blobs raster contains only 0 and 1 values."""
        with rasterio.open(self.small_blobs_path) as src:
            data = src.read(1)
            self.assertTrue(set(data.flatten()).issubset({0, 1}))

        with rasterio.open(self.large_blobs_path) as src:
            data = src.read(1)
            self.assertTrue(set(data.flatten()).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
