# Copyright 2021 Petrov, Danil <ddbihbka@gmail.com>. All Rights Reserved.
# Author: Petrov, Danil <ddbihbka@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for Fusion 360 Gallery dataset."""

import numpy as np
import tensorflow_datasets.public_api as tfds

from datasets.cad.fusion360gallery import fusion360gallery


class Fusion360GallerySegmentationTest(tfds.testing.DatasetBuilderTestCase):
    """Test case for Fusion 360 Gallery Segmentation dataset."""
    DATASET_CLASS = fusion360gallery.Fusion360GallerySegmentation
    SPLITS = {
        "train": 1,  # Number of fake train example
        "test": 1,  # Number of fake test example
    }

    SKIP_CHECKSUMS = True


class SimpleEdgeConfigTest(tfds.testing.TestCase):
    def test_handful_example(self):
        """Example:

          |-----|-----|
          |     |     |
          |    0|1    |
          | F0  |  F1 |
          |__ __|__ __|

          Here we have 1 edges and 2 coedges.

          F0 ----- CE1--
          |  _____| |  |
          | |       |  |
          CE0----- F1  |
           |           |
           ------- E0--|
        """

        face_features = np.ones((2, 1), np.float32)
        edge_features = np.ones((1, 1), np.float32)
        coedge_features = np.ones((2, 1), np.float32)

        coedge_to_next = np.zeros((2, ), np.int32)
        coedge_to_mate = np.array([1, 0], np.int32)
        coedge_to_face = np.array([0, 1])
        coedge_to_edge = np.array([0, 0])

        g = fusion360gallery.SimpleEdgeConfig(
                name="test", description="test").create_graph(
                    face_features, edge_features, coedge_features,
                    coedge_to_next, coedge_to_mate, coedge_to_face,
                    coedge_to_edge)

        self.assertEqual(5, g["n_node"][0])
        self.assertEqual(9 * 2, g["n_edge"][0], list(zip(g["senders"], g["receivers"])))


if __name__ == "__main__":
    tfds.testing.test_main()
