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
"""Fusion 360 Gallery dataset."""

from __future__ import annotations
from typing import Sequence

import json
import pathlib
import warnings

import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

from tfds_fusion_360_gallery_dataset.cad.fusion360gallery import brepnet_features

_DESCRIPTION = """\
The Fusion 360 Gallery Dataset contains rich 2D and 3D geometry data derived
from parametric CAD models. The dataset is produced from designs submitted by
users of the CAD package Autodesk Fusion 360 to the Autodesk Online Gallery.
The dataset provides valuable data for learning how people design, including
sequential CAD design data, designs segmented by modeling operation, and design
hierarchy and connectivity data.
"""
_CITATION = r"""\
@article{willis2020fusion,
    title={Fusion 360 Gallery: A Dataset and Environment for Programmatic CAD
           Construction from Human Design Sequences},
    author={Karl D. D. Willis and Yewen Pu and Jieliang Luo and Hang Chu and
            Tao Du and Joseph G. Lambourne and Armando Solar-Lezama and Wojciech Matusik},
    journal={ACM Transactions on Graphics (TOG)},
    volume={40},
    number={4},
    year={2021},
    publisher={ACM New York, NY, USA}
}
@inproceedings{lambourne2021brepnet,
    author    = {Lambourne, Joseph G. and Willis, Karl D.D. and Jayaraman,
                 Pradeep Kumar and Sanghi, Aditya and Meltzer, Peter and Shayani, Hooman},
    title     = {BRepNet: A Topological Message Passing System for Solid Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {12773-12782}
}
"""
_URL = "https://github.com/AutodeskAILab/Fusion360GalleryDataset"


class SimpleEdgeConfig(tfds.core.BuilderConfig):
    """Simple Edge configuration."""
    def create_graph(self, face_features: np.ndarray,
                     edge_features: np.ndarray, coedge_features: np.ndarray,
                     coedge_to_next: np.ndarray, coedge_to_mate: np.ndarray,
                     coedge_to_face: np.ndarray, coedge_to_edge: np.ndarray):
        """Create graph according to the `simple edge` configuration."""
        del coedge_to_next

        faces_num = face_features.shape[0]
        edges_num = edge_features.shape[0]
        coedges_num = coedge_features.shape[0]

        face_to_node = np.arange(faces_num)
        edge_to_node = np.arange(edges_num) + faces_num
        coedge_to_node = np.arange(coedges_num) + (faces_num + edges_num)

        edges = []
        # Faces
        # F
        for coedge_ix, face_ix in enumerate(coedge_to_face):
            edges.append((coedge_to_node[coedge_ix], face_to_node[face_ix]))
        # MF
        for coedge_from_ix, coedge_to_ix in enumerate(coedge_to_mate):
            edges.append((coedge_to_node[coedge_from_ix],
                          face_to_node[coedge_to_face[coedge_to_ix]]))
        # Edges
        # E
        for coedge_ix, edge_ix in enumerate(coedge_to_edge):
            edges.append((coedge_to_node[coedge_ix], edge_to_node[edge_ix]))
        # CoEdges
        # I
        for coedge_ix in range(coedges_num):
            edges.append(
                (coedge_to_node[coedge_ix], coedge_to_node[coedge_ix]))
        # M
        for coedge_from_ix, coedge_to_ix in enumerate(coedge_to_mate):
            edges.append(
                (coedge_to_node[coedge_from_ix], coedge_to_node[coedge_to_ix]))

        n_node = faces_num + edges_num + coedges_num

        senders = []
        receivers = []
        for edge_ix, (f, t) in enumerate(edges):
            senders.append(f)
            receivers.append(t)
            # don't add self-loops more than once
            if f != t:
                senders.append(t)
                receivers.append(f)

        assert len(senders) == len(receivers)
        n_edge = len(senders)

        nodes = np.concatenate(
            (np.pad(face_features,
                    ((0, 0),
                     (0, edge_features.shape[1] + coedge_features.shape[1]))),
             np.pad(edge_features,
                    ((0, 0),
                     (face_features.shape[1], coedge_features.shape[1]))),
             np.pad(coedge_features,
                    ((0, 0),
                     (face_features.shape[1] + edge_features.shape[1], 0)))))

        return {
            "n_node": np.array([n_node], dtype=np.int32),
            "n_edge": np.array([n_edge], dtype=np.int32),
            "nodes": nodes,
            "senders": np.array(senders, dtype=np.int32),
            "receivers": np.array(receivers, dtype=np.int32),
        }


class Fusion360GallerySegmentation(tfds.core.GeneratorBasedBuilder):
    """Fusion 360 Gallery Dataset (segmentation)"""
    VERSION = tfds.core.Version("2.0.0")
    RELEASE_NOTE = {"2.0.0": "Initial release."}
    # pytype: disable=wrong-keyword-args
    BUILDER_CONFIGS = [
        # `name` (and optionally `description`) are required for each config
        SimpleEdgeConfig(name="simple_edge", description="Simple edge."),
    ]

    # pytype: enable=wrong-keyword-args

    def _info(self):
        """Define the dataset info."""
        return tfds.core.DatasetInfo(builder=self,
                                     description=_DESCRIPTION,
                                     features=tfds.features.FeaturesDict({
                                         "face_labels":
                                         tfds.features.Tensor(shape=(None, ),
                                                              dtype=tf.uint32),
                                         "n_node":
                                         tfds.features.Tensor(shape=(1, ),
                                                              dtype=tf.int32),
                                         "n_edge":
                                         tfds.features.Tensor(shape=(1, ),
                                                              dtype=tf.int32),
                                         "nodes":
                                         tfds.features.Tensor(
                                             shape=(None, 7 + 14 + 1),
                                             dtype=tf.float32),
                                         "senders":
                                         tfds.features.Tensor(shape=(None, ),
                                                              dtype=tf.int32),
                                         "receivers":
                                         tfds.features.Tensor(shape=(None, ),
                                                              dtype=tf.int32),
                                     }),
                                     homepage=_URL,
                                     citation=_CITATION)

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        output_files = dl_manager.download_and_extract(
            "https://fusion-360-gallery-dataset.s3-us-west-2.amazonaws.com/segmentation/s2.0.0/s2.0.0.zip"
        )
        output_files = pathlib.Path(output_files) / "s2.0.0"

        split_info = json.loads((output_files / "train_test.json").read_text())
        step_dir = output_files / "breps" / "step"
        seg_dir = output_files / "breps" / "seg"

        return {
            "train":
            self._generate_examples(step_dir, seg_dir, split_info["train"]),
            "test":
            self._generate_examples(step_dir, seg_dir, split_info["test"])
        }

    def _generate_examples(self, step_dir: pathlib.Path, seg_dir: pathlib.Path,
                           objids: Sequence[str]):
        """Generate examples."""
        for objid in objids:
            seg_file = seg_dir / f"{objid}.seg"
            face_labels = [
                int(x) for x in seg_file.read_text().strip().split("\n")
            ]

            step_file = step_dir / f"{objid}.stp"
            assert step_file.exists()

            loaded_body = brepnet_features.load_body(step_file)
            body = brepnet_features.scale_solid_to_unit_box(loaded_body)

            if not brepnet_features.check_manifold(body):
                warnings.warn("Non-manifold bodies are not supported.")
                continue

            if not brepnet_features.check_closed(body):
                warnings.warn("Bodies which are not closed are not supported")
                continue

            if not brepnet_features.check_unique_coedges(body):
                warnings.warn(
                    "Bodies where the same coedge is uses in multiple loops are not supported"
                )
                continue

            entity_mapper = brepnet_features.EntityMapper(body)

            face_features = brepnet_features.face_features_from_body(
                body, entity_mapper)
            edge_features = brepnet_features.edge_features_from_body(
                body, entity_mapper)
            coedge_features = brepnet_features.coedge_features_from_body(
                body, entity_mapper)

            coedge_to_next, coedge_to_mate, coedge_to_face, coedge_to_edge = brepnet_features.build_incidence_arrays(
                body, entity_mapper)

            graph = self.builder_config.create_graph(
                face_features, edge_features, coedge_features, coedge_to_next,
                coedge_to_mate, coedge_to_face, coedge_to_edge)

            yield objid, {
                "face_labels": face_labels,
                "n_node": graph["n_node"],
                "n_edge": graph["n_edge"],
                "nodes": graph["nodes"],
                "senders": graph["senders"],
                "receivers": graph["receivers"],
            }
