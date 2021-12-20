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

import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

import brep2graph.brepnet_features
import brep2graph.utils

from OCC.Core.TopoDS import TopoDS_Shape

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


class Fusion360GallerySegmentation(tfds.core.GeneratorBasedBuilder):
    """Fusion 360 Gallery Dataset (segmentation)"""
    VERSION = tfds.core.Version("2.0.0")
    RELEASE_NOTE = {"2.0.0": "Initial release."}

    def _info(self):
        """Define the dataset info."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                "face_labels":
                tfds.features.Tensor(shape=(None, ), dtype=tf.uint32),
                "face_features":
                tfds.features.Tensor(shape=(None, 7), dtype=tf.float32),
                "edge_features":
                tfds.features.Tensor(shape=(None, 14), dtype=tf.float32),
                "coedge_features":
                tfds.features.Tensor(shape=(None, 1), dtype=tf.float32),
                "coedge_to_next":
                tfds.features.Tensor(shape=(None, ), dtype=tf.uint32),
                "coedge_to_mate":
                tfds.features.Tensor(shape=(None, ), dtype=tf.uint32),
                "coedge_to_face":
                tfds.features.Tensor(shape=(None, ), dtype=tf.uint32),
                "coedge_to_edge":
                tfds.features.Tensor(shape=(None, ), dtype=tf.uint32),
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

            loaded_body = brep2graph.utils.load_body(step_file)
            body = brep2graph.utils.scale_solid_to_unit_box(loaded_body)

            if not brep2graph.utils.check_manifold(body):
                raise RuntimeError("Non-manifold bodies are not supported.")

            if not brep2graph.utils.check_closed(body):
                raise RuntimeError(
                    "Bodies which are not closed are not supported")

            if not brep2graph.utils.check_unique_coedges(body):
                raise RuntimeError(
                    "Bodies where the same coedge is uses in multiple loops are not supported"
                )

            entity_mapper = brep2graph.brepnet_features.EntityMapper(body)

            face_features = brep2graph.brepnet_features.face_features_from_body(
                body, entity_mapper)
            edge_features = brep2graph.brepnet_features.edge_features_from_body(
                body, entity_mapper)
            coedge_features = brep2graph.brepnet_features.coedge_features_from_body(
                body, entity_mapper)

            coedge_to_next, coedge_to_mate, coedge_to_face, coedge_to_edge = brep2graph.brepnet_features.build_incidence_arrays(
                body, entity_mapper)

            yield objid, {
                "face_labels": face_labels,
                "face_features": face_features,
                "edge_features": edge_features,
                "coedge_features": coedge_features,
                "coedge_to_next": coedge_to_next,
                "coedge_to_mate": coedge_to_mate,
                "coedge_to_face": coedge_to_face,
                "coedge_to_edge": coedge_to_edge,
            }
