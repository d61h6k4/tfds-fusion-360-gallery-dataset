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
"""Extract features from STEP file.

Reference implementation:
https://github.com/AutodeskAILab/BRepNet/blob/master/pipeline/extract_brepnet_data_from_step.py

The reason to copy (retype) the code here is to learn.
Simple retyping requires much more attention, so knowledge
than reading.
"""

from __future__ import annotations
from typing import Optional, Sequence

import pathlib
import numpy as np

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GeomAbs import (GeomAbs_BSplineSurface, GeomAbs_BezierSurface,
                              GeomAbs_Circle, GeomAbs_Cone, GeomAbs_Cylinder,
                              GeomAbs_Ellipse, GeomAbs_Line, GeomAbs_Plane,
                              GeomAbs_Sphere, GeomAbs_SurfaceType,
                              GeomAbs_Torus)
from OCC.Core.gp import gp_Pnt, gp_Trsf, gp_Vec
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopAbs import TopAbs_REVERSED
from OCC.Core.TopoDS import TopoDS_Edge, TopoDS_Face, TopoDS_Shape
from OCC.Extend.DataExchange import read_step_file
from OCC.Extend.TopologyUtils import TopologyExplorer, WireExplorer

from occwl.edge_data_extractor import EdgeDataExtractor, EdgeConvexity
from occwl.edge import Edge
from occwl.face import Face

from datasets.cad.fusion360gallery.entity_mapper import EntityMapper


def load_body(step_filepath: pathlib.Path) -> TopoDS_Shape:
    """Load the body from the given STEP file.
    """
    return read_step_file(str(step_filepath.absolute()), as_compound=False)


def find_box(body: TopoDS_Shape) -> Bnd_Box:
    """Find the bounding box that contains the given body."""
    box = Bnd_Box()
    use_triangulation = True
    use_shapetolerance = False
    brepbndlib.AddOptimal(body, box, use_triangulation, use_shapetolerance)
    return box


def scale_solid_to_unit_box(body: TopoDS_Shape) -> TopoDS_Shape:
    """ Centered the body on the origin and scaled so it just fits
    into a box [-1, 1]^3
    """
    bbox = find_box(body)

    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    longest_length = max(xmax - xmin, ymax - ymin, zmax - zmin)

    orig = gp_Pnt(0.0, 0.0, 0.0)
    center = gp_Pnt(
        (xmin + xmax) / 2.0,
        (ymin + ymax) / 2.0,
        (zmin + zmax) / 2.0,
    )
    vec_center_to_orig = gp_Vec(center, orig)
    move_to_center = gp_Trsf()
    move_to_center.SetTranslation(vec_center_to_orig)

    scale_trsf = gp_Trsf()
    scale_trsf.SetScale(orig, 2.0 / longest_length)
    trsf_to_apply = scale_trsf.Multiplied(move_to_center)

    apply_transform = BRepBuilderAPI_Transform(trsf_to_apply)
    apply_transform.Perform(body)
    transformed_body = apply_transform.ModifiedShape(body)

    return transformed_body


def check_manifold(body: TopoDS_Shape) -> bool:
    """Check that body is manifold."""
    faces = set()
    # Assumption: to create TopologyExplorer costs nothing
    for shell in TopologyExplorer(body, ignore_orientation=True).shells():
        for face in TopologyExplorer(shell, ignore_orientation=True).faces():
            if face in faces:
                return False
            faces.add(face)
    return True


def find_edges_from_wires(body: TopoDS_Shape) -> set[TopoDS_Edge]:
    """Return set of edges from Wires."""
    edge_set = set()
    for wire in TopologyExplorer(body, ignore_orientation=False).wires():
        for edge in WireExplorer(wire).ordered_edges():
            edge_set.add(edge)
    return edge_set


def find_edges_from_top_exp(body: TopoDS_Shape) -> set[TopoDS_Edge]:
    """Return set of edges from Shape."""
    return set(TopologyExplorer(body, ignore_orientation=False).edges())


def check_closed(body: TopoDS_Shape) -> bool:
    """Check the given body is closed.
    In Open Cascade, unlinked (open) edges can be identified
    as they appear in the edges iterator when ignore_orientation=False
    but are not present in any wire
    """
    edges_from_wires = find_edges_from_wires(body)
    edges_from_top_exp = find_edges_from_top_exp(body)

    missing_edges = edges_from_top_exp.symmetric_difference(edges_from_wires)
    return not missing_edges


def check_unique_coedges(body: TopoDS_Shape) -> bool:
    """Check the given body doesn't have coedges that used in a multiple
    loops. Additional check that the body is manifold?
    """
    coedge_set = set()
    for wire in TopologyExplorer(body, ignore_orientation=True).wires():
        for coedge in WireExplorer(wire).ordered_edges():
            oriented_edge = (coedge, coedge.Orientation())
            if oriented_edge in coedge_set:
                return False
            coedge_set.add(oriented_edge)
    return True


def is_type(surf_type: GeomAbs_SurfaceType,
            desire_type: GeomAbs_SurfaceType) -> bool:
    """Check the surf_type equal to desire_type."""
    return surf_type == desire_type


def feature_is_type(surf_type: GeomAbs_SurfaceType,
                    desire_type: GeomAbs_SurfaceType) -> float:
    """Convert is_type to feature."""
    return float(is_type(surf_type, desire_type))


def plane_feature(surf_type: GeomAbs_SurfaceType) -> float:
    """Plane feature."""
    return feature_is_type(surf_type, GeomAbs_Plane)


def cylinder_feature(surf_type: GeomAbs_SurfaceType) -> float:
    """Cylinder feature."""
    return feature_is_type(surf_type, GeomAbs_Cylinder)


def cone_feature(surf_type: GeomAbs_SurfaceType) -> float:
    """Cone feature."""
    return feature_is_type(surf_type, GeomAbs_Cone)


def sphere_feature(surf_type: GeomAbs_SurfaceType) -> float:
    """Sphere feature."""
    return feature_is_type(surf_type, GeomAbs_Sphere)


def torus_feature(surf_type: GeomAbs_SurfaceType) -> float:
    """Torus feature."""
    return feature_is_type(surf_type, GeomAbs_Torus)


def area_feature(face: TopoDS_Face) -> float:
    """Calculate the area of the face."""
    geometry_properties = GProp_GProps()
    brepgprop.SurfaceProperties(face, geometry_properties)
    return geometry_properties.Mass()


def rational_nurbs_feature(surf: BRepAdaptor_Surface,
                           surf_type: GeomAbs_SurfaceType) -> float:
    """Is surface the rational NURB."""
    bspline = None
    if surf_type == GeomAbs_BSplineSurface:
        bspline = surf.BSpline()
    elif surf_type == GeomAbs_BezierSurface:
        bspline = surf.Bezier()

    if bspline is not None:
        if bspline.IsURational() or bspline.IsVRational():
            return 1.
    return 0.


def features_from_face(face: TopoDS_Face) -> np.ndarray:
    """Extract features of the given face."""
    surf = BRepAdaptor_Surface(face)
    surf_type = surf.GetType()
    # we do this trick to be not dependant of
    # order of elements in list
    features = {
        0: plane_feature(surf_type),
        1: cylinder_feature(surf_type),
        2: cone_feature(surf_type),
        3: sphere_feature(surf_type),
        4: torus_feature(surf_type),
        5: area_feature(face),
        6: rational_nurbs_feature(surf, surf_type)
    }
    face_features = np.zeros((len(features), ), dtype=np.float32)

    for k, v in features.items():
        face_features[k] = v

    return face_features


def face_features_from_body(body: TopoDS_Shape,
                            entity_mapper: EntityMapper) -> np.ndarray:
    """Extract the face features from each face of the body."""
    face_features = []

    for ix, face in enumerate(
            TopologyExplorer(body, ignore_orientation=True).faces()):
        assert ix == entity_mapper.face_index(face)
        face_features.append(features_from_face(face))

    return np.stack(face_features)


def find_edge_convexivity(
        edge: TopoDS_Edge,
        faces: Sequence[TopoDS_Face]) -> Optional[EdgeConvexity]:
    """Edge convexivity."""
    edge_data = EdgeDataExtractor(Edge(edge), [Face(f) for f in faces],
                                  use_arclength_params=False)
    if not edge_data.good:
        # This is the case where the edge is a pole of a sphere
        return None
    # defines the smoothnes relative to this angle
    angle_tol_rads = 0.0872664626  # 5 degrees
    convexity = edge_data.edge_convexity(angle_tol_rads)
    return convexity


def concavity_feature(convexity: Optional[EdgeConvexity]) -> float:
    """Is edge concave?"""
    return convexity is not None and convexity == EdgeConvexity.CONCAVE


def convexity_feature(convexity: Optional[EdgeConvexity]) -> float:
    """Is edge convex?"""
    return convexity is not None and convexity == EdgeConvexity.CONVEX


def smoothity_feature(convexity: Optional[EdgeConvexity]) -> float:
    """Is edge smooth?"""
    return convexity is not None and convexity == EdgeConvexity.SMOOTH


def edge_length_feature(edge: TopoDS_Edge) -> float:
    """Edge lenght."""
    geometry_properties = GProp_GProps()
    brepgprop.LinearProperties(edge, geometry_properties)
    return geometry_properties.Mass()


def circular_edge_feature(edge: TopoDS_Edge) -> float:
    """Is edge circular?"""
    brep_adaptor_curve = BRepAdaptor_Curve(edge)
    curv_type = brep_adaptor_curve.GetType()
    if curv_type == GeomAbs_Circle:
        return 1.0
    return 0.0


def closed_edge_feature(edge: TopoDS_Edge) -> float:
    """Is edge closed?"""
    if BRep_Tool().IsClosed(edge):
        return 1.0
    return 0.0


def elliptical_edge_feature(edge: TopoDS_Edge) -> float:
    """Is edge elliptical?"""
    brep_adaptor_curve = BRepAdaptor_Curve(edge)
    curv_type = brep_adaptor_curve.GetType()
    if curv_type == GeomAbs_Ellipse:
        return 1.0
    return 0.0


def straight_edge_feature(edge: TopoDS_Edge) -> float:
    """Is edge line?"""
    brep_adaptor_curve = BRepAdaptor_Curve(edge)
    curv_type = brep_adaptor_curve.GetType()
    if curv_type == GeomAbs_Line:
        return 1.0
    return 0.0


def hyperbolic_edge_feature(edge: TopoDS_Edge) -> float:
    """Is edge hyperbola?"""
    if Edge(edge).curve_type() == "hyperbola":
        return 1.0
    return 0.0


def parabolic_edge_feature(edge: TopoDS_Edge) -> float:
    """Is edge parabola?"""
    if Edge(edge).curve_type() == "parabola":
        return 1.0
    return 0.0


def bezier_edge_feature(edge: TopoDS_Edge) -> float:
    """Is edge bezier?"""
    if Edge(edge).curve_type() == "bezier":
        return 1.0
    return 0.0


def non_rational_bspline_edge_feature(edge: TopoDS_Edge) -> float:
    """Is edge NRBSpline?"""
    occwl_edge = Edge(edge)
    if occwl_edge.curve_type() == "bspline" and not occwl_edge.rational():
        return 1.0
    return 0.0


def rational_bspline_edge_feature(edge: TopoDS_Edge) -> float:
    """Is edge RBSpline?"""
    occwl_edge = Edge(edge)
    if occwl_edge.curve_type() == "bspline" and occwl_edge.rational():
        return 1.0
    return 0.0


def offset_edge_feature(edge: TopoDS_Edge) -> float:
    """Is edge offset?"""
    if Edge(edge).curve_type() == "offset":
        return 1.0
    return 0.0


def features_from_edge(edge: TopoDS_Edge,
                       faces: Sequence[TopoDS_Face]) -> np.ndarray:
    """Extract features of given edge."""
    convexity = find_edge_convexivity(edge, faces)
    features = {
        0: concavity_feature(convexity),
        1: convexity_feature(convexity),
        2: smoothity_feature(convexity),
        3: edge_length_feature(edge),
        4: circular_edge_feature(edge),
        5: closed_edge_feature(edge),
        6: elliptical_edge_feature(edge),
        7: straight_edge_feature(edge),
        8: hyperbolic_edge_feature(edge),
        9: parabolic_edge_feature(edge),
        10: bezier_edge_feature(edge),
        11: non_rational_bspline_edge_feature(edge),
        12: rational_bspline_edge_feature(edge),
        13: offset_edge_feature(edge),
    }

    edge_features = np.zeros((len(features), ), dtype=np.float32)

    for k, v in features.items():
        edge_features[k] = v

    return edge_features


def edge_features_from_body(body: TopoDS_Shape,
                            entity_mapper: EntityMapper) -> np.ndarray:
    """Extract the edge features from each edge of the body."""
    edge_features = []

    top_exp = TopologyExplorer(body, ignore_orientation=True)
    for ix, edge in enumerate(top_exp.edges()):
        assert ix == entity_mapper.edge_index(edge)
        edge_features.append(
            features_from_edge(edge, top_exp.faces_from_edge(edge)))
    return np.stack(edge_features)


def reversed_edge_feature(edge: TopoDS_Edge) -> float:
    """Is the edge reversed?"""
    if edge.Orientation() == TopAbs_REVERSED:
        return 1.0
    return 0.0


def features_from_coedge(coedge: TopoDS_Edge) -> np.ndarray:
    """Extract features from the given coedge."""
    features = {0: reversed_edge_feature(coedge)}

    coedge_features = np.zeros((len(features), ), dtype=np.float32)

    for k, v in features.items():
        coedge_features[k] = v

    return coedge_features


def coedge_features_from_body(body: TopoDS_Shape,
                              entity_mapper: EntityMapper) -> np.ndarray:
    """Extract the coedge features from each face of the body."""
    coedge_features = []

    ix = 0
    for wire in TopologyExplorer(body, ignore_orientation=False).wires():
        for coedge in WireExplorer(wire).ordered_edges():
            assert ix == entity_mapper.halfedge_index(coedge)
            coedge_features.append(features_from_coedge(coedge))
            ix += 1

    return np.stack(coedge_features)


def build_incidence_arrays(
    body: TopoDS_Shape, entity_mapper: EntityMapper
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build incidence arrays."""
    oriented_top_exp = TopologyExplorer(body, ignore_orientation=False)
    num_coedges = len(entity_mapper.halfedge_map)

    next_ = np.zeros(num_coedges, dtype=np.uint32)
    mate = np.zeros(num_coedges, dtype=np.uint32)

    # Create the next, pervious and mate permutations
    for loop in oriented_top_exp.wires():
        wire_exp = WireExplorer(loop)
        first_coedge_index = None
        previous_coedge_index = None
        for coedge in wire_exp.ordered_edges():
            coedge_index = entity_mapper.halfedge_index(coedge)

            # Set up the mating coedge
            mating_coedge = coedge.Reversed()
            if entity_mapper.halfedge_exists(mating_coedge):
                mating_coedge_index = entity_mapper.halfedge_index(
                    mating_coedge)
            else:
                # If a coedge has no mate then we mate it to
                # itself.  This typically happens at the poles
                # of sphere
                mating_coedge_index = coedge_index
            mate[coedge_index] = mating_coedge_index

            # Set up the next coedge
            if first_coedge_index is None:
                first_coedge_index = coedge_index
            else:
                next_[previous_coedge_index] = coedge_index
            previous_coedge_index = coedge_index

        # Close the loop
        next_[previous_coedge_index] = first_coedge_index

    # Create the arrays from coedge to face
    coedge_to_edge = np.zeros(num_coedges, dtype=np.uint32)
    for loop in oriented_top_exp.wires():
        for coedge in WireExplorer(loop).ordered_edges():
            coedge_index = entity_mapper.halfedge_index(coedge)
            edge_index = entity_mapper.edge_index(coedge)
            mating_coedge = coedge.Reversed()
            if entity_mapper.halfedge_exists(mating_coedge):
                mating_coedge_index = entity_mapper.halfedge_index(
                    mating_coedge)
            else:
                # If a coedge has no mate then we mate it to
                # itself.  This typically happens at the poles
                # of sphere
                mating_coedge_index = coedge_index
            coedge_to_edge[coedge_index] = edge_index
            coedge_to_edge[mating_coedge_index] = edge_index

    # Loop over the faces and make the back
    # pointers back to the edges
    coedge_to_face = np.zeros(num_coedges, dtype=np.uint32)
    unoriented_top_exp = TopologyExplorer(body, ignore_orientation=True)
    for face in unoriented_top_exp.faces():
        face_index = entity_mapper.face_index(face)
        for loop in unoriented_top_exp.wires_from_face(face):
            wire_exp = WireExplorer(loop)
            for coedge in wire_exp.ordered_edges():
                coedge_index = entity_mapper.halfedge_index(coedge)
                coedge_to_face[coedge_index] = face_index

    return next_, mate, coedge_to_face, coedge_to_edge
