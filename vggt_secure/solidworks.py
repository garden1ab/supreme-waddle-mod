"""
SolidWorks export: convert VGGT point clouds / meshes to CAD-importable formats.
Supported: STL, OBJ, PLY, STEP, IGES
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {"stl", "obj", "ply", "step", "iges"}


def points_to_mesh(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    method: str = "poisson",
    quality: str = "medium",
) -> "trimesh.Trimesh":
    """
    Reconstruct a watertight mesh from a point cloud.

    Args:
        points: Nx3 float array
        colors: Nx3 float array [0,1] (optional)
        method: 'poisson' (watertight, best for STEP) or 'ball_pivot' (faster)
        quality: 'low', 'medium', 'high'
    """
    import open3d as o3d
    import trimesh

    depth_map_quality = {"low": 7, "medium": 9, "high": 11}
    depth = depth_map_quality.get(quality, 9)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)

    if method == "poisson":
        mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, linear_fit=True
        )
        # Trim low-density boundary artifacts
        densities = np.asarray(densities)
        threshold = np.percentile(densities, 10)
        mesh_o3d.remove_vertices_by_mask(densities < threshold)
    else:
        distances = pcd.compute_nearest_neighbor_distance()
        avg = np.mean(distances)
        radii = [avg, avg * 2, avg * 4]
        mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )

    verts = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    vcol = None
    if mesh_o3d.has_vertex_colors():
        vcol = (np.asarray(mesh_o3d.vertex_colors) * 255).astype(np.uint8)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vcol)
    logger.info("Mesh: %d verts, %d faces", len(verts), len(faces))
    return mesh


def export_stl(mesh, path: str, binary: bool = True) -> str:
    path = _ext(path, ".stl")
    mesh.export(path, file_type="stl" if binary else "stl_ascii")
    logger.info("Exported STL: %s", path)
    return path


def export_obj(mesh, path: str) -> str:
    path = _ext(path, ".obj")
    mesh.export(path, file_type="obj")
    logger.info("Exported OBJ: %s", path)
    return path


def export_ply(points: np.ndarray, colors: Optional[np.ndarray], path: str) -> str:
    import trimesh
    path = _ext(path, ".ply")
    rgba = None
    if colors is not None:
        c = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
        rgba = np.hstack([c, np.full((len(c), 1), 255, dtype=np.uint8)])
    cloud = trimesh.PointCloud(vertices=points, colors=rgba)
    cloud.export(path)
    logger.info("Exported PLY: %s (%d pts)", path, len(points))
    return path


def export_step(mesh, path: str) -> str:
    """Export STEP. Falls back to STL if OCP not available."""
    path = _ext(path, ".step")
    try:
        _export_step_ocp(mesh, path)
    except ImportError:
        logger.warning("OCP not installed — exporting STL instead. pip install cadquery OCP")
        path = export_stl(mesh, path.replace(".step", ".stl"))
    return path


def export_iges(mesh, path: str) -> str:
    """Export IGES. Falls back to STL if OCP not available."""
    path = _ext(path, ".igs")
    try:
        _export_iges_ocp(mesh, path)
    except ImportError:
        logger.warning("OCP not installed — exporting STL instead. pip install cadquery OCP")
        path = export_stl(mesh, path.replace(".igs", ".stl"))
    return path


def export_format(
    fmt: str,
    points: np.ndarray,
    colors: Optional[np.ndarray],
    output_dir: str,
    mesh_method: str = "poisson",
    mesh_quality: str = "medium",
) -> str:
    """High-level export dispatcher."""
    fmt = fmt.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {fmt}. Use: {SUPPORTED_FORMATS}")

    base = os.path.join(output_dir, "vggt_export")

    if fmt == "ply":
        return export_ply(points, colors, base)

    mesh = points_to_mesh(points, colors, method=mesh_method, quality=mesh_quality)

    dispatch = {
        "stl": export_stl,
        "obj": export_obj,
        "step": export_step,
        "iges": export_iges,
    }
    return dispatch[fmt](mesh, base)


def export_all(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    output_dir: str,
    mesh_method: str = "poisson",
    mesh_quality: str = "medium",
) -> List[str]:
    """Export all supported formats."""
    results = []
    mesh = points_to_mesh(points, colors, method=mesh_method, quality=mesh_quality)
    base = os.path.join(output_dir, "vggt_export")

    results.append(export_stl(mesh, base))
    results.append(export_obj(mesh, base))
    results.append(export_ply(points, colors, base))
    results.append(export_step(mesh, base))
    results.append(export_iges(mesh, base))
    return results


# ── Internal helpers ────────────────────────────────────────────────────

def _ext(path: str, ext: str) -> str:
    p = Path(path)
    return str(p.with_suffix(ext)) if p.suffix.lower() != ext else path


def _export_step_ocp(mesh, path: str):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCP.gp import gp_Pnt

    sewing = BRepBuilderAPI_Sewing()
    verts = mesh.vertices
    for face in mesh.faces:
        pts = [gp_Pnt(float(verts[i][0]), float(verts[i][1]), float(verts[i][2])) for i in face]
        poly = BRepBuilderAPI_MakePolygon()
        for p in pts:
            poly.Add(p)
        poly.Close()
        if poly.IsDone():
            f = BRepBuilderAPI_MakeFace(poly.Wire())
            if f.IsDone():
                sewing.Add(f.Face())
    sewing.Perform()
    writer = STEPControl_Writer()
    writer.Transfer(sewing.SewedShape(), STEPControl_AsIs)
    writer.Write(path)
    logger.info("Exported STEP (OCP): %s", path)


def _export_iges_ocp(mesh, path: str):
    from OCP.IGESControl import IGESControl_Writer
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
    from OCP.gp import gp_Pnt

    sewing = BRepBuilderAPI_Sewing()
    verts = mesh.vertices
    for face in mesh.faces:
        pts = [gp_Pnt(float(verts[i][0]), float(verts[i][1]), float(verts[i][2])) for i in face]
        poly = BRepBuilderAPI_MakePolygon()
        for p in pts:
            poly.Add(p)
        poly.Close()
        if poly.IsDone():
            f = BRepBuilderAPI_MakeFace(poly.Wire())
            if f.IsDone():
                sewing.Add(f.Face())
    sewing.Perform()
    writer = IGESControl_Writer()
    writer.AddShape(sewing.SewedShape())
    writer.Write(path)
    logger.info("Exported IGES (OCP): %s", path)
