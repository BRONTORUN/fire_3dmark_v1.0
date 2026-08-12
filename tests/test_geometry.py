import numpy as np

from fire3dmark.geometry import (
    cone_geometry,
    convex_hull,
    cuboid_geometry,
    project_weak_perspective,
    rotation_6d,
    rotation_matrix_xyz,
)


def test_rotation_is_orthonormal():
    r = rotation_matrix_xyz(17.0, -31.0, 42.0)
    np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(r), 1.0, atol=1e-10)


def test_cone_canonical_height_and_radius():
    vertices, _ = cone_geometry(0.3, segments=32)
    base = vertices[:-1]
    tip = vertices[-1]
    assert np.isclose(tip[1], 1.0)
    np.testing.assert_allclose(np.sqrt(base[:, 0] ** 2 + base[:, 2] ** 2), 0.3)


def test_cuboid_canonical_height():
    vertices, edges = cuboid_geometry(0.6, 0.4)
    assert len(vertices) == 8
    assert len(edges) == 12
    assert np.isclose(vertices[:, 1].min(), 0.0)
    assert np.isclose(vertices[:, 1].max(), 1.0)


def test_projection_uses_image_normalized_alignment():
    p = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    uv, _ = project_weak_perspective(p, np.eye(3), 0.5, 0.75, 0.25, 1000, 800)
    np.testing.assert_allclose(uv[0], [500.0, 600.0])
    np.testing.assert_allclose(uv[1], [500.0, 400.0])


def test_convex_hull_drops_interior_points():
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]])
    hull = convex_hull(pts)
    assert len(hull) == 4


def test_rotation_6d_has_six_values():
    assert len(rotation_6d(np.eye(3))) == 6
