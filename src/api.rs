use crate::boundary::{build_boundary_constraints, build_boundary_hierarchy};
use crate::field::{
    freeze_orientation_ivars, freeze_position_ivars, initialize_state, optimize_orientations,
    optimize_orientations_frozen, optimize_positions, optimize_positions_frozen,
    BoundaryConstraint, FieldState,
};
use crate::graph::extract_graph;
use crate::hierarchy::{build_hierarchy, prolong_orientations, prolong_origins, HierarchyLevel};
use crate::meshio::{triangulate_faces, Mesh};
use crate::metrics::{
    abs_volume, area, boundary_edge_count, face_count, fewer_than_three_face_count,
    invalid_quad_face_count, invalid_vertex_index_face_count, isolated_vertex_count,
    non_quad_face_count, nonmanifold_edge_count, quad_face_count, ratio,
    repeated_vertex_face_count, vertex_count,
};
use crate::preprocess::{
    compute_dual_vertex_areas, compute_mesh_stats, generate_smooth_normals,
    generate_uniform_adjacency, preprocess_mesh,
};
use crate::rng::{tag, Rng};
use crate::rotational_symmetry::{Extrinsic, Intrinsic, RoSy4};
use crate::topology::{build_directed_edges, TriMesh};
use std::error::Error;
use std::fmt;

const HIERARCHY_LEVEL_TAG: u64 = tag("hierarchy-level");

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RemeshTarget {
    EdgeLength(f64),
    VertexCount(usize),
    FaceCount(usize),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RemeshMode {
    #[default]
    Intrinsic,
    Extrinsic,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RemeshOptions {
    pub target: RemeshTarget,
    pub hierarchy_orientation_iterations: usize,
    pub hierarchy_position_iterations: usize,
    pub orientation_iterations: usize,
    pub position_iterations: usize,
    pub frozen_orientation_iterations: usize,
    pub frozen_position_iterations: usize,
    pub seed: Option<u64>,
    pub mode: RemeshMode,
}

impl RemeshOptions {
    pub fn new(target: RemeshTarget) -> Self {
        Self {
            target,
            hierarchy_orientation_iterations: 6,
            hierarchy_position_iterations: 6,
            orientation_iterations: 40,
            position_iterations: 80,
            frozen_orientation_iterations: 20,
            frozen_position_iterations: 20,
            seed: None,
            mode: RemeshMode::Intrinsic,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MeshReport {
    pub vertex_count: usize,
    pub face_count: usize,
    pub quad_face_count: usize,
    pub non_quad_face_count: usize,
    pub area: f64,
    pub abs_volume: f64,
    pub boundary_edge_count: usize,
    pub nonmanifold_edge_count: usize,
    pub fewer_than_three_face_count: usize,
    pub repeated_vertex_face_count: usize,
    pub invalid_vertex_index_face_count: usize,
    pub invalid_quad_face_count: usize,
    pub isolated_vertex_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RemeshResult {
    pub mesh: Mesh,
    pub seed: u64,
    pub input: MeshReport,
    pub output: MeshReport,
    pub area_ratio: Option<f64>,
    pub volume_ratio: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemeshError {
    EmptyMesh,
    ZeroSurfaceArea,
    InvalidTarget,
}

impl fmt::Display for RemeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyMesh => f.write_str("mesh must contain at least one face"),
            Self::ZeroSurfaceArea => f.write_str("mesh must have positive surface area"),
            Self::InvalidTarget => f.write_str("remesh target must be positive"),
        }
    }
}

impl Error for RemeshError {}

pub fn analyze_mesh(mesh: &Mesh) -> MeshReport {
    MeshReport {
        vertex_count: vertex_count(mesh),
        face_count: face_count(mesh),
        quad_face_count: quad_face_count(mesh),
        non_quad_face_count: non_quad_face_count(mesh),
        area: area(mesh),
        abs_volume: abs_volume(mesh),
        boundary_edge_count: boundary_edge_count(mesh),
        nonmanifold_edge_count: nonmanifold_edge_count(mesh),
        fewer_than_three_face_count: fewer_than_three_face_count(mesh),
        repeated_vertex_face_count: repeated_vertex_face_count(mesh),
        invalid_vertex_index_face_count: invalid_vertex_index_face_count(mesh),
        invalid_quad_face_count: invalid_quad_face_count(mesh),
        isolated_vertex_count: isolated_vertex_count(mesh),
    }
}

pub fn remesh(mesh: &Mesh, options: &RemeshOptions) -> Result<RemeshResult, RemeshError> {
    let input = analyze_mesh(mesh);
    let tri_mesh = triangulated_mesh(mesh)?;
    let scale = target_scale(options, compute_mesh_stats(&tri_mesh).surface_area)?;
    let tri_mesh = preprocess_mesh(&tri_mesh, scale);
    let dedges = build_directed_edges(&tri_mesh);
    let adjacency = generate_uniform_adjacency(&tri_mesh, &dedges);
    let normals = generate_smooth_normals(&tri_mesh);
    let areas = compute_dual_vertex_areas(&tri_mesh, &dedges);
    let levels = build_hierarchy(&tri_mesh.vertices, &normals, &areas, &adjacency);
    let boundaries = build_boundary_hierarchy(
        &levels,
        build_boundary_constraints(&tri_mesh, &dedges, &normals),
    );
    let seed = options.seed.map(Rng::new).unwrap_or_else(Rng::from_time);
    let mesh = match options.mode {
        RemeshMode::Intrinsic => {
            extract_mesh::<Intrinsic>(&levels, &boundaries, scale, options, seed)
        }
        RemeshMode::Extrinsic => {
            extract_mesh::<Extrinsic>(&levels, &boundaries, scale, options, seed)
        }
    };
    let output = analyze_mesh(&mesh);
    Ok(RemeshResult {
        mesh,
        seed: seed.value(),
        area_ratio: ratio(output.area, input.area),
        volume_ratio: ratio(output.abs_volume, input.abs_volume),
        input,
        output,
    })
}

fn triangulated_mesh(mesh: &Mesh) -> Result<TriMesh, RemeshError> {
    if mesh.faces.is_empty() {
        return Err(RemeshError::EmptyMesh);
    }
    let faces = triangulate_faces(&mesh.faces);
    if faces.is_empty() {
        return Err(RemeshError::EmptyMesh);
    }
    Ok(TriMesh {
        vertices: mesh.vertices.clone(),
        faces,
    })
}

fn target_scale(options: &RemeshOptions, surface_area: f64) -> Result<f64, RemeshError> {
    if !surface_area.is_finite() || surface_area <= 0.0 {
        return Err(RemeshError::ZeroSurfaceArea);
    }
    match options.target {
        RemeshTarget::EdgeLength(length) if length.is_finite() && length > 0.0 => Ok(length),
        RemeshTarget::VertexCount(count) if count > 0 => Ok((surface_area / count as f64).sqrt()),
        RemeshTarget::FaceCount(count) if count > 0 => Ok((surface_area / count as f64).sqrt()),
        _ => Err(RemeshError::InvalidTarget),
    }
}

fn extract_mesh<M: RoSy4>(
    levels: &[HierarchyLevel],
    boundaries: &[Vec<Option<BoundaryConstraint>>],
    scale: f64,
    options: &RemeshOptions,
    seed: Rng,
) -> Mesh {
    let state = solve_hierarchy::<M>(levels, boundaries, scale, options, seed);
    let quad_mesh = extract_graph::<M>(&state).extract_pure_quad_mesh(4, true);
    Mesh {
        vertices: quad_mesh.positions,
        faces: quad_mesh
            .quads
            .into_iter()
            .map(|face| face.to_vec())
            .collect(),
    }
}

fn solve_hierarchy<M: RoSy4>(
    levels: &[HierarchyLevel],
    boundaries: &[Vec<Option<BoundaryConstraint>>],
    scale: f64,
    options: &RemeshOptions,
    seed: Rng,
) -> FieldState {
    let mut states = levels
        .iter()
        .enumerate()
        .map(|(i, level)| {
            initialize_state(
                level.positions.clone(),
                level.normals.clone(),
                level.adjacency.clone(),
                boundaries[i].clone(),
                scale,
                seed.mix(HIERARCHY_LEVEL_TAG).mix(i as u64 + 1),
            )
        })
        .collect::<Vec<_>>();

    for level_idx in (0..levels.len()).rev() {
        for _ in 0..options.hierarchy_orientation_iterations {
            optimize_orientations::<M>(&mut states[level_idx], &levels[level_idx].phases, 1);
        }
        if level_idx > 0 {
            states[level_idx - 1].orientations = prolong_orientations(
                &levels[level_idx],
                &levels[level_idx - 1],
                &states[level_idx].orientations,
            );
        }
    }
    optimize_orientations::<M>(
        &mut states[0],
        &levels[0].phases,
        options.orientation_iterations,
    );
    freeze_orientation_ivars::<M>(&mut states[0]);
    optimize_orientations_frozen(
        &mut states[0],
        &levels[0].phases,
        options.frozen_orientation_iterations,
    );

    for level_idx in (0..levels.len()).rev() {
        for _ in 0..options.hierarchy_position_iterations {
            optimize_positions::<M>(&mut states[level_idx], &levels[level_idx].phases, 1);
        }
        if level_idx > 0 {
            states[level_idx - 1].origins = prolong_origins(
                &levels[level_idx],
                &levels[level_idx - 1],
                &states[level_idx].origins,
            );
        }
    }
    optimize_positions::<M>(
        &mut states[0],
        &levels[0].phases,
        options.position_iterations,
    );
    freeze_position_ivars::<M>(&mut states[0]);
    optimize_positions_frozen(
        &mut states[0],
        &levels[0].phases,
        options.frozen_position_iterations,
    );

    states.remove(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vec3;

    fn unit_square() -> Mesh {
        Mesh {
            vertices: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            faces: vec![vec![0, 1, 2], vec![0, 2, 3]],
        }
    }

    #[test]
    fn rejects_empty_mesh() {
        let err = remesh(
            &Mesh {
                vertices: Vec::new(),
                faces: Vec::new(),
            },
            &RemeshOptions::new(RemeshTarget::FaceCount(4)),
        )
        .unwrap_err();
        assert_eq!(err, RemeshError::EmptyMesh);
    }

    #[test]
    fn remeshes_unit_square() {
        let mut options = RemeshOptions::new(RemeshTarget::FaceCount(4));
        options.seed = Some(1337);
        let result = remesh(&unit_square(), &options).unwrap();
        assert_eq!(result.seed, 1337);
        assert!(!result.mesh.faces.is_empty());
        assert!(result.mesh.faces.iter().all(|face| face.len() == 4));
    }
}
