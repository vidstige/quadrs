use crate::field::{rotate_vector_into_plane, BoundaryConstraint};
use crate::hierarchy::HierarchyLevel;
use crate::meshio::Vec3;
use crate::topology::{DirectedEdges, TriMesh, INVALID};

pub fn build_boundary_constraints(
    mesh: &TriMesh,
    dedges: &DirectedEdges,
    normals: &[Vec3],
) -> Vec<Option<BoundaryConstraint>> {
    let mut constraints = vec![None; mesh.vertices.len()];
    for edge in 0..dedges.e2e.len() {
        if dedges.e2e[edge] != INVALID {
            continue;
        }
        let face = mesh.faces[edge / 3];
        let i0 = face[edge % 3];
        let i1 = face[(edge + 1) % 3];
        let direction = mesh.vertices[i1] - mesh.vertices[i0];
        if direction.norm_squared() <= 1e-12 {
            continue;
        }
        let tangent0 = (direction - normals[i0] * normals[i0].dot(&direction)).normalize();
        let tangent1 = (direction - normals[i1] * normals[i1].dot(&direction)).normalize();
        constraints[i0] = Some(BoundaryConstraint {
            origin: mesh.vertices[i0],
            tangent: tangent0,
            weight: 1.0,
        });
        constraints[i1] = Some(BoundaryConstraint {
            origin: mesh.vertices[i1],
            tangent: tangent1,
            weight: 1.0,
        });
    }
    constraints
}

pub fn build_boundary_hierarchy(
    levels: &[HierarchyLevel],
    fine_boundary: Vec<Option<BoundaryConstraint>>,
) -> Vec<Vec<Option<BoundaryConstraint>>> {
    let mut hierarchy = Vec::with_capacity(levels.len());
    hierarchy.push(fine_boundary);
    for level_idx in 0..levels.len().saturating_sub(1) {
        let fine = &levels[level_idx];
        let coarse = &levels[level_idx + 1];
        let to_coarser = fine.to_coarser.as_ref().unwrap();
        let mut origins = vec![Vec3::zeros(); coarse.positions.len()];
        let mut tangents = vec![Vec3::zeros(); coarse.positions.len()];
        let mut weights = vec![0.0; coarse.positions.len()];
        for (i, constraint) in hierarchy[level_idx].iter().enumerate() {
            let Some(constraint) = constraint else {
                continue;
            };
            let parent = to_coarser[i];
            let weight = fine.areas[i].max(1e-12) * constraint.weight.max(1e-12);
            origins[parent] += constraint.origin * weight;
            tangents[parent] +=
                rotate_vector_into_plane(constraint.tangent, fine.normals[i], coarse.normals[parent]) * weight;
            weights[parent] += weight;
        }

        let mut coarse_boundary = vec![None; coarse.positions.len()];
        for i in 0..coarse.positions.len() {
            if weights[i] == 0.0 {
                continue;
            }
            let normal = coarse.normals[i];
            let position = coarse.positions[i];
            let mut origin = origins[i] / weights[i];
            origin -= normal * normal.dot(&(origin - position));
            let mut tangent = tangents[i];
            tangent -= normal * normal.dot(&tangent);
            if tangent.norm_squared() <= 1e-12 {
                continue;
            }
            coarse_boundary[i] = Some(BoundaryConstraint {
                origin,
                tangent: tangent.normalize(),
                weight: 1.0,
            });
        }
        hierarchy.push(coarse_boundary);
    }
    hierarchy
}
