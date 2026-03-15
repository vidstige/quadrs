use crate::connectivity::{boundary_edges, face_edges};
use crate::geom::quad_is_valid;
use crate::meshio::{triangulate_faces, ObjMesh, Vec3};
use std::collections::{HashMap, HashSet};

const EPS: f64 = 1e-12;

#[derive(Clone, Debug)]
pub struct MeshReport {
    pub vertex_count: usize,
    pub face_count: usize,
    pub quad_faces: usize,
    pub non_quad_faces: usize,
    pub area: f64,
    pub abs_volume: f64,
    pub boundary_edges: usize,
    pub nonmanifold_edges: usize,
    pub fewer_than_three_faces: usize,
    pub repeated_vertex_faces: usize,
    pub invalid_vertex_index_faces: usize,
    pub invalid_quad_faces: usize,
    pub isolated_vertices: usize,
}

pub fn analyze(mesh: &ObjMesh) -> MeshReport {
    let triangles = triangulate_faces(&mesh.faces);
    let mut edge_counts = HashMap::<(usize, usize), usize>::new();
    let mut vertex_neighbors = vec![Vec::new(); mesh.vertices.len()];
    let mut used_vertices = vec![false; mesh.vertices.len()];
    let mut quad_faces = 0;
    let mut fewer_than_three_faces = 0;
    let mut repeated_vertex_faces = 0;
    let mut invalid_vertex_index_faces = 0;
    let mut invalid_quad_faces = 0;
    let mut area = 0.0;
    let mut signed_volume = 0.0;

    for face in &mesh.faces {
        if face.len() == 4 {
            quad_faces += 1;
        }
        fewer_than_three_faces += has_too_few_vertices(face) as usize;
        repeated_vertex_faces += has_repeated_vertex(face) as usize;
        invalid_vertex_index_faces += has_invalid_vertex_index(face, mesh.vertices.len()) as usize;
        invalid_quad_faces += has_invalid_quad(face, &mesh.vertices) as usize;
        for &vertex in face {
            if vertex < used_vertices.len() {
                used_vertices[vertex] = true;
            }
        }
        for edge in face_edges(face) {
            *edge_counts.entry(edge).or_insert(0) += 1;
            if !vertex_neighbors[edge.0].contains(&edge.1) {
                vertex_neighbors[edge.0].push(edge.1);
            }
            if !vertex_neighbors[edge.1].contains(&edge.0) {
                vertex_neighbors[edge.1].push(edge.0);
            }
        }
    }

    for [a, b, c] in triangles {
        let pa = mesh.vertices[a];
        let pb = mesh.vertices[b];
        let pc = mesh.vertices[c];
        area += 0.5 * (pb - pa).cross(&(pc - pa)).norm();
        signed_volume += pa.dot(&pb.cross(&pc)) / 6.0;
    }

    let boundary_edges_list = boundary_edges(&edge_counts);
    let boundary_edges = boundary_edges_list.len();
    let nonmanifold_edges = edge_counts.values().filter(|&&count| count > 2).count();
    let isolated_vertices = vertex_neighbors.iter().filter(|neighbors| neighbors.is_empty()).count();

    MeshReport {
        vertex_count: mesh.vertices.len(),
        face_count: mesh.faces.len(),
        quad_faces,
        non_quad_faces: mesh.faces.len().saturating_sub(quad_faces),
        area,
        abs_volume: signed_volume.abs(),
        boundary_edges,
        nonmanifold_edges,
        fewer_than_three_faces,
        repeated_vertex_faces,
        invalid_vertex_index_faces,
        invalid_quad_faces,
        isolated_vertices,
    }
}

pub fn ratio(output: f64, input: f64) -> Option<f64> {
    (input.abs() > EPS).then_some(output / input)
}

fn has_too_few_vertices(face: &[usize]) -> bool {
    face.len() < 3
}

fn has_repeated_vertex(face: &[usize]) -> bool {
    let distinct: HashSet<_> = face.iter().copied().collect();
    distinct.len() != face.len()
}

fn has_invalid_vertex_index(face: &[usize], vertex_count: usize) -> bool {
    face.iter().any(|&index| index >= vertex_count)
}

fn has_invalid_quad(face: &[usize], vertices: &[Vec3]) -> bool {
    face.len() == 4
        && !has_repeated_vertex(face)
        && !has_invalid_vertex_index(face, vertices.len())
        && !quad_is_valid(vertices, [face[0], face[1], face[2], face[3]])
}
