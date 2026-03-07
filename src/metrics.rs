use crate::geom::{quad_is_valid, triangle_area};
use crate::meshio::{triangulate_faces, ObjMesh, Vec3};
use std::collections::{HashMap, HashSet, VecDeque};

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
    pub boundary_loops: usize,
    pub nonmanifold_edges: usize,
    pub invalid_faces: usize,
    pub duplicate_faces: usize,
    pub isolated_vertices: usize,
    pub connected_components: usize,
}

pub fn analyze(mesh: &ObjMesh) -> MeshReport {
    let triangles = triangulate_faces(&mesh.faces);
    let mut edge_counts = HashMap::<(usize, usize), usize>::new();
    let mut vertex_neighbors = vec![HashSet::<usize>::new(); mesh.vertices.len()];
    let mut used_vertices = vec![false; mesh.vertices.len()];
    let mut seen_faces = HashSet::<Vec<usize>>::new();
    let mut quad_faces = 0;
    let mut invalid_faces = 0;
    let mut duplicate_faces = 0;
    let mut area = 0.0;
    let mut signed_volume = 0.0;

    for face in &mesh.faces {
        if face.len() == 4 {
            quad_faces += 1;
        }
        if !is_valid_face(face, &mesh.vertices) {
            invalid_faces += 1;
        }
        if !seen_faces.insert(canonical_face(face)) {
            duplicate_faces += 1;
        }
        for &vertex in face {
            if vertex < used_vertices.len() {
                used_vertices[vertex] = true;
            }
        }
        for edge in face_edges(face) {
            *edge_counts.entry(edge).or_insert(0) += 1;
            vertex_neighbors[edge.0].insert(edge.1);
            vertex_neighbors[edge.1].insert(edge.0);
        }
    }

    for [a, b, c] in triangles {
        let pa = mesh.vertices[a];
        let pb = mesh.vertices[b];
        let pc = mesh.vertices[c];
        area += 0.5 * (pb - pa).cross(&(pc - pa)).norm();
        signed_volume += pa.dot(&pb.cross(&pc)) / 6.0;
    }

    let boundary_edges_list: Vec<_> = edge_counts
        .iter()
        .filter_map(|(&edge, &count)| (count == 1).then_some(edge))
        .collect();
    let boundary_edges = boundary_edges_list.len();
    let boundary_loops = count_boundary_components(mesh.vertices.len(), &boundary_edges_list);
    let nonmanifold_edges = edge_counts.values().filter(|&&count| count > 2).count();
    let isolated_vertices = vertex_neighbors.iter().filter(|neighbors| neighbors.is_empty()).count();
    let connected_components = count_components(&vertex_neighbors, &used_vertices);

    MeshReport {
        vertex_count: mesh.vertices.len(),
        face_count: mesh.faces.len(),
        quad_faces,
        non_quad_faces: mesh.faces.len().saturating_sub(quad_faces),
        area,
        abs_volume: signed_volume.abs(),
        boundary_edges,
        boundary_loops,
        nonmanifold_edges,
        invalid_faces,
        duplicate_faces,
        isolated_vertices,
        connected_components,
    }
}

pub fn ratio(output: f64, input: f64) -> Option<f64> {
    (input.abs() > EPS).then_some(output / input)
}

fn count_components(vertex_neighbors: &[HashSet<usize>], used_vertices: &[bool]) -> usize {
    let mut visited = vec![false; vertex_neighbors.len()];
    let mut components = 0;

    for start in 0..vertex_neighbors.len() {
        if visited[start] || !used_vertices[start] {
            continue;
        }
        components += 1;
        visited[start] = true;
        let mut queue = VecDeque::from([start]);
        while let Some(vertex) = queue.pop_front() {
            for &next in &vertex_neighbors[vertex] {
                if visited[next] {
                    continue;
                }
                visited[next] = true;
                queue.push_back(next);
            }
        }
    }

    components
}

fn count_boundary_components(vertex_count: usize, boundary_edges: &[(usize, usize)]) -> usize {
    let mut neighbors = vec![Vec::<usize>::new(); vertex_count];
    for &(a, b) in boundary_edges {
        neighbors[a].push(b);
        neighbors[b].push(a);
    }

    let mut visited = vec![false; vertex_count];
    let mut components = 0;
    for start in 0..vertex_count {
        if visited[start] || neighbors[start].is_empty() {
            continue;
        }
        components += 1;
        visited[start] = true;
        let mut queue = VecDeque::from([start]);
        while let Some(vertex) = queue.pop_front() {
            for &next in &neighbors[vertex] {
                if visited[next] {
                    continue;
                }
                visited[next] = true;
                queue.push_back(next);
            }
        }
    }
    components
}

fn is_valid_face(face: &[usize], vertices: &[Vec3]) -> bool {
    if face.len() < 3 {
        return false;
    }
    let distinct: HashSet<_> = face.iter().copied().collect();
    if distinct.len() != face.len() {
        return false;
    }
    if face.iter().any(|&index| index >= vertices.len()) {
        return false;
    }
    if face.len() == 4 {
        return quad_is_valid(vertices, [face[0], face[1], face[2], face[3]]);
    }

    let triangles = triangulate_faces(&[face.to_vec()]);
    !triangles.is_empty()
        && triangles.iter().all(|&tri| {
            let area = triangle_area(vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]);
            area > EPS
        })
}

fn face_edges(face: &[usize]) -> Vec<(usize, usize)> {
    let mut edges = Vec::with_capacity(face.len());
    for i in 0..face.len() {
        edges.push(edge_key(face[i], face[(i + 1) % face.len()]));
    }
    edges
}

fn edge_key(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

fn canonical_face(face: &[usize]) -> Vec<usize> {
    let mut best = rotate_face(face, min_index(face));
    let reversed: Vec<_> = face.iter().rev().copied().collect();
    let reversed_best = rotate_face(&reversed, min_index(&reversed));
    if reversed_best < best {
        best = reversed_best;
    }
    best
}

fn min_index(face: &[usize]) -> usize {
    let mut best = 0;
    for i in 1..face.len() {
        if face[i] < face[best] {
            best = i;
        }
    }
    best
}

fn rotate_face(face: &[usize], shift: usize) -> Vec<usize> {
    (0..face.len())
        .map(|i| face[(shift + i) % face.len()])
        .collect()
}
