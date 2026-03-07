use std::collections::{HashMap, VecDeque};

pub fn face_edges(face: &[usize]) -> Vec<(usize, usize)> {
    let mut edges = Vec::with_capacity(face.len());
    for i in 0..face.len() {
        edges.push(edge_key(face[i], face[(i + 1) % face.len()]));
    }
    edges
}

pub fn quad_edge_counts(quads: &[[usize; 4]]) -> HashMap<(usize, usize), usize> {
    let mut counts = HashMap::new();
    for face in quads {
        for i in 0..4 {
            *counts.entry(edge_key(face[i], face[(i + 1) % 4])).or_insert(0) += 1;
        }
    }
    counts
}

pub fn boundary_edges(edge_counts: &HashMap<(usize, usize), usize>) -> Vec<(usize, usize)> {
    edge_counts
        .iter()
        .filter_map(|(&edge, &count)| (count == 1).then_some(edge))
        .collect()
}

pub fn neighbors_from_edges(vertex_count: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut neighbors = vec![Vec::new(); vertex_count];
    for &(a, b) in edges {
        connect_undirected(&mut neighbors, a, b);
    }
    neighbors
}

pub fn vertex_neighbors_from_quads(vertex_count: usize, quads: &[[usize; 4]]) -> Vec<Vec<usize>> {
    let mut neighbors = vec![Vec::new(); vertex_count];
    for face in quads {
        for i in 0..4 {
            connect_undirected(&mut neighbors, face[i], face[(i + 1) % 4]);
        }
    }
    neighbors
}

pub fn boundary_neighbors_from_quads(vertex_count: usize, quads: &[[usize; 4]]) -> Vec<Vec<usize>> {
    neighbors_from_edges(vertex_count, &boundary_edges(&quad_edge_counts(quads)))
}

pub fn count_components(neighbors: &[Vec<usize>], active: &[bool]) -> usize {
    let mut visited = vec![false; neighbors.len()];
    let mut components = 0;

    for start in 0..neighbors.len() {
        if visited[start] || !active[start] {
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

pub fn edge_key(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

fn connect_undirected(neighbors: &mut [Vec<usize>], a: usize, b: usize) {
    if !neighbors[a].contains(&b) {
        neighbors[a].push(b);
    }
    if !neighbors[b].contains(&a) {
        neighbors[b].push(a);
    }
}
