use crate::meshio::Vec3;
use crate::extract::{EmbeddedGraph, TaggedLink};
use crate::field::{
    FieldState, RoSy4,
};
use std::collections::{HashMap, HashSet};

#[derive(Clone)]
struct CollapseEdge {
    a: usize,
    b: usize,
    error: f64,
}

pub fn extract_graph<M: RoSy4>(state: &FieldState) -> EmbeddedGraph {
    let inv_scale = 1.0 / state.scale;
    let mut adjacency = vec![HashSet::<usize>::new(); state.positions.len()];
    let mut collapse_edges = Vec::new();

    for i in 0..state.positions.len() {
        for link in &state.adjacency[i] {
            let j = link.id;
            if j < i {
                continue;
            }
            let (q_i, q_j) = M::match_orientation(
                state.orientations[i],
                state.normals[i],
                state.orientations[j],
                state.normals[j],
            );
            let (_, shift_j, error) = M::match_position_index(
                state.positions[i],
                state.normals[i],
                q_i,
                state.origins[i],
                state.positions[j],
                state.normals[j],
                q_j,
                state.origins[j],
                state.scale,
                inv_scale,
            );
            let abs_diff = shift_j.map(|x| x.abs());
            if abs_diff.x.max(abs_diff.y) > 1 || (abs_diff.x == 1 && abs_diff.y == 1) {
                continue;
            }
            if abs_diff.x + abs_diff.y == 0 {
                collapse_edges.push(CollapseEdge { a: i, b: j, error });
            } else {
                adjacency[i].insert(j);
                adjacency[j].insert(i);
            }
        }
    }

    collapse_edges.sort_by(|lhs, rhs| lhs.error.total_cmp(&rhs.error));
    let mut dsu = Dsu::new(state.positions.len());
    let mut collapse_counts = vec![0usize; state.positions.len()];
    for edge in collapse_edges {
        let a = dsu.find(edge.a);
        let b = dsu.find(edge.b);
        if a == b || has_cluster_edge(&adjacency, &dsu, a, b) {
            continue;
        }
        let merged = dsu.union(a, b);
        let other = if merged == a { b } else { a };
        let neighbors = adjacency[merged]
            .iter()
            .chain(adjacency[other].iter())
            .copied()
            .map(|n| dsu.find(n))
            .filter(|&n| n != merged && n != other)
            .collect::<HashSet<_>>();
        adjacency[merged].clear();
        adjacency[other].clear();
        adjacency[merged] = neighbors;
        collapse_counts[merged] = collapse_counts[a] + collapse_counts[b] + 1;
    }

    let mut root_to_index = HashMap::new();
    for i in 0..state.positions.len() {
        let root = dsu.find(i);
        if adjacency[root].is_empty() {
            continue;
        }
        let next = root_to_index.len();
        root_to_index.entry(root).or_insert(next);
    }

    let mut positions = vec![Vec3::zeros(); root_to_index.len()];
    let mut normals = vec![Vec3::zeros(); root_to_index.len()];
    let mut weights = vec![0.0; root_to_index.len()];
    let mut cluster_counts = vec![0usize; root_to_index.len()];
    for i in 0..state.positions.len() {
        let root = dsu.find(i);
        let Some(&cluster) = root_to_index.get(&root) else {
            continue;
        };
        let weight = (-((state.origins[i] - state.positions[i]).norm_squared()) * inv_scale * inv_scale * 9.0).exp();
        positions[cluster] += state.origins[i] * weight;
        normals[cluster] += state.normals[i] * weight;
        weights[cluster] += weight;
        cluster_counts[cluster] = collapse_counts[root];
    }

    for i in 0..positions.len() {
        if weights[i] > 0.0 {
            positions[i] /= weights[i];
            normals[i] = normals[i].normalize();
        }
    }

    let mut graph_adjacency = vec![Vec::new(); root_to_index.len()];
    for (&root, &index) in &root_to_index {
        let mut neighbors = adjacency[root]
            .iter()
            .filter_map(|&neighbor| root_to_index.get(&dsu.find(neighbor)).copied())
            .filter(|&neighbor| neighbor != index)
            .collect::<HashSet<_>>();
        graph_adjacency[index] = neighbors.drain().map(TaggedLink::new).collect();
    }

    let mut graph = EmbeddedGraph {
        positions,
        normals,
        adjacency: graph_adjacency,
        crease: HashSet::new(),
    };
    graph.cleanup(Some(&cluster_counts), state.scale, 4);
    graph.orient_edges();
    graph
}

fn has_cluster_edge(adjacency: &[HashSet<usize>], dsu: &Dsu, a: usize, b: usize) -> bool {
    adjacency[a].iter().any(|&neighbor| dsu.find(neighbor) == b)
        || adjacency[b].iter().any(|&neighbor| dsu.find(neighbor) == a)
}

struct Dsu {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl Dsu {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }

    fn find(&self, mut x: usize) -> usize {
        while self.parent[x] != x {
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) -> usize {
        let mut a = self.find(a);
        let mut b = self.find(b);
        if a == b {
            return a;
        }
        if self.rank[a] < self.rank[b] {
            std::mem::swap(&mut a, &mut b);
        }
        self.parent[b] = a;
        if self.rank[a] == self.rank[b] {
            self.rank[a] += 1;
        }
        a
    }
}
