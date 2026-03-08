use crate::connectivity::quad_edge_counts;
use crate::field::coordinate_system;
use crate::geom::quad_is_valid;
use crate::meshio::Vec3;
use std::collections::VecDeque;

pub fn repair_quads(positions: &mut [Vec3], quads: &mut [[usize; 4]]) {
    for face in quads {
        let centroid =
            (positions[face[0]] + positions[face[1]] + positions[face[2]] + positions[face[3]]) / 4.0;
        let mut normal = Vec3::zeros();
        for i in 0..4 {
            let a = positions[face[i]] - centroid;
            let b = positions[face[(i + 1) % 4]] - centroid;
            normal += a.cross(&b);
        }
        if normal.norm_squared() <= 1e-12 {
            continue;
        }
        let (s, t) = coordinate_system(normal.normalize());
        let mut ordered = face
            .iter()
            .copied()
            .map(|index| {
                let d = positions[index] - centroid;
                (index, t.dot(&d).atan2(s.dot(&d)))
            })
            .collect::<Vec<_>>();
        ordered.sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1));
        *face = [ordered[0].0, ordered[1].0, ordered[2].0, ordered[3].0];
    }
}

pub fn fill_small_boundary_loops(positions: &mut Vec<Vec3>, quads: &mut Vec<[usize; 4]>, max_len: usize) {
    let mut boundary = vec![Vec::<usize>::new(); positions.len()];
    for ((a, b), count) in quad_edge_counts(quads) {
        if count != 1 {
            continue;
        }
        boundary[a].push(b);
        boundary[b].push(a);
    }

    let mut visited = vec![false; positions.len()];
    let mut additions = Vec::new();
    for start in 0..boundary.len() {
        if visited[start] || boundary[start].is_empty() {
            continue;
        }
        let mut component = Vec::new();
        let mut queue = VecDeque::from([start]);
        visited[start] = true;
        while let Some(vertex) = queue.pop_front() {
            component.push(vertex);
            for &neighbor in &boundary[vertex] {
                if visited[neighbor] {
                    continue;
                }
                visited[neighbor] = true;
                queue.push_back(neighbor);
            }
        }
        if component.len() < 4 || component.len() > max_len || component.iter().any(|&v| boundary[v].len() != 2) {
            continue;
        }

        let mut ordered = vec![component[0]];
        let mut prev = usize::MAX;
        while ordered.len() < component.len() {
            let current = *ordered.last().unwrap();
            let next = boundary[current]
                .iter()
                .copied()
                .find(|&neighbor| neighbor != prev && !ordered.contains(&neighbor));
            let Some(next) = next else {
                break;
            };
            prev = current;
            ordered.push(next);
        }
        if ordered.len() != component.len() {
            continue;
        }
        if ordered.len() == 4 {
            let mut face = [ordered[0], ordered[1], ordered[2], ordered[3]];
            repair_quads(positions, std::slice::from_mut(&mut face));
            if quad_is_valid(positions, face) {
                additions.push(face);
            }
            continue;
        }

        let center = ordered
            .iter()
            .copied()
            .map(|index| positions[index])
            .fold(Vec3::zeros(), |acc, p| acc + p)
            / ordered.len() as f64;
        let center_index = positions.len();
        positions.push(center);
        let mut mids = Vec::with_capacity(ordered.len());
        for i in 0..ordered.len() {
            let a = ordered[i];
            let b = ordered[(i + 1) % ordered.len()];
            let mid_index = positions.len();
            positions.push((positions[a] + positions[b]) * 0.5);
            mids.push(mid_index);
        }
        for i in 0..ordered.len() {
            additions.push([
                mids[i],
                ordered[(i + 1) % ordered.len()],
                mids[(i + 1) % ordered.len()],
                center_index,
            ]);
        }
    }
    quads.extend(additions);
}

pub fn compact_quads(positions: &mut Vec<Vec3>, quads: &mut Vec<[usize; 4]>) {
    let mut used = vec![false; positions.len()];
    for face in quads.iter() {
        for &index in face {
            used[index] = true;
        }
    }
    let mut remap = vec![usize::MAX; positions.len()];
    let mut compact = Vec::new();
    for (i, position) in positions.iter().copied().enumerate() {
        if !used[i] {
            continue;
        }
        remap[i] = compact.len();
        compact.push(position);
    }
    for face in quads.iter_mut() {
        for index in face {
            *index = remap[*index];
        }
    }
    *positions = compact;
}
