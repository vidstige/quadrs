use crate::connectivity::{boundary_neighbors_from_quads, quad_edge_counts, vertex_neighbors_from_quads};
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

pub fn smooth_and_reproject_invalid_quads(
    positions: &mut [Vec3],
    quads: &[[usize; 4]],
    source_vertices: &[Vec3],
    source_faces: &[[usize; 3]],
    source_boundary: &[(Vec3, Vec3)],
) {
    let neighbors = vertex_neighbors_from_quads(positions.len(), quads);
    let boundary_neighbors = boundary_neighbors_from_quads(positions.len(), quads);
    for _ in 0..4 {
        let invalid_faces = quads
            .iter()
            .enumerate()
            .filter_map(|(i, &face)| (!quad_is_valid(positions, face)).then_some(i))
            .collect::<Vec<_>>();
        if invalid_faces.is_empty() {
            break;
        }

        let mut affected = vec![false; positions.len()];
        for &face_idx in &invalid_faces {
            for &vertex in &quads[face_idx] {
                affected[vertex] = true;
                for &neighbor in &neighbors[vertex] {
                    affected[neighbor] = true;
                }
            }
        }

        let prev = positions.to_vec();
        for vertex in 0..positions.len() {
            if !affected[vertex] {
                continue;
            }
            if boundary_neighbors[vertex].len() >= 2 && !source_boundary.is_empty() {
                let mut target = Vec3::zeros();
                for &neighbor in &boundary_neighbors[vertex] {
                    target += prev[neighbor];
                }
                target /= boundary_neighbors[vertex].len() as f64;
                positions[vertex] = closest_point_on_segments(target, source_boundary);
                continue;
            }
            if neighbors[vertex].is_empty() {
                continue;
            }
            let mut target = Vec3::zeros();
            for &neighbor in &neighbors[vertex] {
                target += prev[neighbor];
            }
            target /= neighbors[vertex].len() as f64;
            let projected = closest_point_on_triangles(target, source_vertices, source_faces);
            positions[vertex] = prev[vertex] * 0.25 + projected * 0.75;
        }
    }
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

fn closest_point_on_triangles(point: Vec3, vertices: &[Vec3], faces: &[[usize; 3]]) -> Vec3 {
    let mut best = point;
    let mut best_dist = f64::INFINITY;
    for face in faces {
        let candidate = closest_point_on_triangle(point, vertices[face[0]], vertices[face[1]], vertices[face[2]]);
        let dist = (candidate - point).norm_squared();
        if dist < best_dist {
            best = candidate;
            best_dist = dist;
        }
    }
    best
}

fn closest_point_on_segments(point: Vec3, segments: &[(Vec3, Vec3)]) -> Vec3 {
    let mut best = point;
    let mut best_dist = f64::INFINITY;
    for &(a, b) in segments {
        let ab = b - a;
        let t = ((point - a).dot(&ab) / ab.norm_squared().max(1e-12)).clamp(0.0, 1.0);
        let candidate = a + ab * t;
        let dist = (candidate - point).norm_squared();
        if dist < best_dist {
            best = candidate;
            best_dist = dist;
        }
    }
    best
}

fn closest_point_on_triangle(point: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let ab = b - a;
    let ac = c - a;
    let ap = point - a;
    let d1 = ab.dot(&ap);
    let d2 = ac.dot(&ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }

    let bp = point - b;
    let d3 = ab.dot(&bp);
    let d4 = ac.dot(&bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }

    let cp = point - c;
    let d5 = ab.dot(&cp);
    let d6 = ac.dot(&cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let bc = c - b;
        return b + bc * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }

    let denom = 1.0 / (va + vb + vc);
    a + ab * (vb * denom) + ac * (vc * denom)
}
