use remesh::hierarchy::{build_hierarchy, prolong_origins, prolong_orientations, HierarchyLevel};
use remesh::meshio::{load_obj, triangulate_faces, write_obj, ObjMesh};
use remesh::metrics::{analyze, ratio, MeshReport};
use remesh::graph::extract_graph;
use remesh::field::{
    coordinate_system, freeze_orientation_ivars, freeze_position_ivars, initialize_state,
    optimize_orientations, optimize_orientations_frozen, optimize_positions,
    optimize_positions_frozen, rotate_vector_into_plane, BoundaryConstraint, NativeState,
};
use remesh::preprocess::{
    as_trimesh, compute_dual_vertex_areas, compute_mesh_stats, dedge, generate_smooth_normals,
    generate_uniform_adjacency, subdivide_to_max_edge,
};
use remesh::topology::{DirectedEdges, TriMesh, INVALID};
use remesh::meshio::Vec3;
use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::collections::{HashMap, VecDeque};

fn main() {
    if matches!(env::args().nth(1).as_deref(), Some("-h" | "--help")) {
        println!("{}", usage());
        return;
    }
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = parse_args(env::args().skip(1))?;
    let input = load_obj(&args.input)?;
    let input_report = analyze(&input);
    print_report("input", &input_report, None);

    let tri_mesh = as_trimesh(input.vertices, triangulate_faces(&input.faces));
    let scale = target_scale(&args, compute_mesh_stats(&tri_mesh).surface_area);
    let tri_mesh = preprocess_mesh(&tri_mesh, scale);
    let dedges = dedge(&tri_mesh);
    let adjacency = generate_uniform_adjacency(&tri_mesh, &dedges);
    let normals = generate_smooth_normals(&tri_mesh);
    let areas = compute_dual_vertex_areas(&tri_mesh, &dedges);
    let boundary_segments = build_boundary_segments(&tri_mesh, &dedges);
    let levels = build_hierarchy(&tri_mesh.vertices, &normals, &areas, &adjacency);
    let boundaries = build_boundary_hierarchy(&levels, build_boundary_constraints(&tri_mesh, &dedges, &normals));
    let best = pick_best_candidate(
        &levels,
        &boundaries,
        scale,
        &args,
        &input_report,
        &tri_mesh.vertices,
        &tri_mesh.faces,
        &boundary_segments,
    )?;
    eprintln!("selected restart seed {}", best.seed);
    write_obj(&args.output, &best.mesh.vertices, &best.mesh.faces)?;

    let output_report = best.report;
    print_report(
        "output",
        &output_report,
        Some((
            &input_report,
            ratio(output_report.area, input_report.area),
            ratio(output_report.abs_volume, input_report.abs_volume),
        )),
    );
    if output_report.face_count == 0 {
        return Err("remesher produced no faces".into());
    }
    if output_report.non_quad_faces > 0 {
        return Err(format!("remesher produced {} non-quad faces", output_report.non_quad_faces).into());
    }
    Ok(())
}

#[derive(Clone)]
struct Args {
    input: PathBuf,
    output: PathBuf,
    edge_length: Option<f64>,
    target_vertices: Option<usize>,
    target_faces: Option<usize>,
    restarts: usize,
    hierarchy_orientation_iters: usize,
    hierarchy_position_iters: usize,
    orientation_iters: usize,
    position_iters: usize,
    frozen_orientation_iters: usize,
    frozen_position_iters: usize,
    seed: u64,
    intrinsic: bool,
}

fn parse_args<I>(args: I) -> Result<Args, Box<dyn Error>>
where
    I: IntoIterator<Item = String>,
{
    let mut input = None;
    let mut output = None;
    let mut edge_length = None;
    let mut target_vertices = None;
    let mut target_faces = None;
    let mut restarts = 40;
    let mut hierarchy_orientation_iters = 6;
    let mut hierarchy_position_iters = 6;
    let mut orientation_iters = 40;
    let mut position_iters = 80;
    let mut frozen_orientation_iters = 20;
    let mut frozen_position_iters = 20;
    let mut seed = 1;
    let mut intrinsic = true;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-o" | "--output" => output = Some(PathBuf::from(next_value(&mut args, &arg)?)),
            "--edge-length" => edge_length = Some(next_value(&mut args, &arg)?.parse()?),
            "--target-vertices" => target_vertices = Some(next_value(&mut args, &arg)?.parse()?),
            "--target-faces" => target_faces = Some(next_value(&mut args, &arg)?.parse()?),
            "--restarts" => restarts = next_value(&mut args, &arg)?.parse()?,
            "--hierarchy-orientation-iters" => hierarchy_orientation_iters = next_value(&mut args, &arg)?.parse()?,
            "--hierarchy-position-iters" => hierarchy_position_iters = next_value(&mut args, &arg)?.parse()?,
            "--orientation-iters" => orientation_iters = next_value(&mut args, &arg)?.parse()?,
            "--position-iters" => position_iters = next_value(&mut args, &arg)?.parse()?,
            "--frozen-orientation-iters" => frozen_orientation_iters = next_value(&mut args, &arg)?.parse()?,
            "--frozen-position-iters" => frozen_position_iters = next_value(&mut args, &arg)?.parse()?,
            "--seed" => seed = next_value(&mut args, &arg)?.parse()?,
            "--intrinsic" => intrinsic = true,
            "--extrinsic" => intrinsic = false,
            _ if arg.starts_with('-') => return Err(format!("unknown flag: {arg}\n\n{}", usage()).into()),
            _ if input.is_none() => input = Some(PathBuf::from(arg)),
            _ => return Err(format!("unexpected argument: {arg}\n\n{}", usage()).into()),
        }
    }
    match (input, output) {
        (Some(input), Some(output)) if edge_length.is_some() || target_vertices.is_some() || target_faces.is_some() => Ok(Args {
            input,
            output,
            edge_length,
            target_vertices,
            target_faces,
            restarts,
            hierarchy_orientation_iters,
            hierarchy_position_iters,
            orientation_iters,
            position_iters,
            frozen_orientation_iters,
            frozen_position_iters,
            seed,
            intrinsic,
        }),
        _ => Err(usage().into()),
    }
}

fn next_value<I>(args: &mut I, flag: &str) -> Result<String, Box<dyn Error>>
where
    I: Iterator<Item = String>,
{
    args.next().ok_or_else(|| format!("missing value for {flag}").into())
}

fn target_scale(args: &Args, surface_area: f64) -> f64 {
    if let Some(length) = args.edge_length {
        length
    } else if let Some(faces) = args.target_faces {
        (surface_area / faces as f64).sqrt()
    } else {
        (surface_area / args.target_vertices.unwrap() as f64).sqrt()
    }
}

fn preprocess_mesh(mesh: &TriMesh, scale: f64) -> TriMesh {
    let stats = compute_mesh_stats(mesh);
    if stats.maximum_edge_length * 2.0 > scale || stats.maximum_edge_length > stats.average_edge_length * 2.0 {
        subdivide_to_max_edge(mesh, (scale * 0.5).min(stats.average_edge_length * 2.0))
    } else {
        mesh.clone()
    }
}

fn build_boundary_constraints(mesh: &TriMesh, dedges: &DirectedEdges, normals: &[remesh::meshio::Vec3]) -> Vec<Option<BoundaryConstraint>> {
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

fn build_boundary_segments(mesh: &TriMesh, dedges: &DirectedEdges) -> Vec<(Vec3, Vec3)> {
    let mut segments = Vec::new();
    for edge in 0..dedges.e2e.len() {
        if dedges.e2e[edge] != INVALID {
            continue;
        }
        let face = mesh.faces[edge / 3];
        let a = mesh.vertices[face[edge % 3]];
        let b = mesh.vertices[face[(edge + 1) % 3]];
        segments.push((a, b));
    }
    segments
}

fn build_boundary_hierarchy(
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
            tangents[parent] += rotate_vector_into_plane(constraint.tangent, fine.normals[i], coarse.normals[parent]) * weight;
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

struct Candidate {
    seed: u64,
    mesh: ObjMesh,
    report: MeshReport,
}

fn pick_best_candidate(
    levels: &[HierarchyLevel],
    boundaries: &[Vec<Option<BoundaryConstraint>>],
    scale: f64,
    args: &Args,
    input_report: &MeshReport,
    source_vertices: &[Vec3],
    source_faces: &[[usize; 3]],
    source_boundary: &[(Vec3, Vec3)],
) -> Result<Candidate, Box<dyn Error>> {
    let mut best = None;
    for restart in 0..args.restarts.max(1) {
        let seed = args.seed + restart as u64;
        let state = solve_hierarchy(levels, boundaries, scale, args, seed);
        let graph = extract_graph(&state, args.intrinsic);
        let mut quad_mesh = graph.extract_pure_quad_mesh(4, true);
        repair_quads(&mut quad_mesh.positions, &mut quad_mesh.quads);
        fill_small_boundary_loops(&mut quad_mesh.positions, &mut quad_mesh.quads, 7);
        repair_quads(&mut quad_mesh.positions, &mut quad_mesh.quads);
        smooth_and_reproject_invalid_quads(
            &mut quad_mesh.positions,
            &quad_mesh.quads,
            source_vertices,
            source_faces,
            source_boundary,
        );
        compact_quads(&mut quad_mesh.positions, &mut quad_mesh.quads);
        let mesh = ObjMesh {
            vertices: quad_mesh.positions,
            faces: quad_mesh.quads.into_iter().map(|face| face.to_vec()).collect(),
        };
        let report = analyze(&mesh);
        eprintln!(
            "restart seed {}: F={} boundary={} loops={} invalid={} area-ratio={:.3} volume-ratio={:.3}",
            seed,
            report.face_count,
            report.boundary_edges,
            report.boundary_loops,
            report.invalid_faces,
            ratio(report.area, input_report.area).unwrap_or(0.0),
            ratio(report.abs_volume, input_report.abs_volume).unwrap_or(0.0),
        );
        let candidate = Candidate { seed, mesh, report };
        if best.as_ref().map_or(true, |current| better_candidate(&candidate, current, input_report)) {
            best = Some(candidate);
        }
        if is_clean_candidate(best.as_ref().unwrap(), input_report) {
            break;
        }
    }
    best.ok_or_else(|| "remesher produced no candidate".into())
}

fn solve_hierarchy(
    levels: &[HierarchyLevel],
    boundaries: &[Vec<Option<BoundaryConstraint>>],
    scale: f64,
    args: &Args,
    seed: u64,
) -> NativeState {
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
                seed ^ ((i as u64 + 1) * 0x9e3779b97f4a7c15),
            )
        })
        .collect::<Vec<_>>();

    for level_idx in (0..levels.len()).rev() {
        for _ in 0..args.hierarchy_orientation_iters {
            optimize_orientations(&mut states[level_idx], &levels[level_idx].phases, 1, args.intrinsic);
        }
        if level_idx > 0 {
            states[level_idx - 1].orientations = prolong_orientations(
                &levels[level_idx],
                &levels[level_idx - 1],
                &states[level_idx].orientations,
            );
        }
    }
    optimize_orientations(&mut states[0], &levels[0].phases, args.orientation_iters, args.intrinsic);
    freeze_orientation_ivars(&mut states[0], args.intrinsic);
    optimize_orientations_frozen(&mut states[0], &levels[0].phases, args.frozen_orientation_iters);

    for level_idx in (0..levels.len()).rev() {
        for _ in 0..args.hierarchy_position_iters {
            optimize_positions(&mut states[level_idx], &levels[level_idx].phases, 1, args.intrinsic);
        }
        if level_idx > 0 {
            states[level_idx - 1].origins = prolong_origins(
                &levels[level_idx],
                &levels[level_idx - 1],
                &states[level_idx].origins,
            );
        }
    }
    optimize_positions(&mut states[0], &levels[0].phases, args.position_iters, args.intrinsic);
    freeze_position_ivars(&mut states[0], args.intrinsic);
    optimize_positions_frozen(&mut states[0], &levels[0].phases, args.frozen_position_iters);

    states.remove(0)
}

fn better_candidate(lhs: &Candidate, rhs: &Candidate, input_report: &MeshReport) -> bool {
    let lhs_components = lhs.report.connected_components.abs_diff(1);
    let rhs_components = rhs.report.connected_components.abs_diff(1);
    if lhs.report.non_quad_faces != rhs.report.non_quad_faces {
        return lhs.report.non_quad_faces < rhs.report.non_quad_faces;
    }
    if lhs.report.nonmanifold_edges != rhs.report.nonmanifold_edges {
        return lhs.report.nonmanifold_edges < rhs.report.nonmanifold_edges;
    }
    if lhs.report.invalid_faces != rhs.report.invalid_faces {
        return lhs.report.invalid_faces < rhs.report.invalid_faces;
    }
    if lhs_components != rhs_components {
        return lhs_components < rhs_components;
    }
    let lhs_loops = lhs.report.boundary_loops.abs_diff(input_report.boundary_loops);
    let rhs_loops = rhs.report.boundary_loops.abs_diff(input_report.boundary_loops);
    if lhs_loops != rhs_loops {
        return lhs_loops < rhs_loops;
    }
    let lhs_boundary = lhs.report.boundary_edges.abs_diff(input_report.boundary_edges);
    let rhs_boundary = rhs.report.boundary_edges.abs_diff(input_report.boundary_edges);
    if lhs_boundary != rhs_boundary {
        return lhs_boundary < rhs_boundary;
    }
    let lhs_area = (ratio(lhs.report.area, input_report.area).unwrap_or(0.0) - 1.0).abs();
    let rhs_area = (ratio(rhs.report.area, input_report.area).unwrap_or(0.0) - 1.0).abs();
    if (lhs_area - rhs_area).abs() > 1e-9 {
        return lhs_area < rhs_area;
    }
    let lhs_volume = (ratio(lhs.report.abs_volume, input_report.abs_volume).unwrap_or(0.0) - 1.0).abs();
    let rhs_volume = (ratio(rhs.report.abs_volume, input_report.abs_volume).unwrap_or(0.0) - 1.0).abs();
    if (lhs_volume - rhs_volume).abs() > 1e-9 {
        return lhs_volume < rhs_volume;
    }
    lhs.report.face_count > rhs.report.face_count
}

fn is_clean_candidate(candidate: &Candidate, input_report: &MeshReport) -> bool {
    candidate.report.non_quad_faces == 0
        && candidate.report.nonmanifold_edges == 0
        && candidate.report.invalid_faces == 0
        && candidate.report.connected_components == input_report.connected_components
        && candidate.report.boundary_loops == input_report.boundary_loops
}

fn print_report(label: &str, report: &MeshReport, baseline: Option<(&MeshReport, Option<f64>, Option<f64>)>) {
    eprintln!(
        "{label}: V={} F={} quads={} non-quads={} area={:.9} abs-volume={:.9}",
        report.vertex_count, report.face_count, report.quad_faces, report.non_quad_faces, report.area, report.abs_volume
    );
    eprintln!(
        "{label}: boundary edges={} loops={} non-manifold edges={} invalid faces={} duplicate faces={} isolated vertices={} components={}",
        report.boundary_edges, report.boundary_loops, report.nonmanifold_edges, report.invalid_faces, report.duplicate_faces, report.isolated_vertices, report.connected_components
    );
    if let Some((_, area_ratio, volume_ratio)) = baseline {
        if let Some(value) = area_ratio {
            eprintln!("{label}: area ratio vs input = {:.3}", value);
        }
        if let Some(value) = volume_ratio {
            eprintln!("{label}: volume ratio vs input = {:.3}", value);
        }
    }
}

fn usage() -> &'static str {
    "usage: remesh <input.obj> -o <output.obj> (--edge-length L | --target-vertices N | --target-faces N) [--restarts N] [--hierarchy-orientation-iters N] [--hierarchy-position-iters N] [--orientation-iters N] [--position-iters N] [--frozen-orientation-iters N] [--frozen-position-iters N] [--seed N] [--intrinsic | --extrinsic]"
}

fn repair_quads(positions: &mut [Vec3], quads: &mut [[usize; 4]]) {
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

fn fill_small_boundary_loops(positions: &mut Vec<Vec3>, quads: &mut Vec<[usize; 4]>, max_len: usize) {
    let mut edge_counts = HashMap::<(usize, usize), usize>::new();
    for face in quads.iter() {
        for i in 0..4 {
            let a = face[i];
            let b = face[(i + 1) % 4];
            let key = if a < b { (a, b) } else { (b, a) };
            *edge_counts.entry(key).or_insert(0) += 1;
        }
    }

    let mut boundary = vec![Vec::<usize>::new(); positions.len()];
    for ((a, b), count) in edge_counts {
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

fn smooth_and_reproject_invalid_quads(
    positions: &mut [Vec3],
    quads: &[[usize; 4]],
    source_vertices: &[Vec3],
    source_faces: &[[usize; 3]],
    source_boundary: &[(Vec3, Vec3)],
) {
    let neighbors = build_vertex_neighbors(positions.len(), quads);
    let boundary_neighbors = build_boundary_neighbors(positions.len(), quads);
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

fn build_vertex_neighbors(vertex_count: usize, quads: &[[usize; 4]]) -> Vec<Vec<usize>> {
    let mut neighbors = vec![Vec::new(); vertex_count];
    for face in quads {
        for i in 0..4 {
            let a = face[i];
            let b = face[(i + 1) % 4];
            if !neighbors[a].contains(&b) {
                neighbors[a].push(b);
            }
            if !neighbors[b].contains(&a) {
                neighbors[b].push(a);
            }
        }
    }
    neighbors
}

fn build_boundary_neighbors(vertex_count: usize, quads: &[[usize; 4]]) -> Vec<Vec<usize>> {
    let mut counts = HashMap::<(usize, usize), usize>::new();
    for face in quads {
        for i in 0..4 {
            let a = face[i];
            let b = face[(i + 1) % 4];
            let key = if a < b { (a, b) } else { (b, a) };
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    let mut neighbors = vec![Vec::new(); vertex_count];
    for ((a, b), count) in counts {
        if count != 1 {
            continue;
        }
        neighbors[a].push(b);
        neighbors[b].push(a);
    }
    neighbors
}

fn quad_is_valid(positions: &[Vec3], face: [usize; 4]) -> bool {
    let points = face.map(|index| positions[index]);
    let normal =
        (points[1] - points[0]).cross(&(points[2] - points[0])) + (points[2] - points[0]).cross(&(points[3] - points[0]));
    if normal.norm_squared() <= 1e-12 {
        return false;
    }
    let axis = dominant_axis(normal);
    let projected = points.map(|point| project_2d(point, axis));
    if segments_intersect(projected[0], projected[1], projected[2], projected[3])
        || segments_intersect(projected[1], projected[2], projected[3], projected[0])
    {
        return false;
    }
    triangle_area(points[0], points[1], points[2]) > 1e-12
        && triangle_area(points[0], points[2], points[3]) > 1e-12
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

fn dominant_axis(normal: Vec3) -> usize {
    let x = normal.x.abs();
    let y = normal.y.abs();
    let z = normal.z.abs();
    if x >= y && x >= z {
        0
    } else if y >= z {
        1
    } else {
        2
    }
}

fn project_2d(point: Vec3, axis: usize) -> (f64, f64) {
    match axis {
        0 => (point.y, point.z),
        1 => (point.x, point.z),
        _ => (point.x, point.y),
    }
}

fn segments_intersect(a0: (f64, f64), a1: (f64, f64), b0: (f64, f64), b1: (f64, f64)) -> bool {
    let o1 = orient2d(a0, a1, b0);
    let o2 = orient2d(a0, a1, b1);
    let o3 = orient2d(b0, b1, a0);
    let o4 = orient2d(b0, b1, a1);
    o1 * o2 < -1e-12 && o3 * o4 < -1e-12
}

fn orient2d(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> f64 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

fn triangle_area(a: Vec3, b: Vec3, c: Vec3) -> f64 {
    0.5 * (b - a).cross(&(c - a)).norm()
}

fn compact_quads(positions: &mut Vec<Vec3>, quads: &mut Vec<[usize; 4]>) {
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
