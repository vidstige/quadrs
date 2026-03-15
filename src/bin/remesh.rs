use remesh::boundary::{build_boundary_constraints, build_boundary_hierarchy};
use remesh::hierarchy::{build_hierarchy, prolong_origins, prolong_orientations, HierarchyLevel};
use remesh::meshio::{load_obj, triangulate_faces, write_obj, ObjMesh};
use remesh::metrics::{
    abs_volume, area, boundary_edge_count, face_count, fewer_than_three_face_count,
    invalid_quad_face_count, invalid_vertex_index_face_count, isolated_vertex_count,
    non_quad_face_count, nonmanifold_edge_count, quad_face_count, ratio,
    repeated_vertex_face_count, vertex_count,
};
use remesh::graph::extract_graph;
use remesh::field::{
    freeze_orientation_ivars, freeze_position_ivars, initialize_state, optimize_orientations,
    optimize_orientations_frozen, optimize_positions, optimize_positions_frozen, BoundaryConstraint, NativeState,
};
use remesh::preprocess::{
    compute_dual_vertex_areas, compute_mesh_stats, generate_smooth_normals, generate_uniform_adjacency, preprocess_mesh,
};
use remesh::topology::{build_directed_edges, TriMesh};
use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

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
    let input_area = area(&input);
    let input_abs_volume = abs_volume(&input);
    print_report("input", &input, None);

    let tri_mesh = TriMesh {
        vertices: input.vertices,
        faces: triangulate_faces(&input.faces),
    };
    let scale = target_scale(&args, compute_mesh_stats(&tri_mesh).surface_area);
    let tri_mesh = preprocess_mesh(&tri_mesh, scale);
    let dedges = build_directed_edges(&tri_mesh);
    let adjacency = generate_uniform_adjacency(&tri_mesh, &dedges);
    let normals = generate_smooth_normals(&tri_mesh);
    let areas = compute_dual_vertex_areas(&tri_mesh, &dedges);
    let levels = build_hierarchy(&tri_mesh.vertices, &normals, &areas, &adjacency);
    let boundaries = build_boundary_hierarchy(&levels, build_boundary_constraints(&tri_mesh, &dedges, &normals));
    let seed = args.seed.unwrap_or_else(current_time_seed);
    let result = remesh_once(
        &levels,
        &boundaries,
        scale,
        &args,
        input_area,
        input_abs_volume,
        seed,
    )?;
    eprintln!("seed {}", result.seed);
    write_obj(&args.output, &result.mesh.vertices, &result.mesh.faces)?;

    print_report(
        "output",
        &result.mesh,
        Some((input_area, input_abs_volume)),
    );
    Ok(())
}

#[derive(Clone)]
struct Args {
    input: PathBuf,
    output: PathBuf,
    edge_length: Option<f64>,
    target_vertices: Option<usize>,
    target_faces: Option<usize>,
    hierarchy_orientation_iters: usize,
    hierarchy_position_iters: usize,
    orientation_iters: usize,
    position_iters: usize,
    frozen_orientation_iters: usize,
    frozen_position_iters: usize,
    seed: Option<u64>,
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
    let mut hierarchy_orientation_iters = 6;
    let mut hierarchy_position_iters = 6;
    let mut orientation_iters = 40;
    let mut position_iters = 80;
    let mut frozen_orientation_iters = 20;
    let mut frozen_position_iters = 20;
    let mut seed = None;
    let mut intrinsic = true;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-o" | "--output" => output = Some(PathBuf::from(next_value(&mut args, &arg)?)),
            "--edge-length" => edge_length = Some(next_value(&mut args, &arg)?.parse()?),
            "--target-vertices" => target_vertices = Some(next_value(&mut args, &arg)?.parse()?),
            "--target-faces" => target_faces = Some(next_value(&mut args, &arg)?.parse()?),
            "--hierarchy-orientation-iters" => hierarchy_orientation_iters = next_value(&mut args, &arg)?.parse()?,
            "--hierarchy-position-iters" => hierarchy_position_iters = next_value(&mut args, &arg)?.parse()?,
            "--orientation-iters" => orientation_iters = next_value(&mut args, &arg)?.parse()?,
            "--position-iters" => position_iters = next_value(&mut args, &arg)?.parse()?,
            "--frozen-orientation-iters" => frozen_orientation_iters = next_value(&mut args, &arg)?.parse()?,
            "--frozen-position-iters" => frozen_position_iters = next_value(&mut args, &arg)?.parse()?,
            "--seed" => seed = Some(next_value(&mut args, &arg)?.parse()?),
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

struct Candidate {
    seed: u64,
    mesh: ObjMesh,
}

fn remesh_once(
    levels: &[HierarchyLevel],
    boundaries: &[Vec<Option<BoundaryConstraint>>],
    scale: f64,
    args: &Args,
    input_area: f64,
    input_abs_volume: f64,
    seed: u64,
) -> Result<Candidate, Box<dyn Error>> {
    let state = solve_hierarchy(levels, boundaries, scale, args, seed);
    let graph = extract_graph(&state, args.intrinsic);
    let quad_mesh = graph.extract_pure_quad_mesh(4, true);
    let mesh = ObjMesh {
        vertices: quad_mesh.positions,
        faces: quad_mesh.quads.into_iter().map(|face| face.to_vec()).collect(),
    };
    eprintln!(
        "seed {}: F={} boundary={} invalid-lt3={} invalid-repeat={} invalid-index={} invalid-quad={} area-ratio={:.3} volume-ratio={:.3}",
        seed,
        face_count(&mesh),
        boundary_edge_count(&mesh),
        fewer_than_three_face_count(&mesh),
        repeated_vertex_face_count(&mesh),
        invalid_vertex_index_face_count(&mesh),
        invalid_quad_face_count(&mesh),
        ratio(area(&mesh), input_area).unwrap_or(0.0),
        ratio(abs_volume(&mesh), input_abs_volume).unwrap_or(0.0),
    );
    Ok(Candidate { seed, mesh })
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

fn current_time_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

fn print_report(label: &str, mesh: &ObjMesh, baseline: Option<(f64, f64)>) {
    let mesh_area = area(mesh);
    let mesh_abs_volume = abs_volume(mesh);
    eprintln!(
        "{label}: V={} F={} quads={} non-quads={} area={:.9} abs-volume={:.9}",
        vertex_count(mesh),
        face_count(mesh),
        quad_face_count(mesh),
        non_quad_face_count(mesh),
        mesh_area,
        mesh_abs_volume,
    );
    eprintln!(
        "{label}: boundary edges={} non-manifold edges={} invalid-lt3={} invalid-repeat={} invalid-index={} invalid-quad={} isolated vertices={}",
        boundary_edge_count(mesh),
        nonmanifold_edge_count(mesh),
        fewer_than_three_face_count(mesh),
        repeated_vertex_face_count(mesh),
        invalid_vertex_index_face_count(mesh),
        invalid_quad_face_count(mesh),
        isolated_vertex_count(mesh),
    );
    if let Some((input_area, input_abs_volume)) = baseline {
        if let Some(value) = ratio(mesh_area, input_area) {
            eprintln!("{label}: area ratio vs input = {:.3}", value);
        }
        if let Some(value) = ratio(mesh_abs_volume, input_abs_volume) {
            eprintln!("{label}: volume ratio vs input = {:.3}", value);
        }
    }
}

fn usage() -> &'static str {
    "usage: remesh <input.obj> -o <output.obj> (--edge-length L | --target-vertices N | --target-faces N) [--hierarchy-orientation-iters N] [--hierarchy-position-iters N] [--orientation-iters N] [--position-iters N] [--frozen-orientation-iters N] [--frozen-position-iters N] [--seed N] [--intrinsic | --extrinsic]"
}
