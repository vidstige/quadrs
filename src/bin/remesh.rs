use remesh::boundary::{build_boundary_constraints, build_boundary_hierarchy, build_boundary_segments};
use remesh::hierarchy::{build_hierarchy, prolong_origins, prolong_orientations, HierarchyLevel};
use remesh::meshio::{load_obj, triangulate_faces, write_obj, ObjMesh};
use remesh::metrics::{analyze, ratio, MeshReport};
use remesh::graph::extract_graph;
use remesh::postprocess::{compact_quads, fill_small_boundary_loops, repair_quads, smooth_and_reproject_invalid_quads};
use remesh::field::{
    freeze_orientation_ivars, freeze_position_ivars, initialize_state, optimize_orientations,
    optimize_orientations_frozen, optimize_positions, optimize_positions_frozen, BoundaryConstraint, NativeState,
};
use remesh::preprocess::{
    compute_dual_vertex_areas, compute_mesh_stats, generate_smooth_normals, generate_uniform_adjacency, preprocess_mesh,
};
use remesh::topology::{build_directed_edges, TriMesh};
use remesh::meshio::Vec3;
use std::env;
use std::error::Error;
use std::path::PathBuf;

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
