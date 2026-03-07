use remesh::meshio::{load_obj, triangulate_faces, write_obj};
use remesh::metrics::{analyze, ratio};
use remesh::preprocess::{as_trimesh, compute_dual_vertex_areas, compute_mesh_stats, dedge, generate_smooth_normals, generate_uniform_adjacency, subdivide_to_max_edge};
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
    let tri_mesh = as_trimesh(input.vertices, triangulate_faces(&input.faces));
    let input_stats = compute_mesh_stats(&tri_mesh);
    let scale = target_scale(&args, input_stats.surface_area);
    let subdiv_limit = (scale * 0.5).min(input_stats.average_edge_length * 2.0);
    let output_mesh = if input_stats.maximum_edge_length * 2.0 > scale
        || input_stats.maximum_edge_length > input_stats.average_edge_length * 2.0
    {
        subdivide_to_max_edge(&tri_mesh, subdiv_limit)
    } else {
        tri_mesh
    };

    let output_faces: Vec<Vec<usize>> = output_mesh.faces.iter().map(|face| face.to_vec()).collect();
    write_obj(&args.output, &output_mesh.vertices, &output_faces)?;

    let output_report = analyze(&load_obj(&args.output)?);
    let output_stats = compute_mesh_stats(&output_mesh);
    let dedge = dedge(&output_mesh);
    let dual_areas = compute_dual_vertex_areas(&output_mesh, &dedge);
    let normals = generate_smooth_normals(&output_mesh);
    let adjacency = generate_uniform_adjacency(&output_mesh, &dedge);

    println!("input_area: {:.9}", input_report.area);
    println!("output_area: {:.9}", output_report.area);
    println!("area_ratio: {:.9}", ratio(output_report.area, input_report.area).unwrap_or(0.0));
    println!("input_volume: {:.9}", input_report.abs_volume);
    println!("output_volume: {:.9}", output_report.abs_volume);
    println!(
        "volume_ratio: {:.9}",
        ratio(output_report.abs_volume, input_report.abs_volume).unwrap_or(0.0)
    );
    println!("input_vertices: {}", input_report.vertex_count);
    println!("output_vertices: {}", output_report.vertex_count);
    println!("input_faces: {}", input_report.face_count);
    println!("output_faces: {}", output_report.face_count);
    println!("target_edge_length: {:.9}", scale);
    println!("avg_edge_length: {:.9}", output_stats.average_edge_length);
    println!("max_edge_length: {:.9}", output_stats.maximum_edge_length);
    println!("boundary_vertices: {}", dedge.boundary.iter().filter(|&&v| v).count());
    println!("nonmanifold_vertices: {}", dedge.nonmanifold.iter().filter(|&&v| v).count());
    println!(
        "avg_dual_area: {:.9}",
        if dual_areas.is_empty() {
            0.0
        } else {
            dual_areas.iter().sum::<f64>() / dual_areas.len() as f64
        }
    );
    println!(
        "avg_neighbors: {:.9}",
        if adjacency.is_empty() {
            0.0
        } else {
            adjacency.iter().map(|row| row.len() as f64).sum::<f64>() / adjacency.len() as f64
        }
    );
    println!(
        "avg_normal_length: {:.9}",
        if normals.is_empty() {
            0.0
        } else {
            normals.iter().map(|normal| normal.norm()).sum::<f64>() / normals.len() as f64
        }
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

    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-o" | "--output" => output = Some(PathBuf::from(next_value(&mut args, &arg)?)),
            "--edge-length" => edge_length = Some(next_value(&mut args, &arg)?.parse()?),
            "--target-vertices" => target_vertices = Some(next_value(&mut args, &arg)?.parse()?),
            "--target-faces" => target_faces = Some(next_value(&mut args, &arg)?.parse()?),
            "-h" | "--help" => return Err(usage().into()),
            _ if arg.starts_with('-') => {
                return Err(format!("unknown flag: {arg}\n\n{}", usage()).into())
            }
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
        }),
        _ => Err(usage().into()),
    }
}

fn next_value<I>(args: &mut I, flag: &str) -> Result<String, Box<dyn Error>>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| format!("missing value for {flag}").into())
}

fn target_scale(args: &Args, surface_area: f64) -> f64 {
    if let Some(length) = args.edge_length {
        return length;
    }
    if let Some(face_count) = args.target_faces {
        return (surface_area / face_count as f64).sqrt();
    }
    if let Some(vertex_count) = args.target_vertices {
        return (surface_area / vertex_count as f64).sqrt();
    }
    unreachable!()
}

fn usage() -> &'static str {
    "usage: preprocess <input.obj> -o <output.obj> (--edge-length L | --target-vertices N | --target-faces N)"
}
