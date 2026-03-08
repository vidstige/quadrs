use remesh::meshio::{load_obj, triangulate_faces};
use remesh::metrics::analyze;
use remesh::preprocess::{average_valence, compute_dual_vertex_areas, compute_mesh_stats, generate_smooth_normals, generate_uniform_adjacency};
use remesh::topology::{build_directed_edges, TriMesh};
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
    let input = env::args().nth(1).ok_or_else(|| usage().to_string())?;
    let input = PathBuf::from(input);
    let mesh = load_obj(&input)?;
    let report = analyze(&mesh);
    let tri_mesh = TriMesh {
        vertices: mesh.vertices,
        faces: triangulate_faces(&mesh.faces),
    };
    let stats = compute_mesh_stats(&tri_mesh);
    let dedge = build_directed_edges(&tri_mesh);
    let dual_areas = compute_dual_vertex_areas(&tri_mesh, &dedge);
    let normals = generate_smooth_normals(&tri_mesh);
    let adjacency = generate_uniform_adjacency(&tri_mesh, &dedge);
    let avg_dual_area = if dual_areas.is_empty() {
        0.0
    } else {
        dual_areas.iter().sum::<f64>() / dual_areas.len() as f64
    };
    let avg_neighbors = if adjacency.is_empty() {
        0.0
    } else {
        adjacency.iter().map(|row| row.len() as f64).sum::<f64>() / adjacency.len() as f64
    };
    let avg_normal_len = if normals.is_empty() {
        0.0
    } else {
        normals.iter().map(|n| n.norm()).sum::<f64>() / normals.len() as f64
    };

    println!("file: {}", input.display());
    println!("vertices: {}", report.vertex_count);
    println!("faces: {}", report.face_count);
    println!("quads: {}", report.quad_faces);
    println!("non_quads: {}", report.non_quad_faces);
    println!("area: {:.9}", report.area);
    println!("abs_volume: {:.9}", report.abs_volume);
    println!("boundary_edges: {}", report.boundary_edges);
    println!("boundary_loops: {}", report.boundary_loops);
    println!("nonmanifold_edges: {}", report.nonmanifold_edges);
    println!("invalid_lt3: {}", report.fewer_than_three_faces);
    println!("invalid_repeat: {}", report.repeated_vertex_faces);
    println!("invalid_index: {}", report.invalid_vertex_index_faces);
    println!("invalid_quad: {}", report.invalid_quad_faces);
    println!("duplicate_faces: {}", report.duplicate_faces);
    println!("isolated_vertices: {}", report.isolated_vertices);
    println!("components: {}", report.connected_components);
    println!("avg_valence: {:.6}", average_valence(&tri_mesh));
    println!("avg_edge_length: {:.9}", stats.average_edge_length);
    println!("max_edge_length: {:.9}", stats.maximum_edge_length);
    println!("surface_area_tri: {:.9}", stats.surface_area);
    println!(
        "weighted_center: {:.9} {:.9} {:.9}",
        stats.weighted_center.x, stats.weighted_center.y, stats.weighted_center.z
    );
    println!("boundary_vertices: {}", dedge.boundary.iter().filter(|&&v| v).count());
    println!("nonmanifold_vertices: {}", dedge.nonmanifold.iter().filter(|&&v| v).count());
    println!("avg_dual_area: {:.9}", avg_dual_area);
    println!("avg_neighbors: {:.6}", avg_neighbors);
    println!("avg_normal_length: {:.6}", avg_normal_len);
    Ok(())
}

fn usage() -> &'static str {
    "usage: mesh-stats <input.obj>"
}
