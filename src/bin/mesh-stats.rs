use remesh::meshio::{load_obj, triangulate_faces};
use remesh::metrics::{
    abs_volume, area, boundary_edge_count, face_count, fewer_than_three_face_count,
    invalid_quad_face_count, invalid_vertex_index_face_count, isolated_vertex_count,
    non_quad_face_count, nonmanifold_edge_count, quad_face_count, repeated_vertex_face_count,
    vertex_count,
};
use remesh::preprocess::{compute_dual_vertex_areas, compute_mesh_stats, generate_smooth_normals, generate_uniform_adjacency};
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
    let tri_mesh = TriMesh {
        vertices: mesh.vertices.clone(),
        faces: triangulate_faces(&mesh.faces),
    };
    let stats = compute_mesh_stats(&tri_mesh);
    let dedge = build_directed_edges(&tri_mesh);
    let dual_areas = compute_dual_vertex_areas(&tri_mesh, &dedge);
    let normals = generate_smooth_normals(&tri_mesh);
    let adjacency = generate_uniform_adjacency(&tri_mesh, &dedge);
    let avg_dual_area = average_or_zero(&dual_areas, |area| *area);
    let avg_neighbors = average_or_zero(&adjacency, |row| row.len() as f64);
    let avg_normal_len = average_or_zero(&normals, |normal| normal.norm());

    println!("file: {}", input.display());
    println!("vertices: {}", vertex_count(&mesh));
    println!("faces: {}", face_count(&mesh));
    println!("quads: {}", quad_face_count(&mesh));
    println!("non_quads: {}", non_quad_face_count(&mesh));
    println!("area: {:.9}", area(&mesh));
    println!("abs_volume: {:.9}", abs_volume(&mesh));
    println!("boundary_edges: {}", boundary_edge_count(&mesh));
    println!("nonmanifold_edges: {}", nonmanifold_edge_count(&mesh));
    println!("invalid_lt3: {}", fewer_than_three_face_count(&mesh));
    println!("invalid_repeat: {}", repeated_vertex_face_count(&mesh));
    println!("invalid_index: {}", invalid_vertex_index_face_count(&mesh));
    println!("invalid_quad: {}", invalid_quad_face_count(&mesh));
    println!("isolated_vertices: {}", isolated_vertex_count(&mesh));
    println!("avg_edge_length: {:.9}", stats.average_edge_length);
    println!("max_edge_length: {:.9}", stats.maximum_edge_length);
    println!("surface_area_tri: {:.9}", stats.surface_area);
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

fn average_or_zero<T>(items: &[T], value: impl Fn(&T) -> f64) -> f64 {
    if items.is_empty() {
        0.0
    } else {
        items.iter().map(value).sum::<f64>() / items.len() as f64
    }
}
