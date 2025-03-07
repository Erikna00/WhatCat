#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Regular_triangulation_vertex_base_3.h>
#include <CGAL/Regular_triangulation_cell_base_3.h>
#include <CGAL/Triangulation_data_structure_3.h>
#include <CGAL/Regular_triangulation_3.h>
#include <vector>
#include <map>
#include <set>
#include <cmath>
#include <cstring>

namespace py = pybind11;
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Regular_triangulation_vertex_base_3<K> Vb;
typedef CGAL::Regular_triangulation_cell_base_3<K> Cb;
typedef CGAL::Triangulation_data_structure_3<Vb, Cb> Tds;
typedef CGAL::Regular_triangulation_3<K, Tds> Regular_triangulation;
typedef Regular_triangulation::Weighted_point Weighted_point;
typedef Regular_triangulation::Cell_handle Cell_handle;
typedef K::Point_3 Point;

// Helper: compare points with a tolerance.
bool approx_equal(const Point& p, const Point& q, double tol = 1e-9) {
    return std::abs(p.x() - q.x()) < tol &&
           std::abs(p.y() - q.y()) < tol &&
           std::abs(p.z() - q.z()) < tol;
}

// Parse input weighted points (note: weights are squared).
std::vector<Weighted_point> parse_input(py::array_t<double> points, py::array_t<double> weights) {
    auto buf_points = points.request(), buf_weights = weights.request();
    if (buf_points.ndim != 2 || buf_points.shape[1] != 3 || buf_weights.ndim != 1)
        throw std::runtime_error("Invalid input shape");
    double* ptr_points = static_cast<double*>(buf_points.ptr);
    double* ptr_weights = static_cast<double*>(buf_weights.ptr);
    size_t num_points = buf_points.shape[0];
    std::vector<Weighted_point> weighted_points;
    for (size_t i = 0; i < num_points; i++) {
        Point p(ptr_points[3*i], ptr_points[3*i+1], ptr_points[3*i+2]);
        weighted_points.emplace_back(p, ptr_weights[i]*ptr_weights[i]);
    }
    return weighted_points;
}

py::tuple compute_voronoi(py::array_t<double> points, py::array_t<double> weights) {
    auto weighted_points = parse_input(points, weights);
    Regular_triangulation rt;
    rt.insert(weighted_points.begin(), weighted_points.end());
    
    // Build unique dual vertices:
    std::map<Cell_handle, size_t> cell_to_index;
    std::vector<Point> unique_dual_pts;
    for (auto cit = rt.finite_cells_begin(); cit != rt.finite_cells_end(); ++cit) {
        Point d = rt.dual(cit);
        bool found = false;
        size_t idx = 0;
        for (size_t i = 0; i < unique_dual_pts.size(); i++) {
            if (approx_equal(d, unique_dual_pts[i])) {
                found = true;
                idx = i;
                break;
            }
        }
        if (!found) {
            unique_dual_pts.push_back(d);
            idx = unique_dual_pts.size() - 1;
        }
        cell_to_index[cit] = idx;
    }
    
    // Build edges by iterating over finite facets.
    std::set<std::pair<size_t, size_t>> edge_set;
    for (auto fit = rt.finite_facets_begin(); fit != rt.finite_facets_end(); ++fit) {
        Cell_handle cell = fit->first;
        int i = fit->second;
        Cell_handle nbr = cell->neighbor(i);
        if (rt.is_infinite(nbr))
            continue;
        size_t idx1 = cell_to_index[cell];
        size_t idx2 = cell_to_index[nbr];
        // Skip if the dual vertices are (approximately) the same.
        if (idx1 == idx2)
            continue;
        if (idx1 > idx2)
            std::swap(idx1, idx2);
        edge_set.insert({idx1, idx2});
    }
    
    std::vector<std::pair<size_t, size_t>> edges(edge_set.begin(), edge_set.end());
    
    // Create NumPy arrays for vertices and edges.
    ssize_t num_dual = unique_dual_pts.size();
    std::vector<ssize_t> v_shape = { num_dual, 3 };
    std::vector<ssize_t> v_strides = { 3 * sizeof(double), sizeof(double) };
    py::array_t<double> vertices(v_shape, v_strides);
    auto buf_vertices = vertices.request();
    double* ptr_vertices = static_cast<double*>(buf_vertices.ptr);
    for (size_t i = 0; i < unique_dual_pts.size(); i++) {
        ptr_vertices[3*i + 0] = unique_dual_pts[i].x();
        ptr_vertices[3*i + 1] = unique_dual_pts[i].y();
        ptr_vertices[3*i + 2] = unique_dual_pts[i].z();
    }
    
    ssize_t num_edges = edges.size();
    std::vector<ssize_t> e_shape = { static_cast<ssize_t>(edges.size()), 2 };
    std::vector<ssize_t> e_strides = { 2 * sizeof(size_t), sizeof(size_t) };
    py::array_t<size_t> edge_array(e_shape, e_strides);
    auto buf_edges = edge_array.request();
    size_t* ptr_edges = static_cast<size_t*>(buf_edges.ptr);
    for (size_t i = 0; i < edges.size(); i++) {
        ptr_edges[2*i + 0] = edges[i].first;
        ptr_edges[2*i + 1] = edges[i].second;
    }
    
    return py::make_tuple(vertices, edge_array);
}

PYBIND11_MODULE(voronoi_cgal, m) {
    m.def("compute_voronoi", &compute_voronoi, "Compute weighted Voronoi (power) diagram from points and weights");
}
