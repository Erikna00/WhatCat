

###
#code belonging to ProteinTunnelAnalyzer
###

def bulk_vertices_check(self):
        "checks if a point is outside the convex hull"

        # Calculate convex hull for the protein from the atom positions
        self.hull = ConvexHull(self.positions)
        self.bulk_vertices_index = []

        # Create a Delaunay triangulation of the hull points for faster point-in-hull check
        hull_triangulation = Delaunay(self.hull.points)

        #classify bulk and nonbulk
        for i, vertex in enumerate(self.graph_node_positions):
            if hull_triangulation.find_simplex(vertex) < 0:
                self.bulk_vertices_index.append(i)

        return self.bulk_vertices_index   

def find_tunnels(self, start_point=None, max_tunnels=10, min_bottleneck=1.0):
        """
        Find tunnels from a starting point to bulk solvent.
        
        Parameters:
        -----------
        start_point : array-like or None
            Starting point coordinates. If None, use previously set start_point.
        max_tunnels : int
            Maximum number of tunnels to find
        min_bottleneck : float
            Minimum bottleneck radius (Å) for a valid tunnel
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing tunnel information
        """
        # Determine the starting point
        if start_point is not None:
            self.start_point = np.array(start_point)
        elif self.start_point is None:
            raise ValueError("No starting point provided or previously set")
            
        # Find the closest valid vertex to the starting point
        self.start_vertex = self._find_closest_vertex_to_point(self.start_point)
        print(f"start vertex: {self.start_vertex}")
        
        # Prepare tunnel data storage
        self.tunnels = []
        tunnel_data = []
        
        bulk_vertices_to_search = self.bulk_vertices_index
        
        # Calculate distances from start vertex to all bulk vertices
        bulk_coords = self.graph_node_positions[bulk_vertices_to_search]
        start_coord = self.graph_node_positions[self.start_vertex]
        distances = np.linalg.norm(bulk_coords - start_coord, axis=1)
        
        # Sort bulk vertices by distance to starting point (closer first)
        sorted_indices = np.argsort(distances)
        sorted_bulk_vertices = [bulk_vertices_to_search[i] for i in sorted_indices]
        
        # Find tunnels to each bulk solvent vertex
        for i, bulk_idx in enumerate(sorted_bulk_vertices):
            if len(self.tunnels) >= max_tunnels:
                break
                
            try:
                # Find shortest path from start to this bulk vertex
                path = nx.shortest_path(self.graph, self.start_vertex, bulk_idx, weight='weight')
                
                # Extract path details
                path_vertices = [self.graph_node_positions[idx] for idx in path]
                path_rmax = [self.graph.nodes[idx]['rmax'] for idx in path]
                
                # Only keep tunnels with sufficient bottleneck radius
                min_rmax = min(path_rmax)
                if min_rmax >= min_bottleneck:
                    # Calculate length efficiently
                    path_length = sum(self.graph[path[i]][path[i+1]]['distance'] for i in range(len(path)-1))
                    
                    self.tunnels.append({
                        'path_indices': path,
                        'path_vertices': path_vertices,
                        'path_rmax': path_rmax,
                        'length': path_length,
                        'bottleneck': min_rmax,
                        'bottleneck_idx': path_rmax.index(min_rmax)
                    })
                    
                    # Add data for DataFrame
                    tunnel_data.append({
                        'tunnel_id': len(self.tunnels),
                        'length': path_length,
                        'bottleneck': min_rmax,
                        'bottleneck_position': path_vertices[path_rmax.index(min_rmax)],
                        'start': path_vertices[0],
                        'end': path_vertices[-1],
                        'num_points': len(path),
                        'start_point': self.start_point
                    })
            except nx.NetworkXNoPath:
                continue
        
        # Create DataFrame with tunnel data
        self.tunnel_dataframe = pd.DataFrame(tunnel_data) if tunnel_data else pd.DataFrame()
        return self.tunnel_dataframe


# Define this helper function at the class or module level (outside any method)
@staticmethod
def _process_vertex_chunk(chunk_data):
    """
    Process a chunk of vertices to find valid and bulk solvent vertices.
    """
    start_idx, end_idx, vertices, positions, atom_radii, probe_radius, hull_points, hull_simplices = chunk_data
    
    import numpy as np
    from scipy.spatial import Delaunay, cKDTree
    
    # Results for this chunk
    local_valid = []
    local_bulk = []
    
    # Pre-compute squared radii for faster comparison
    radii_squared = atom_radii ** 2
    
    # Use KDTree for more efficient near-neighbor queries
    atom_tree = cKDTree(positions)
    
    # Create a Delaunay triangulation of the hull points for faster point-in-hull check
    hull_triangulation = Delaunay(hull_points)
    
    # Process vertices in this chunk
    for i in range(start_idx, end_idx):
        vertex = vertices[i]
        
        # Find atoms that could possibly overlap with this vertex
        max_radius = np.max(atom_radii)
        indices = atom_tree.query_ball_point(vertex, max_radius + 1.0)
        
        # Check if vertex is inside any atom's vdW radius
        is_valid = True
        for j in indices:
            dist_squared = np.sum((vertex - positions[j])**2)
            if dist_squared < radii_squared[j]:
                is_valid = False
                break
        
        if is_valid:
            local_valid.append(i)
            
            # Check if vertex is outside the convex hull (potentially bulk)
            # Use Delaunay triangulation for faster point-in-hull check
            if not hull_triangulation.find_simplex(vertex) >= 0:
                local_bulk.append(i)
    
    return local_valid, local_bulk


def filter_voronoi_vertices(self, n_processes=None):
    """
    Filter out Voronoi vertices that are inside atom vdW radii and
    identify vertices in bulk solvent. Reindex vertices after filtering.
    
    Parameters:
    -----------
    n_processes : int or None
        Number of processes to use for multiprocessing.
        If None, will use number of available CPU cores.
        
    Returns:
    --------
    tuple
        (valid_vertices, bulk_vertices) lists of vertex indices
    """
    # Determine number of processes to use
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)  # Leave one core free
    
    print(f"Using {n_processes} processes for vertex filtering")
    
    # Calculate convex hull for the protein
    hull = ConvexHull(self.positions)
    
    # Get all vertices for checking
    vertices = self.enhanced_voronoi.vertices
    n_vertices = len(vertices)
    
    # Prepare chunks for multiprocessing
    chunk_size = max(1, n_vertices // n_processes)
    chunks = []
    
    for i in range(0, n_vertices, chunk_size):
        end_idx = min(i + chunk_size, n_vertices)
        start_idx = i
        
        # Package all data needed for processing this chunk
        chunk_data = (
            start_idx, end_idx, 
            vertices, 
            self.positions, 
            self.atom_radii, 
            self.probe_radius,
            self.positions[hull.vertices],  # Use hull vertices directly
            hull.simplices  # Use hull simplices directly
        )
        chunks.append(chunk_data)
    
    # Execute in parallel
    valid_vertices = []
    bulk_vertices = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_processes) as executor:
        results = list(executor.map(self._process_vertex_chunk, chunks))
        
    # Combine results from all processes
    for local_valid, local_bulk in results:
        valid_vertices.extend(local_valid)
        bulk_vertices.extend(local_bulk)

    # **Reindexing step**
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_vertices)}

    # Update valid and bulk vertex lists
    self.valid_vertices = np.array([old_to_new[v] for v in valid_vertices], dtype=int)
    self.bulk_vertices = np.array([old_to_new[v] for v in bulk_vertices if v in old_to_new], dtype=int)

    # Store only the valid Voronoi vertices
    self.filtered_voronoi_vertices = self.enhanced_voronoi.vertices[valid_vertices]

    print(f"Found {len(self.valid_vertices)} valid vertices and {len(self.bulk_vertices)} bulk vertices")
    
    return self.valid_vertices, self.bulk_vertices    

def enhance_voronoi_with_surface_points(self):
    """
    Enhance the Voronoi diagram by adding points on vdW surface 
    in the direction of neighboring vertices.
    """
    # Initialize storage for enhanced diagram data
    enhanced_positions = self.positions.copy()
    surface_points = []
    surface_atom_indices = []
    
    # Use optimized KDTree for vertex neighbor searches
    vertex_tree = cKDTree(self.voronoi.vertices)
    
    # Faster approach: For each atom, query nearby vertices
    for i, (pos, radius) in enumerate(zip(self.positions, self.atom_radii)):
        # Find vertices within 3x the atom radius
        nearby_indices = vertex_tree.query_ball_point(pos, radius * 3)
        
        if not nearby_indices:
            continue
            
        nearby_vertices = self.voronoi.vertices[nearby_indices]
        directions = nearby_vertices - pos
        direction_norms = np.linalg.norm(directions, axis=1)
        
        # Skip if all norms are zero (unlikely but possible)
        valid_indices = direction_norms > 0
        if not np.any(valid_indices):
            continue
            
        # Normalize directions
        valid_directions = directions[valid_indices]
        valid_norms = direction_norms[valid_indices].reshape(-1, 1)
        unit_directions = valid_directions / valid_norms
        
        # Create points on the vdW surface
        new_surface_points = pos + unit_directions * radius
        
        #add points to graph
        for surface_point in new_surface_points:
            surface_points.append(surface_point)
            surface_atom_indices.append(i)
    
    # Add surface points to the positions
    if surface_points:
        enhanced_positions = np.vstack([enhanced_positions, np.array(surface_points)])
        
        # Rebuild Voronoi diagram with enhanced positions
        self.enhanced_voronoi = Voronoi(enhanced_positions)
        
        # Store mapping from surface points to their parent atoms
        self.surface_to_atom = {}
        for i, idx in enumerate(surface_atom_indices):
            self.surface_to_atom[i + len(self.positions)] = idx
            
        return self.enhanced_voronoi
    else:
        # If no valid surface points were found, use the original diagram
        self.enhanced_voronoi = self.voronoi
        return self.enhanced_voronoi