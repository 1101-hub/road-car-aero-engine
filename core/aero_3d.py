"""
NetFold 3D Aerodynamic Solver
=============================
A constant-strength 3D Source Panel Method that computes the inviscid pressure 
distribution over a reconstructed NetFold mesh.
"""

import numpy as np

def calculate_mesh_properties(placed: dict, n_tris: int):
    """
    Computes centroids, normals, and areas for all triangles.
    """
    centroids = np.zeros((n_tris, 3))
    normals = np.zeros((n_tris, 3))
    areas = np.zeros(n_tris)
    
    for i in range(n_tris):
        pts = placed[i]
        
        # Centroid
        centroids[i] = pts.mean(axis=0)
        
        # Cross product for area and normal
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        cross = np.cross(v1, v2)
        norm = np.linalg.norm(cross)
        
        if norm > 1e-12:
            normals[i] = cross / norm
            areas[i] = 0.5 * norm
        else:
            normals[i] = np.array([0.0, 0.0, 1.0])
            areas[i] = 0.0
            
    return centroids, normals, areas

def solve_3d_panel_method(placed: dict, v_inf: np.ndarray = np.array([-1.0, 0.0, 0.0])):
    """
    Solves the 3D Source Panel Method for the given mesh and freestream velocity.
    
    Returns:
        cp_array: array of pressure coefficients (Cp) for each triangle.
        velocity_field: (N, 3) array of total velocity vectors at each centroid.
    """
    n_tris = len(placed)
    centroids, normals, areas = calculate_mesh_properties(placed, n_tris)
    
    # 1. Build Influence Matrix A and RHS vector b
    A = np.zeros((n_tris, n_tris))
    b = np.zeros(n_tris)
    
    for i in range(n_tris):
        # Boundary condition: V_inf dot n_i + V_induced dot n_i = 0
        b[i] = -np.dot(v_inf, normals[i])
        
        for j in range(n_tris):
            if i == j:
                # Self-influence normal velocity is exactly 0.5
                A[i, j] = 0.5
            else:
                # Point-source approximation for far-field influence
                r_vec = centroids[i] - centroids[j]
                r_mag = np.linalg.norm(r_vec)
                if r_mag > 1e-12:
                    # Velocity induced by source j at point i
                    v_ind = (areas[j] / (4 * np.pi * r_mag**3)) * r_vec
                    # Normal component
                    A[i, j] = np.dot(v_ind, normals[i])
                    
    # 2. Solve for source strengths (sigma)
    sigma = np.linalg.solve(A, b)
    
    # 3. Calculate total velocity and Cp at each centroid
    velocity_field = np.zeros((n_tris, 3))
    cp_array = np.zeros(n_tris)
    
    v_inf_mag_sq = np.dot(v_inf, v_inf)
    
    for i in range(n_tris):
        v_total = v_inf.copy()
        for j in range(n_tris):
            if i == j:
                # Self-induced velocity is normal to the panel
                v_total += 0.5 * sigma[i] * normals[i]
            else:
                r_vec = centroids[i] - centroids[j]
                r_mag = np.linalg.norm(r_vec)
                if r_mag > 1e-12:
                    v_ind = (areas[j] / (4 * np.pi * r_mag**3)) * r_vec
                    v_total += sigma[j] * v_ind
                    
        velocity_field[i] = v_total
        
        # Tangential velocity (remove any residual normal component due to approx)
        v_norm_comp = np.dot(v_total, normals[i])
        v_tangent = v_total - v_norm_comp * normals[i]
        v_tangent_mag_sq = np.dot(v_tangent, v_tangent)
        
        # Pressure coefficient: Cp = 1 - (V/V_inf)^2
        cp_array[i] = 1.0 - (v_tangent_mag_sq / v_inf_mag_sq)
        
    return cp_array, velocity_field
