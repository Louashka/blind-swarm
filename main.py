import re
import mujoco
import mujoco.viewer
import numpy as np
import time
import xml.etree.ElementTree as ET
from shapely.geometry import box
from shapely.ops import unary_union
from swarm import AntSwarm

# --- 1. Helper: Parse XML for Contour and Body Z-Position ---
def get_contour_and_body_z(xml_path, body_name="load"):
    """
    Parses a MuJoCo XML to find a specific body.
    1. Extracts the 'pos' Z-value from the body tag itself.
    2. Merges all box geoms within that body to create a 2D contour.
    
    Returns:
        corners (np.array): Ordered exterior coordinates (x, y).
        body_z (float): The Z coordinate from the body's 'pos' attribute.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Find the body
    target_body = None
    for body in root.iter('body'):
        if body.get('name') == body_name:
            target_body = body
            break
            
    if target_body is None:
        raise ValueError(f"Body '{body_name}' not found in XML.")

    # --- Extract Z from the Body tag ---
    body_pos_str = target_body.get('pos', "0 0 0")
    body_pos_vals = [float(x) for x in body_pos_str.split()]
    body_z = body_pos_vals[2]

    # --- Extract Contour from Geoms ---
    polygons = []
    
    for geom in target_body.findall('geom'):
        g_type = geom.get('type')
        
        if g_type == 'box':
            # Get size (half-extents): "x y z"
            size_str = geom.get('size')
            if not size_str: continue
            size = np.array([float(x) for x in size_str.split()])
            
            # Get pos (relative to body center): "x y z"
            pos_str = geom.get('pos', "0 0 0")
            pos = np.array([float(x) for x in pos_str.split()])
            
            # Create Shapely Box for Contour (using x and y)
            minx = pos[0] - size[0]
            miny = pos[1] - size[1]
            maxx = pos[0] + size[0]
            maxy = pos[1] + size[1]
            
            poly = box(minx, miny, maxx, maxy)
            polygons.append(poly)

    if not polygons:
        raise ValueError(f"No 'box' geoms found in body '{body_name}'")

    # Merge Polygons
    merged_shape = unary_union(polygons)
    x, y = merged_shape.exterior.xy
    corners = np.column_stack((x, y))[:-1]
    
    return corners, body_z

def get_puzzle_geometry(xml_path, load_z_height):
    """
    Parses the XML to find the slit positions and chamber boundaries.
    Returns a dictionary with the critical coordinates.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    puzzle_body = root.find(f".//body[@name='puzzle_walls']")
    if puzzle_body is None:
        raise ValueError("Body 'puzzle_walls' not found in XML.")
        
    puzzle_pos = np.array([float(v) for v in puzzle_body.get('pos', '0 0 0').split()])

    # Find slit walls by name
    slit1_wall = None
    slit2_wall = None
    
    # Iterate through all geoms within the 'puzzle_walls' body
    for geom in puzzle_body.findall('geom'):
        geom_name = geom.get('name', '') # Use default to avoid error if name is missing
        
        # Find the first geom related to slit1
        if 'slit1' in geom_name and slit1_wall is None:
            slit1_wall = geom
        
        # Find the first geom related to slit2
        if 'slit2' in geom_name and slit2_wall is None:
            slit2_wall = geom
            
        # Optimization: stop if both have been found
        if slit1_wall is not None and slit2_wall is not None:
            break
    
    if slit1_wall is None or slit2_wall is None:
        raise ValueError("Could not find 'slit1' or 'slit2' geoms in XML.")

    # Slit position is defined by the wall's X-coordinate
    slit1_pos_local = np.array([float(v) for v in slit1_wall.get('pos').split()])
    slit2_pos_local = np.array([float(v) for v in slit2_wall.get('pos').split()])

    # Global X-coordinates define chamber boundaries
    chamber1_boundary_x = puzzle_pos[0] + slit1_pos_local[0]
    chamber2_boundary_x = puzzle_pos[0] + slit2_pos_local[0]

    # Slit midpoints are the targets for the ants
    # We use the load's Z-height for the target Z for simplicity
    slit1_midpoint = np.array([chamber1_boundary_x, 0, load_z_height])
    slit2_midpoint = np.array([chamber2_boundary_x, 0, load_z_height])
    
    geometry = {
        'chamber1_x_boundary': chamber1_boundary_x,
        'chamber2_x_boundary': chamber2_boundary_x,
        'slit1_midpoint': slit1_midpoint,
        'slit2_midpoint': slit2_midpoint,
    }
    print("Parsed Puzzle Geometry:")
    for key, val in geometry.items():
        print(f"  - {key}: {val}")
        
    return geometry

# --- 2. Helper: Generate Attachment Points ---
def generate_perimeter_points(corners, num_ants, ant_radius, body_z):
    """
    Generates positions around the contour and their corresponding local normal vectors.
    
    Returns:
        tuple: (np.array of points, np.array of normals)
    """
    points = []
    normals = [] # We will store the normals here
    
    # Calculate local Z position relative to the body center
    z_ground = -body_z + ant_radius 
    
    # Ensure the corners form a closed loop for segment calculation
    closed_corners = np.vstack([corners, corners[0]])
    
    # Calculate total perimeter length and individual segment lengths
    total_perimeter = 0
    segment_lengths = []
    for i in range(len(corners)):
        p1 = closed_corners[i]
        p2 = closed_corners[i+1]
        dist = np.linalg.norm(p2[:2] - p1[:2]) # Use 2D distance for perimeter
        segment_lengths.append(dist)
        total_perimeter += dist
        
    # Avoid division by zero if there's no perimeter
    if total_perimeter < 1e-9:
        return np.array([]), np.array([])
        
    step_dist = total_perimeter / num_ants
    
    for i in range(num_ants):
        target_dist = i * step_dist
        
        accumulated_len = 0
        for seg_idx, seg_len in enumerate(segment_lengths):
            # Check if the target distance falls within the current segment
            if accumulated_len + seg_len >= target_dist:
                p1 = closed_corners[seg_idx]
                p2 = closed_corners[seg_idx+1]
                
                # This is the direction vector of the edge
                edge_vec = p2 - p1
                
                # --- THIS IS THE KEY PART ---
                # Calculate the 2D outward-facing normal vector from the 2D edge vector.
                # Assumes corners are defined in Counter-Clockwise (CCW) order.
                normal_2d = np.array([-edge_vec[1], edge_vec[0]])
                norm_len = np.linalg.norm(normal_2d)
                if norm_len > 1e-9:
                    normal_2d = normal_2d / norm_len
                
                # Store the 3D version of the normal (Z is 0 in local space)
                normals.append([normal_2d[0], normal_2d[1], 0.0])
                # --- END OF KEY PART ---
                
                # Find the position on the contour line itself
                # Handle division by zero for zero-length segments
                ratio = 0.0
                if seg_len > 1e-9:
                    local_dist = target_dist - accumulated_len
                    ratio = local_dist / seg_len
                
                pos_on_line = p1 + edge_vec * ratio
                
                # Calculate the final ant position, offset by the radius along the normal
                final_pos = pos_on_line[:2] + (normal_2d * ant_radius)
                
                points.append([final_pos[0], final_pos[1], z_ground])
                break # Move to the next ant
            else:
                accumulated_len += seg_len

    return np.array(points), np.array(normals)

# --- 3. Helper: Inject Visualization Sites ---
def load_model_with_ants(xml_path, ant_positions, ant_radius):
    with open(xml_path, 'r') as f:
        xml_content = f.read()

    site_str = ""
    for i, pos in enumerate(ant_positions):
        site_str += f'<site name="ant_{i}" pos="{pos[0]} {pos[1]} {pos[2]}" size="{ant_radius}" rgba="0.5 0.5 0.5 0.5"/>\n'

    pattern = r'(<body[^>]*name="load"[^>]*>)'
    match = re.search(pattern, xml_content)
    
    if not match:
        raise ValueError("Could not find body 'load' in XML")
        
    insertion_point = match.end()
    new_xml = xml_content[:insertion_point] + "\n" + site_str + xml_content[insertion_point:]
    
    return mujoco.MjModel.from_xml_string(new_xml)

# --- 4. Main Simulation ---
def main():
    xml_path = "env.xml"
    
    ANT_COUNT = 50      
    ANT_RADIUS = 0.015   

    print(f"Extracting geometry from {xml_path}...")
    
    # 1. Parse XML
    try:
        raw_corners, body_z = get_contour_and_body_z(xml_path, body_name="load")
        print(f"Detected body 'load' at Z-height: {body_z}")
        puzzle_geometry = get_puzzle_geometry(xml_path, load_z_height=body_z)
    except Exception as e:
        print(f"Error parsing XML geometry: {e}")
        return

    # 2. Generate Points
    ant_local_pos, ant_normals_local = generate_perimeter_points(raw_corners, ANT_COUNT, ANT_RADIUS, body_z)
    print(f"Generated {len(ant_local_pos)} ants around the contour.")
    
    # 3. Load Model
    model = load_model_with_ants(xml_path, ant_local_pos, ANT_RADIUS)
    data = mujoco.MjData(model)
    
    # 4. Initialize Swarm
    # Note: We pass the local positions to the swarm so it knows where ants are relative to center
    swarm = AntSwarm(ANT_COUNT, ant_local_pos, puzzle_geometry)
    
    # 5. Simulation Loop Setup
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "load")
    
    # The target direction in global coordinates (e.g., along X axis)
    target_dir = np.array([1.0, 0.0, 0.0]) 
    
    # Visualization colors
    color_detached = np.array([0.5, 0.5, 0.5, 0.3])     # Grey/Transparent
    color_informed = np.array([0.8, 0.1, 0.1, 1.0])     # Red
    color_puller = np.array([0.1, 0.1, 0.8, 1.0])       # Blue
    color_lifter = np.array([0.1, 0.8, 0.1, 1.0])

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 1. Set the lookat point to the center of the scene
        viewer.cam.lookat[0] = 1.5  # x
        viewer.cam.lookat[1] = 0.0  # y
        viewer.cam.lookat[2] = 0.0  # z

        # 2. Set the distance (height) of the camera
        viewer.cam.distance = 4.0

        # 3. Set the azimuth (horizontal rotation)
        viewer.cam.azimuth = 90.0  # Rotate so +X is to the right

        # 4. Set the elevation (vertical angle) to look straight down
        viewer.cam.elevation = -90.0

        print("\nViewer launched. Pausing for 5 seconds to allow you to resize the window.")

        time.sleep(5) 

        print("Resuming simulation...\n")

        while viewer.is_running():
            step_start = time.time()
            
            # --- Get Physical State from MuJoCo ---
            load_pos = data.xpos[body_id]
            # Linear velocity of the load (global frame)
            lin_vel = data.cvel[body_id][3:6]
            # Angular velocity of the load (global frame) -> ADDED THIS
            ang_vel = data.cvel[body_id][0:3]
            # Rotation matrix of the load (global frame)
            rot_mat = data.xmat[body_id].reshape(3, 3)

            lever_arms_global = (rot_mat @ ant_local_pos.T).T 
            ant_global_positions = load_pos + lever_arms_global

            # --- Update Swarm Logic ---
            # The swarm needs to know how the load is moving and rotating to calculate
            # the effective force each ant feels and whether they should detach.
            forces_global = swarm.update_logic(
                dt=model.opt.timestep,
                load_velocity_global=lin_vel,
                load_angular_vel=ang_vel,
                load_rotation_matrix=rot_mat,
                ant_global_positions=ant_global_positions,
                ant_local_normals=ant_normals_local
            )

            # --- Apply Forces ---            
            # 1. Sum forces for net linear force
            net_force = np.sum(forces_global, axis=0)
            
            # 2. Calculate torques
            # lever_arms_global = (Rotation Matrix * Local Pos)
            lever_arms_global = (rot_mat @ ant_local_pos.T).T 
            torques = np.cross(lever_arms_global, forces_global)
            net_torque = np.sum(torques, axis=0)

            # 3. Apply to MuJoCo data
            data.xfrc_applied[body_id][:3] = net_force
            data.xfrc_applied[body_id][3:] = net_torque

            # --- Update Visuals based on State ---
            for i in range(ANT_COUNT):
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"ant_{i}")
                state = swarm.states[i]
                
                # Assuming standard paper states: 0=Detached, 1=Puller, 2=Lifter
                if state == 0:
                    model.site_rgba[site_id] = color_detached
                elif state == 1:
                    model.site_rgba[site_id] = color_informed
                elif state == 2:
                    model.site_rgba[site_id] = color_puller
                elif state == 3:
                    model.site_rgba[site_id] = color_lifter

            # =========================================================
            # VISUALIZATION LOGIC
            # =========================================================
            
            viewer.user_scn.ngeom = 0

            def add_arrow(pos, vector, color, scale=1.0, radius=0.02):
                norm = np.linalg.norm(vector)
                if norm < 1e-6: return
                if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom: return

                def get_z_alignment_matrix(target_vec):
                    z_new = target_vec / np.linalg.norm(target_vec)
                    if abs(z_new[2]) < 0.99:
                        ref = np.array([0.0, 0.0, 1.0])
                    else:
                        ref = np.array([0.0, 1.0, 0.0])
                    x_new = np.cross(ref, z_new)
                    x_new /= np.linalg.norm(x_new)
                    y_new = np.cross(z_new, x_new)
                    y_new /= np.linalg.norm(y_new)
                    mat = np.column_stack((x_new, y_new, z_new))
                    return mat.flatten()

                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_ARROW,
                    size=[radius, radius, norm * scale], 
                    pos=pos,
                    mat=get_z_alignment_matrix(vector),
                    rgba=color
                )
                viewer.user_scn.ngeom += 1

            # Visualize Target
            load_pos = data.xpos[body_id]
            # add_arrow(pos=load_pos, vector=target_dir, color=[0, 1, 0, 1], scale=1.0, radius=0.03)

            # Visualize Ant Forces
            # We scale the visualization up because ant forces are usually small
            FORCE_VIS_SCALE = 100.0 
            ant_global_positions = load_pos + lever_arms_global

            for i in range(ANT_COUNT):
                # Only draw arrow if ant is active (applying force)
                if swarm.states[i] != 0:
                    if swarm.states[i] == 1:
                        arrow_color = color_informed
                    else:
                        arrow_color = color_puller
                    add_arrow(
                        pos=ant_global_positions[i], 
                        vector=forces_global[i], 
                        color=arrow_color, 
                        scale=0.8, 
                        radius=0.008
                    )

            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()