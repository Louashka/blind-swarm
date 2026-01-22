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

# --- 2. Helper: Generate Attachment Points ---
def generate_perimeter_points(corners, num_ants, ant_radius, body_z):
    """
    Generates positions around the contour.
    """
    points = []
    
    # Calculate local Z position relative to the body center
    # If body is at 0.1, and ground is 0.0, relative Z is -0.1 + radius
    z_ground = -body_z + ant_radius 
    
    closed_corners = np.vstack([corners, corners[0]])
    
    # Calculate total perimeter length
    total_perimeter = 0
    segment_lengths = []
    for i in range(len(closed_corners) - 1):
        p1 = closed_corners[i]
        p2 = closed_corners[i+1]
        dist = np.linalg.norm(p2 - p1)
        segment_lengths.append(dist)
        total_perimeter += dist
        
    step_dist = total_perimeter / num_ants
    
    for i in range(num_ants):
        target_dist = i * step_dist
        
        accumulated_len = 0
        for seg_idx, seg_len in enumerate(segment_lengths):
            if accumulated_len + seg_len >= target_dist:
                p1 = closed_corners[seg_idx]
                p2 = closed_corners[seg_idx+1]
                
                edge_vec = p2 - p1
                
                # Normal vector (outward facing)
                normal = np.array([-edge_vec[1], edge_vec[0]])
                norm_len = np.linalg.norm(normal)
                if norm_len > 0:
                    normal = normal / norm_len
                
                local_dist = target_dist - accumulated_len
                ratio = local_dist / seg_len
                
                pos_on_line = p1 + edge_vec * ratio
                final_pos = pos_on_line + (normal * ant_radius)
                
                points.append([final_pos[0], final_pos[1], z_ground])
                break
            else:
                accumulated_len += seg_len

    return np.array(points)

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
    
    ANT_COUNT = 40      
    ANT_RADIUS = 0.02   

    print(f"Extracting geometry from {xml_path}...")
    
    # 1. Parse XML
    try:
        raw_corners, body_z = get_contour_and_body_z(xml_path, body_name="load")
        print(f"Detected body 'load' at Z-height: {body_z}")
    except Exception as e:
        print(f"Error parsing XML geometry: {e}")
        return

    # 2. Generate Points
    ant_local_pos = generate_perimeter_points(raw_corners, ANT_COUNT, ANT_RADIUS, body_z)
    print(f"Generated {len(ant_local_pos)} ants around the contour.")
    
    # 3. Load Model
    model = load_model_with_ants(xml_path, ant_local_pos, ANT_RADIUS)
    data = mujoco.MjData(model)
    
    # 4. Initialize Swarm
    # Note: We pass the local positions to the swarm so it knows where ants are relative to center
    swarm = AntSwarm(ANT_COUNT, ant_local_pos)
    
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
        while viewer.is_running():
            step_start = time.time()
            
            # --- Get Physical State from MuJoCo ---
            # Linear velocity of the load (global frame)
            lin_vel = data.cvel[body_id][3:6]
            # Angular velocity of the load (global frame) -> ADDED THIS
            ang_vel = data.cvel[body_id][0:3]
            # Rotation matrix of the load (global frame)
            rot_mat = data.xmat[body_id].reshape(3, 3)

            # --- Update Swarm Logic ---
            # The swarm needs to know how the load is moving and rotating to calculate
            # the effective force each ant feels and whether they should detach.
            forces_global = swarm.update_logic(
                dt=model.opt.timestep,
                load_velocity_global=lin_vel,
                load_angular_vel=ang_vel,
                target_direction_global=target_dir,
                load_rotation_matrix=rot_mat
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
                    add_arrow(
                        pos=ant_global_positions[i], 
                        vector=forces_global[i], 
                        color=[1, 0, 0, 1], 
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