import mujoco
import mujoco.viewer
import numpy as np
import time

# --- Helper function for orientation ---
def vec_to_quat(v1, v2):
    """ Calculates the quaternion to rotate v1 to v2. """
    v1 = v1 / (np.linalg.norm(v1) + 1e-9)
    v2 = v2 / (np.linalg.norm(v2) + 1e-9)
    axis = np.cross(v1, v2)
    angle = np.arccos(np.dot(v1, v2))
    if np.linalg.norm(axis) < 1e-9:
        if np.allclose(v1, v2):
            return np.array([1.0, 0.0, 0.0, 0.0]) # w, x, y, z
        else: # anti-parallel
            perp_axis = np.cross(v1, np.array([1, 0, 0]))
            if np.linalg.norm(perp_axis) < 1e-9:
                perp_axis = np.cross(v1, np.array([0, 1, 0]))
            perp_axis /= np.linalg.norm(perp_axis)
            return np.array([0.0, *perp_axis])
    
    axis /= np.linalg.norm(axis)
    sin_half = np.sin(angle / 2)
    cos_half = np.cos(angle / 2)
    return np.array([cos_half, axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half])

# --- Your Physics Classes (Unchanged) ---

class SoftBody:
    def __init__(self, node_positions, node_masses, springs_info):
        self.nodes_pos = np.array(node_positions, dtype=float)
        self.nodes_vel = np.zeros_like(self.nodes_pos)
        self.nodes_mass = np.array(node_masses).reshape(-1, 1)
        self.springs = springs_info
        self.num_nodes = len(self.nodes_pos)

    def update_physics(self, dt, external_forces):
        spring_forces = np.zeros_like(self.nodes_pos)
        for spring in self.springs:
            p1_idx, p2_idx = spring['p1'], spring['p2']
            p1_pos, p2_pos = self.nodes_pos[p1_idx], self.nodes_pos[p2_idx]
            vec = p2_pos - p1_pos
            dist = np.linalg.norm(vec)
            if dist == 0: continue
            force_magnitude = spring['stiffness'] * (dist - spring['rest_length'])
            force_vec = force_magnitude * (vec / dist)
            spring_forces[p1_idx] += force_vec
            spring_forces[p2_idx] -= force_vec

        damping_forces = -0.8 * self.nodes_vel
        gravity = self.nodes_mass * np.array([0, 0, -2.81])
        
        total_forces = spring_forces + external_forces + damping_forces + gravity
        
        for i in range(self.num_nodes):
            if self.nodes_pos[i, 2] < 0.0:
                self.nodes_pos[i, 2] = 0.0
                self.nodes_vel[i, 2] = -0.5 * self.nodes_vel[i, 2]

        acceleration = total_forces / self.nodes_mass
        self.nodes_vel += acceleration * dt
        self.nodes_pos += self.nodes_vel * dt

    def get_node_positions(self):
        return self.nodes_pos

class AntSwarm:
    def __init__(self, soft_body, puzzle_geometry):
        self.soft_body = soft_body
        self.num_sites = soft_body.num_nodes
        self.states = np.zeros(self.num_sites, dtype=int)
        self.orientations = np.zeros((self.num_sites, 3))
        self.f_0 = 1.0
        self.puzzle_target = puzzle_geometry['target_pos']
        self.timer = 0.0
        self.pull_ants = [1, 3]

    @property
    def pullers(self):
        return np.where(self.states == 2)[0]

    def update_logic(self, dt):
        self.timer += dt
        if self.timer > 2.0:
            for i in self.pull_ants:
                self.states[i] = 2
                current_pos = self.soft_body.get_node_positions()[i]
                desired_dir = self.puzzle_target - current_pos
                desired_dir /= (np.linalg.norm(desired_dir) + 1e-9)
                self.orientations[i] = desired_dir
        
        for i in range(self.num_sites):
            if self.states[i] == 0:
                self.states[i] = 1

        forces = np.zeros((self.num_sites, 3))
        puller_indices = self.pullers
        if len(puller_indices) > 0:
            forces[puller_indices] = self.orientations[puller_indices] * self.f_0
        return forces

# --- Main Simulation and Rendering Script ---

# 1. Load the MuJoCo model from the XML string
# We add a name to the cylinder geoms to be able to access them later
xml = """
<mujoco model="soft_body_visualization">
  <option gravity="0 0 0" timestep="0.002" />
  <visual>
    <headlight active="1" ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8" specular="0.1 0.1 0.1"/>
    <map znear="0.01" zfar="50"/> <quality shadowsize="4096"/>
  </visual>
  <asset>
    <material name="detached_mat" rgba="0.5 0.5 0.5 0.7"/>
    <material name="attached_mat" rgba="0.1 0.8 0.1 0.9"/>
    <material name="pulling_mat" rgba="0.9 0.2 0.2 1.0"/>
  </asset>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" diffuse="1 1 1"/>
    <geom type="plane" size="5 5 0.1" rgba="0.9 0.9 0.9 1"/>
    <body name="target_vis" pos="3 3 0.1"><geom type="sphere" size="0.1" rgba="1 1 0 0.5"/></body>
    <body name="node_0" mocap="true" pos="0 0 0.1"><geom type="sphere" size="0.05" rgba="0.2 0.2 0.8 1" group="1"/><geom name="ant_0_geom" type="sphere" size="0.03" pos="0 0 0.05" material="detached_mat"/></body>
    <body name="node_1" mocap="true" pos="0.5 0 0.1"><geom type="sphere" size="0.05" rgba="0.2 0.2 0.8 1" group="1"/><geom name="ant_1_geom" type="sphere" size="0.03" pos="0 0 0.05" material="detached_mat"/></body>
    <body name="node_2" mocap="true" pos="1.0 0 0.1"><geom type="sphere" size="0.05" rgba="0.2 0.2 0.8 1" group="1"/><geom name="ant_2_geom" type="sphere" size="0.03" pos="0 0 0.05" material="detached_mat"/></body>
    <body name="node_3" mocap="true" pos="1.5 0 0.1"><geom type="sphere" size="0.05" rgba="0.2 0.2 0.8 1" group="1"/><geom name="ant_3_geom" type="sphere" size="0.03" pos="0 0 0.05" material="detached_mat"/></body>
    <body name="node_4" mocap="true" pos="2.0 0 0.1"><geom type="sphere" size="0.05" rgba="0.2 0.2 0.8 1" group="1"/><geom name="ant_4_geom" type="sphere" size="0.03" pos="0 0 0.05" material="detached_mat"/></body>
    <body name="link_0_1" mocap="true" pos="0.25 0 0.1"><geom name="link_0_1_geom" type="cylinder" size="0.02 0.25" rgba="0.5 0.5 0.5 1" group="1"/></body>
    <body name="link_1_2" mocap="true" pos="0.75 0 0.1"><geom name="link_1_2_geom" type="cylinder" size="0.02 0.25" rgba="0.5 0.5 0.5 1" group="1"/></body>
    <body name="link_2_3" mocap="true" pos="1.25 0 0.1"><geom name="link_2_3_geom" type="cylinder" size="0.02 0.25" rgba="0.5 0.5 0.5 1" group="1"/></body>
    <body name="link_3_4" mocap="true" pos="1.75 0 0.1"><geom name="link_3_4_geom" type="cylinder" size="0.02 0.25" rgba="0.5 0.5 0.5 1" group="1"/></body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# 2. Setup your physics simulation
NUM_NODES = 5
REST_LENGTH = 0.5
initial_positions = [[i * REST_LENGTH, 0, 0.1] for i in range(NUM_NODES)]
masses = [0.1] * NUM_NODES
springs = []
for i in range(NUM_NODES - 1):
    springs.append({'p1': i, 'p2': i+1, 'rest_length': REST_LENGTH, 'stiffness': 80})

my_soft_body = SoftBody(initial_positions, masses, springs)
puzzle_geom = {'target_pos': np.array([3.0, 3.0, 0.1])}
my_ant_swarm = AntSwarm(my_soft_body, puzzle_geom)

# ###################### CHANGE STARTS HERE ######################

# Get body IDs first
node_body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'node_{i}') for i in range(NUM_NODES)]
link_body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'link_{i}_{i+1}') for i in range(NUM_NODES - 1)]

# Then, map body IDs to mocap IDs
mocap_node_ids = [model.body_mocapid[bid] for bid in node_body_ids]
mocap_link_ids = [model.body_mocapid[bid] for bid in link_body_ids]

# Get geom IDs for ants (for color) and links (for resizing)
ant_geom_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'ant_{i}_geom') for i in range(NUM_NODES)]
link_geom_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'link_{i}_{i+1}_geom') for i in range(NUM_NODES - 1)]

# ###################### CHANGE ENDS HERE ######################

# Get material IDs for color changes
mat_ids = {
    0: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, 'detached_mat'),
    1: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, 'attached_mat'),
    2: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, 'pulling_mat')
}

# 3. Launch the viewer and run the loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    time.sleep(5)
    while viewer.is_running():
        step_start = time.time()
        
        dt = 0.002
        ant_forces = my_ant_swarm.update_logic(dt)
        my_soft_body.update_physics(dt, ant_forces)
        
        node_positions = my_soft_body.get_node_positions()
        
        # Update node mocap bodies using their MOCAP IDs
        for i in range(NUM_NODES):
            data.mocap_pos[mocap_node_ids[i]] = node_positions[i]
        
        # Update link mocap bodies using their MOCAP IDs
        for i in range(NUM_NODES - 1):
            pos1 = node_positions[i]
            pos2 = node_positions[i+1]
            
            midpoint = (pos1 + pos2) / 2
            vec = pos2 - pos1
            length = np.linalg.norm(vec)
            
            # Use the correct MOCAP ID to set position and orientation
            mocap_id = mocap_link_ids[i]
            data.mocap_pos[mocap_id] = midpoint
            
            z_axis = np.array([0, 0, 1])
            quat = vec_to_quat(z_axis, vec)
            data.mocap_quat[mocap_id] = quat

            # ###################### IMPROVEMENT STARTS HERE ######################
            # Dynamically update the cylinder's length
            geom_id = link_geom_ids[i]
            # size[1] is the half-length of the cylinder
            model.geom_size[geom_id][1] = length / 2.0
            # ###################### IMPROVEMENT ENDS HERE ######################

        # Update ant geom colors based on state
        for i in range(NUM_NODES):
            state = my_ant_swarm.states[i]
            model.geom_matid[ant_geom_ids[i]] = mat_ids[state]
            
        mujoco.mj_forward(model, data)

        viewer.sync()
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)