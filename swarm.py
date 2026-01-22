import numpy as np

class Load:
    def __init__(self) -> None:
        self.linear_damping = 0
        self.angular_damping = 0

class AntSwarm:
    def __init__(self, num_sites, attachment_sites_local_pos):
        """
        Ant Swarm Logic based on Gelblum et al.
        """
        self.num_sites = num_sites
        # Shape (N, 3), assuming z=0 for planar physics
        self.sites_pos_local = np.array(attachment_sites_local_pos)
        
        # State: 0=Empty, 1=Informed Puller, 2=Uninformed Puller, 3=Uninformed Lifter
        self.states = np.zeros(num_sites, dtype=int) 
        
        # Orientations p_i (N, 3). 
        # For Pullers (1, 2), this is the pulling direction.
        # For Lifters/Empty, this value is ignored/irrelevant but kept for array shape.
        self.orientations = np.zeros((num_sites, 3))
        
        # Initialize random orientations for start
        random_angles = np.random.uniform(0, 2*np.pi, num_sites)
        self.orientations[:, 0] = np.cos(random_angles)
        self.orientations[:, 1] = np.sin(random_angles)

        # Local force signal f_loc (N, 3)
        self.f_loc = np.zeros((num_sites, 3))

        # Time accumulation for Gillespie
        self.time_accumulator = 0.0

        # Parameters (from Table S5)
        self.k_on = 0.015       # Rate: Empty -> Informed
        self.k_forget = 0.09    # Rate: Informed -> Uninformed
        self.k_c = 1.0          # Basal role-switching rate
        self.k_off = 0.015      # Rate: Uninformed -> Empty
        self.k_orient = 0.7     # Reorientation rate
        self.f_0 = 2.8          # Force magnitude
        self.F_ind = 10 * self.f_0  # Individuality parameter
        self.phi_max = np.deg2rad(52) # Angular limit

        self.gamma_per_ant = 1.48 
        self.gamma_rot_per_ant = 1.44

        self.load = Load()

    @property
    def empty(self) -> np.ndarray:
        return np.where(self.states == 0)[0]

    @property
    def informed(self) -> np.ndarray:
        return np.where(self.states == 1)[0]

    @property
    def uninformed(self) -> np.ndarray:
        return np.where((self.states == 2) | (self.states == 3))[0]

    @property
    def pullers(self) -> np.ndarray:
        return np.where((self.states == 1) | (self.states == 2))[0]
    
    @property
    def uninformed_pullers(self) -> np.ndarray:
        return np.where(self.states == 2)[0]

    @property
    def lifters(self) -> np.ndarray:
        return np.where(self.states == 3)[0]

    def calculate_f_loc(self, load_linear_vel, load_angular_vel, load_rotation_matrix):
        """
        Calculate f_loc = gamma * v_loc for all sites.
        v_loc_i = v_cm + omega x r_i
        """
        # 1. Calculate macroscopic gamma based on N_att (Eq. 3)
        N_att = np.count_nonzero(self.states)
        # If N_att is low, friction is high (simulating load touching ground)
        if N_att <= self.num_sites / 5:
            self.load.linear_damping = self.gamma_per_ant * self.num_sites
            self.load.angular_damping = self.gamma_rot_per_ant * self.num_sites
        else:
            self.load.linear_damping = self.gamma_per_ant * N_att
            self.load.angular_damping = self.gamma_rot_per_ant * N_att

        # 2. Calculate local velocities in Global Frame
        # r_i_global = R * r_i_local
        r_global = np.dot(self.sites_pos_local, load_rotation_matrix.T)
        
        # angular velocity vector (0, 0, w)
        omega_vec = np.array([0, 0, load_angular_vel[2]])
        
        # v_rot = omega x r
        v_rot = np.cross(omega_vec, r_global) # Broadcasting works here
        
        # v_loc = v_linear + v_rot
        v_loc = load_linear_vel + v_rot
        
        # 3. f_loc = gamma * v_loc
        self.f_loc = self.load.linear_damping * v_loc

    def calculate_propensities(self):
        """
        Calculate rates for Gillespie algorithm.
        """
        N_empty = len(self.empty)
        N_info = len(self.informed)
        N_uninfo = len(self.uninformed)
        N_p_total = len(self.pullers)

        self.R_att = self.k_on * N_empty
        self.R_forget = self.k_forget * N_info

        # Role Switching Rates (Eq. 6)
        # Calculate dot product (p_i . f_loc / F_ind)
        dot_prod = np.sum(self.orientations * self.f_loc, axis=1) / self.F_ind
        
        # Store exponentials for use in event selection later
        self.exp_neg = np.exp(-dot_prod) # Favors becoming Lifter
        self.exp_pos = np.exp(dot_prod)  # Favors becoming Puller
        
        # Sum rates only for active uninformed ants
        # Pullers switch to lifters with rate proportional to exp_neg
        # Lifters switch to pullers with rate proportional to exp_pos
        rate_p_to_l = np.sum(self.exp_neg[self.uninformed_pullers])
        rate_l_to_p = np.sum(self.exp_pos[self.lifters])
        
        self.R_c = self.k_c * (rate_p_to_l + rate_l_to_p)

        self.R_det = self.k_off * N_uninfo
        self.R_orient = self.k_orient * N_p_total

        self.R_tot = self.R_att + self.R_forget + self.R_c + self.R_det + self.R_orient

    def _execute_switch_event(self):
        """
        Selects which specific ant switches role based on weights.
        """
        u_pullers = self.uninformed_pullers
        u_lifters = self.lifters
        
        # Weights
        w_p2l = self.exp_neg[u_pullers]
        w_l2p = self.exp_pos[u_lifters]
        
        sum_p2l = np.sum(w_p2l)
        sum_l2p = np.sum(w_l2p)
        total_weight = sum_p2l + sum_l2p
        
        if total_weight == 0: return

        r = np.random.uniform(0, total_weight)
        
        if r < sum_p2l:
            # A puller becomes a lifter
            # Sample index from u_pullers based on w_p2l
            cumsum = np.cumsum(w_p2l)
            idx_in_subset = np.searchsorted(cumsum, r)
            global_idx = u_pullers[min(idx_in_subset, len(u_pullers)-1)]
            self.states[global_idx] = 3
        else:
            # A lifter becomes a puller
            r -= sum_p2l
            cumsum = np.cumsum(w_l2p)
            idx_in_subset = np.searchsorted(cumsum, r)
            global_idx = u_lifters[min(idx_in_subset, len(u_lifters)-1)]
            self.states[global_idx] = 2

    def reorient_ant(self, idx, target_dir_global, load_rotation_matrix):
        """
        Updates orientation p_i for ant idx.
        """
        # 1. Determine Desired Direction
        if self.states[idx] == 1: # Informed
            desired_dir = target_dir_global
        elif self.states[idx] == 2: # Uninformed Puller
            f_mag = np.linalg.norm(self.f_loc[idx])
            if f_mag > 1e-6:
                desired_dir = self.f_loc[idx] / f_mag
            else:
                desired_dir = self.orientations[idx]
        else:
            return 

        # 2. Calculate Local Normal in Global Frame
        # Assuming local normal points radially outward from center of load?
        # Or defined by geometry. Here assuming radial for simplicity:
        # normal_local = normalized(pos_local)
        pos_local = self.sites_pos_local[idx]
        norm_local = pos_local / (np.linalg.norm(pos_local) + 1e-9)
        normal_global = np.dot(load_rotation_matrix, norm_local)

        # 3. Apply Angular Limit (phi_max)
        # Project to 2D
        normal_2d = normal_global[:2]
        desired_2d = desired_dir[:2]
        
        theta_n = np.arctan2(normal_2d[1], normal_2d[0])
        theta_d = np.arctan2(desired_2d[1], desired_2d[0])
        
        diff = theta_d - theta_n
        diff = (diff + np.pi) % (2 * np.pi) - np.pi # Wrap -pi to pi
        
        if diff > self.phi_max:
            final_angle = theta_n + self.phi_max
        elif diff < -self.phi_max:
            final_angle = theta_n - self.phi_max
        else:
            final_angle = theta_d
            
        self.orientations[idx] = np.array([np.cos(final_angle), np.sin(final_angle), 0])

    def update_logic(self, dt, load_velocity_global, load_angular_vel, target_direction_global, load_rotation_matrix):
        """
        Main simulation step.
        """
        # 1. Update Physics-based values
        self.calculate_f_loc(load_velocity_global, load_angular_vel, load_rotation_matrix)
        
        self.time_accumulator += dt
        
        # 2. Gillespie Loop
        while True:
            self.calculate_propensities()
            
            if self.R_tot <= 1e-9:
                self.time_to_next_event = float('inf')
            else:
                r_1 = np.random.rand()
                self.time_to_next_event = (1.0 / self.R_tot) * np.log(1.0 / r_1)
                
            if self.time_accumulator >= self.time_to_next_event:
                self.time_accumulator -= self.time_to_next_event
                
                r_2 = np.random.rand()
                
                # Thresholds
                thresh_att = self.R_att / self.R_tot
                thresh_forget = (self.R_att + self.R_forget) / self.R_tot
                thresh_switch = (self.R_att + self.R_forget + self.R_c) / self.R_tot
                thresh_det = (self.R_att + self.R_forget + self.R_c + self.R_det) / self.R_tot
                
                if r_2 < thresh_att:
                    # Attachment
                    if len(self.empty) > 0:
                        idx = np.random.choice(self.empty)
                        self.states[idx] = 1 # Informed
                        # Initialize orientation towards target immediately
                        self.reorient_ant(idx, target_direction_global, load_rotation_matrix)
                        
                elif r_2 < thresh_forget:
                    # Forgetting
                    if len(self.informed) > 0:
                        idx = np.random.choice(self.informed)
                        # Decide Puller vs Lifter (Eq. 14)
                        dot_val = np.dot(self.orientations[idx], self.f_loc[idx]) / self.F_ind
                        prob_puller = 1.0 / (1.0 + np.exp(-2 * dot_val))
                        
                        if np.random.rand() < prob_puller:
                            self.states[idx] = 2 # Uninformed Puller
                        else:
                            self.states[idx] = 3 # Lifter
                            
                elif r_2 < thresh_switch:
                    # Role Switching
                    self._execute_switch_event()

                    # idx = np.random.choice(self.uninformed)
                    # if self.states[idx] == 2:
                    #     self.states[idx] = 3
                    # else:
                    #     self.states[idx] = 2
                    
                elif r_2 < thresh_det:
                    # Detachment
                    if len(self.uninformed) > 0:
                        idx = np.random.choice(self.uninformed)
                        self.states[idx] = 0 # Empty
                        self.orientations[idx] = 0 # Reset orientation
                        
                else:
                    # Reorientation
                    candidates = self.pullers # Informed + Uninformed Pullers
                    if len(candidates) > 0:
                        idx = np.random.choice(candidates)
                        self.reorient_ant(idx, target_direction_global, load_rotation_matrix)
            else:
                break
        
        # 3. Return forces to be applied to the physics engine
        # Only pullers exert force f_0 in direction p_i
        active_pullers = self.pullers
        forces = np.zeros_like(self.orientations)
        forces[active_pullers] = self.orientations[active_pullers] * self.f_0
        
        return forces

    