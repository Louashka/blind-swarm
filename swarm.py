import numpy as np

class AntSwarm:
    def __init__(self, num_sites, attachment_sites_local_pos):
        """
        Vectorized Ant Swarm Logic (No Lifters).
        """
        self.num_sites = num_sites
        self.sites_pos_local = np.array(attachment_sites_local_pos)
        
        # State: 0=Empty, 1=Informed, 2=Uninformed
        self.states = np.zeros(num_sites, dtype=int) 
        
        # Initialize uninformed directions (N, 3)
        random_angles = np.random.uniform(0, 2*np.pi, num_sites)
        self.uninformed_directions = np.column_stack((
            np.cos(random_angles), 
            np.sin(random_angles), 
            np.zeros(num_sites)
        ))

        # Parameters
        self.k_on = 0.015       # Rate: Empty -> Informed
        self.k_forget = 0.09    # Rate: Informed -> Uninformed
        self.k_off = 0.015      # Rate: Uninformed -> Empty
        self.f_0 = 2.8 
        self.conformity_rate = 0.1 

    def _normalize_rows(self, vectors):
        """Vectorized normalization of an (N, 3) array."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0 
        return vectors / norms

    def update_logic(self, dt, load_velocity_global, target_direction_global, load_rotation_matrix):
        """
        Vectorized stochastic updates.
        """
        # Pre-calculate global vectors
        vel_norm = np.linalg.norm(load_velocity_global)
        vel_dir = load_velocity_global / vel_norm if vel_norm > 1e-6 else np.zeros(3)
        
        # Generate random numbers for transitions
        r_attach = np.random.rand(self.num_sites)
        r_forget = np.random.rand(self.num_sites)
        r_detach = np.random.rand(self.num_sites)

        # --- 1. State Transitions (Vectorized) ---
        
        # Masks for current states
        is_empty = (self.states == 0)
        is_informed = (self.states == 1)
        is_uninformed = (self.states == 2)

        # Empty -> Informed
        transition_to_informed = is_empty & (r_attach < self.k_on * dt)
        self.states[transition_to_informed] = 1

        # Informed -> Uninformed
        transition_to_uninformed = is_informed & (r_forget < self.k_forget * dt)
        self.states[transition_to_uninformed] = 2
        
        # Reset direction for newly uninformed ants to current velocity
        if np.any(transition_to_uninformed) and vel_norm > 0.01:
            self.uninformed_directions[transition_to_uninformed] = vel_dir

        # Uninformed -> Empty (Detachment)
        # Note: Only Uninformed ants detach in this model topology
        transition_to_empty = is_uninformed & (r_detach < self.k_off * dt)
        self.states[transition_to_empty] = 0

        # --- 2. Conformity Logic (Update Directions) ---
        
        # Only update directions if moving
        if vel_norm > 0.01:
            # Re-evaluate mask as states might have changed (Informed -> Uninformed)
            mask_u = (self.states == 2)
            if np.any(mask_u):
                # Vectorized blend: current_dir * (1-rate) + vel_dir * rate
                new_dirs = (
                    self.uninformed_directions[mask_u] * (1 - self.conformity_rate) + 
                    vel_dir * self.conformity_rate
                )
                self.uninformed_directions[mask_u] = self._normalize_rows(new_dirs)

    def get_forces(self, target_direction_global):
        """
        Vectorized force calculation.
        Returns (N, 3) array of forces.
        """
        forces = np.zeros((self.num_sites, 3))
        
        target_dir = target_direction_global / np.linalg.norm(target_direction_global)
        
        is_informed = (self.states == 1)
        is_uninformed = (self.states == 2)
        
        # Apply Informed Forces (All informed ants pull to target)
        if np.any(is_informed):
            forces[is_informed] = target_dir * self.f_0
            
        # Apply Uninformed Forces (Each pulls in its own specific direction)
        if np.any(is_uninformed):
            forces[is_uninformed] = self.uninformed_directions[is_uninformed] * self.f_0
            
        return forces