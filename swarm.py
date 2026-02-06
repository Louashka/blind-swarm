import numpy as np


class Load:
    def __init__(self) -> None:
        self.linear_damping = 0.0   # gamma (N*s/m in SI terms)
        self.angular_damping = 0.0  # gamma_rot (N*m*s/rad in SI terms)


def _wrap_pi(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi

class AntSwarm:
    """
    Ant swarm model following the Supplementary Notes you provided.

    Key implementation details:
    - Stores orientation as an angle offset relative to the local outward normal
      in the *load frame*. This preserves "retain angle w.r.t normal as load moves".
    - Computes f_loc = gamma(N_att) * v_loc (Eq. 2), where v_loc is in world frame.
    - Implements Eq. (3) microscopic damping coefficients for both logic and physics.
    - Implements near-wall detachment + attachment exclusion within epsilon.
    """

    def __init__(
        self,
        num_sites: int,
        attachment_sites_local_pos_dyn: np.ndarray,  # positions in load frame (planar), used for dynamics
        attachment_sites_local_normals: np.ndarray,  # normals in load frame (planar)
        puzzle_geometry: dict,
        wall_aabbs_2d=None,          # list of (minx, miny, maxx, maxy)
        eps_wall: float = 0.0005,    # 0.5 mm in meters (paper)
    ):
        self.num_sites = int(num_sites)
        self.sites_pos_local = np.asarray(attachment_sites_local_pos_dyn, dtype=float)
        self.sites_normals_local = np.asarray(attachment_sites_local_normals, dtype=float)

        assert self.sites_pos_local.shape == (self.num_sites, 3)
        assert self.sites_normals_local.shape == (self.num_sites, 3)

        # State: 0=Empty, 1=Informed Puller, 2=Uninformed Puller, 3=Uninformed Lifter
        self.states = np.zeros(self.num_sites, dtype=int)

        # Orientation representation: angle offset relative to outward normal (load frame),
        # constrained to [-phi_max, +phi_max] at (re)orientation events.
        # Initialize randomly within allowed biomechanical range.
        self.phi_max = np.deg2rad(52)
        self.offset_angles = np.random.uniform(-self.phi_max, self.phi_max, self.num_sites)

        # Cached world-frame vectors (updated each update_logic call)
        self.p_world = np.zeros((self.num_sites, 3), dtype=float)     # carrier body axis unit vector (world)
        self.normal_world = np.zeros((self.num_sites, 3), dtype=float)

        # Local force signal (world)
        self.f_loc = np.zeros((self.num_sites, 3), dtype=float)

        # Wall proximity
        self.wall_aabbs_2d = wall_aabbs_2d or []
        self.eps_wall = float(eps_wall)
        self.blocked = np.zeros(self.num_sites, dtype=bool)

        # Gillespie timer
        self.time_to_next_event = None  # countdown in seconds; None means "not initialized"

        # Store puzzle geometry (chamber logic for informed)
        self.puzzle_geometry = puzzle_geometry

        # Parameters (Table S5)
        self.k_on = 0.015
        self.k_forget = 0.09
        self.k_c = 1.0
        self.k_off = 0.015
        self.k_orient = 0.7

        self.f_0 = 2.8
        self.F_ind = 10 * self.f_0

        self.gamma_per_ant = 1.48
        self.gamma_rot_per_ant = 1.44
        self.load = Load()

        # Propensity components
        self.R_att = 0.0
        self.R_forget = 0.0
        self.R_c = 0.0
        self.R_det = 0.0
        self.R_orient = 0.0
        self.R_tot = 0.0

        # cached exp weights (for switch event)
        self.exp_neg = np.ones(self.num_sites)
        self.exp_pos = np.ones(self.num_sites)

    # --- index sets ---
    @property
    def empty(self) -> np.ndarray:
        return np.where(self.states == 0)[0]

    @property
    def informed(self) -> np.ndarray:
        return np.where(self.states == 1)[0]

    @property
    def pullers(self) -> np.ndarray:
        return np.where((self.states == 1) | (self.states == 2))[0]

    @property
    def uninformed(self) -> np.ndarray:
        return np.where((self.states == 2) | (self.states == 3))[0]

    @property
    def uninformed_pullers(self) -> np.ndarray:
        return np.where(self.states == 2)[0]

    @property
    def lifters(self) -> np.ndarray:
        return np.where(self.states == 3)[0]

    # --- geometry helpers ---
    def _update_normals_and_p_world(self, load_rotation_matrix: np.ndarray):
        """
        Update world-frame outward normals and carrier body-axis vectors p_world
        from stored offset angles (load-frame relative to normals).
        """
        R = load_rotation_matrix
        # normals in world
        self.normal_world = (R @ self.sites_normals_local.T).T

        # normalize 2D normals and compute their angles
        n2 = self.normal_world[:, :2]
        n_norm = np.linalg.norm(n2, axis=1)
        n_norm = np.where(n_norm < 1e-12, 1.0, n_norm)
        n2_unit = n2 / n_norm[:, None]

        theta_n = np.arctan2(n2_unit[:, 1], n2_unit[:, 0])
        theta_p = theta_n + self.offset_angles

        self.p_world[:, 0] = np.cos(theta_p)
        self.p_world[:, 1] = np.sin(theta_p)
        self.p_world[:, 2] = 0.0

    # --- wall proximity ---
    @staticmethod
    def _point_aabb_distance_2d(xy: np.ndarray, aabb) -> float:
        x, y = float(xy[0]), float(xy[1])
        minx, miny, maxx, maxy = aabb
        dx = max(minx - x, 0.0, x - maxx)
        dy = max(miny - y, 0.0, y - maxy)
        return float(np.hypot(dx, dy))

    def _update_blocked_sites(self, ant_global_positions: np.ndarray):
        """
        Mark sites within eps of any wall AABB as blocked.
        """
        if not self.wall_aabbs_2d:
            self.blocked[:] = False
            return

        xy = ant_global_positions[:, :2]
        blocked = np.zeros(self.num_sites, dtype=bool)
        for i in range(self.num_sites):
            dmin = float("inf")
            for aabb in self.wall_aabbs_2d:
                d = self._point_aabb_distance_2d(xy[i], aabb)
                if d < dmin:
                    dmin = d
                    if dmin <= self.eps_wall:
                        break
            blocked[i] = dmin <= self.eps_wall
        self.blocked[:] = blocked

    def _apply_wall_detachments(self):
        """
        Paper rule: carriers within eps of wall are removed.
        We apply this to *all* occupied states (including informed).
        """
        to_detach = self.blocked & (self.states != 0)
        if np.any(to_detach):
            self.states[to_detach] = 0
            self.offset_angles[to_detach] = 0.0  # reset; value irrelevant while empty

    # --- damping and local signal ---
    def _update_effective_damping(self):
        """
        Implements Eq. (3) for gamma and gamma_rot based on N_att.
        """
        N_att = int(np.count_nonzero(self.states))
        N0 = self.num_sites / 5.0

        if N_att <= N0:
            self.load.linear_damping = float(self.gamma_per_ant * self.num_sites)
            self.load.angular_damping = float(self.gamma_rot_per_ant * self.num_sites)
        else:
            self.load.linear_damping = float(self.gamma_per_ant * N_att)
            self.load.angular_damping = float(self.gamma_rot_per_ant * N_att)

    def _calculate_f_loc(self, load_linear_vel_world: np.ndarray, load_angular_vel_world: np.ndarray, load_rotation_matrix: np.ndarray):
        """
        f_loc_i = gamma * v_loc_i
        v_loc_i = v_cm + omega x r_i
        where r_i is site position relative to COM in world.
        """
        self._update_effective_damping()

        # r_global = R * r_local
        r_global = (load_rotation_matrix @ self.sites_pos_local.T).T

        omega = np.array([0.0, 0.0, float(load_angular_vel_world[2])])
        v_rot = np.cross(omega, r_global)
        v_loc = load_linear_vel_world + v_rot

        self.f_loc = self.load.linear_damping * v_loc

    # --- propensities ---
    def _calculate_propensities(self):
        """
        Compute total propensities (Eqs. 4–9) using current p_world and f_loc,
        with attachment excluded on blocked sites.
        """
        # Attachment: only empty AND not blocked
        empty_and_allowed = np.where((self.states == 0) & (~self.blocked))[0]
        N_empty_allowed = len(empty_and_allowed)

        N_info = len(self.informed)
        N_uninfo = len(self.uninformed)
        N_p_total = len(self.pullers)

        self.R_att = self.k_on * N_empty_allowed
        self.R_forget = self.k_forget * N_info

        # Role switching (Eq. 6): weights depend on p_i · f_loc / F_ind
        dot_prod = np.sum(self.p_world * self.f_loc, axis=1) / self.F_ind

        self.exp_neg = np.exp(-dot_prod)
        self.exp_pos = np.exp(+dot_prod)

        u_pullers = self.uninformed_pullers
        u_lifters = self.lifters

        rate_p_to_l = float(np.sum(self.exp_neg[u_pullers])) if len(u_pullers) else 0.0
        rate_l_to_p = float(np.sum(self.exp_pos[u_lifters])) if len(u_lifters) else 0.0
        self.R_c = self.k_c * (rate_p_to_l + rate_l_to_p)

        self.R_det = self.k_off * N_uninfo
        self.R_orient = self.k_orient * N_p_total

        self.R_tot = self.R_att + self.R_forget + self.R_c + self.R_det + self.R_orient

    def _execute_switch_event(self):
        """
        Select which specific uninformed ant switches role, weighted by Eq. (1)/(6).
        """
        u_pullers = self.uninformed_pullers
        u_lifters = self.lifters

        if len(u_pullers) == 0 and len(u_lifters) == 0:
            return None

        w_p2l = self.exp_neg[u_pullers] if len(u_pullers) else np.array([])
        w_l2p = self.exp_pos[u_lifters] if len(u_lifters) else np.array([])

        sum_p2l = float(np.sum(w_p2l)) if w_p2l.size else 0.0
        sum_l2p = float(np.sum(w_l2p)) if w_l2p.size else 0.0
        total = sum_p2l + sum_l2p
        if total <= 0.0:
            return None

        r = np.random.uniform(0.0, total)
        if r < sum_p2l:
            cumsum = np.cumsum(w_p2l)
            idx_in_subset = int(np.searchsorted(cumsum, r, side="right"))
            idx_in_subset = min(idx_in_subset, len(u_pullers) - 1)
            global_idx = int(u_pullers[idx_in_subset])
            self.states[global_idx] = 3  # puller -> lifter
            return global_idx
        else:
            r -= sum_p2l
            cumsum = np.cumsum(w_l2p)
            idx_in_subset = int(np.searchsorted(cumsum, r, side="right"))
            idx_in_subset = min(idx_in_subset, len(u_lifters) - 1)
            global_idx = int(u_lifters[idx_in_subset])
            self.states[global_idx] = 2  # lifter -> puller
            return global_idx

    # --- reorientation ---
    def reorient_ant(self, idx: int, load_rotation_matrix: np.ndarray, ant_global_pos: np.ndarray, ant_local_normal: np.ndarray):
        """
        Update offset angle relative to outward normal, clamped to phi_max,
        towards desired direction (informed: chamber target; uninformed: f_loc direction).
        """
        idx = int(idx)

        # Determine desired direction (world)
        if self.states[idx] == 1:  # Informed puller
            ant_x = float(ant_global_pos[0])
            if ant_x < self.puzzle_geometry["chamber1_x_boundary"]:
                target_pos = self.puzzle_geometry["slit1_midpoint"]
                desired = target_pos - ant_global_pos
            elif ant_x < self.puzzle_geometry["chamber2_x_boundary"]:
                target_pos = self.puzzle_geometry["slit2_midpoint"]
                desired = target_pos - ant_global_pos
            else:
                desired = np.array([1.0, 0.0, 0.0], dtype=float)
        elif self.states[idx] == 2:  # Uninformed puller
            f = self.f_loc[idx]
            nrm = float(np.linalg.norm(f))
            if nrm < 1e-9:
                return
            desired = f / nrm
        else:
            return  # lifter/empty do not reorient

        # Normalize desired (2D)
        d2 = desired[:2]
        dn = float(np.linalg.norm(d2))
        if dn < 1e-9:
            return
        d2 = d2 / dn
        theta_d = float(np.arctan2(d2[1], d2[0]))

        # Outward normal in world (2D)
        n_world = (load_rotation_matrix @ ant_local_normal)[:2]
        nn = float(np.linalg.norm(n_world))
        if nn < 1e-9:
            return
        n_world = n_world / nn
        theta_n = float(np.arctan2(n_world[1], n_world[0]))

        diff = _wrap_pi(theta_d - theta_n)
        diff = float(np.clip(diff, -self.phi_max, +self.phi_max))

        self.offset_angles[idx] = diff

    # --- main update ---
    def update_logic(
        self,
        dt: float,
        load_velocity_world: np.ndarray,       # linear velocity of COM (world)
        load_angular_vel_world: np.ndarray,    # angular velocity (world)
        load_rotation_matrix: np.ndarray,
        ant_global_positions: np.ndarray,
    ):
        """
        Updates swarm (Gillespie events) and returns per-site pulling forces in world frame.

        This function:
        1) applies near-wall detachment and blocks near-wall attachments
        2) updates p_world from stored offset angles and current load rotation
        3) updates f_loc from v_loc and gamma(N_att)
        4) advances Gillespie events in a "countdown" fashion over dt
        5) returns forces for current pullers
        """
        dt = float(dt)

        # Wall proximity bookkeeping
        self._update_blocked_sites(ant_global_positions)
        self._apply_wall_detachments()

        # Update p_world and f_loc for current continuous state
        self._update_normals_and_p_world(load_rotation_matrix)
        self._calculate_f_loc(load_velocity_world, load_angular_vel_world, load_rotation_matrix)
        self._calculate_propensities()

        # Initialize or refresh event timer
        if self.R_tot <= 1e-12:
            self.time_to_next_event = float("inf")
        elif self.time_to_next_event is None or not np.isfinite(self.time_to_next_event):
            self.time_to_next_event = -np.log(np.random.rand()) / self.R_tot

        # Advance time by dt, firing all events that occur within dt
        self.time_to_next_event -= dt

        while self.time_to_next_event <= 0.0:
            # Recompute propensities at the event time (states may have changed)
            self._update_normals_and_p_world(load_rotation_matrix)
            self._calculate_f_loc(load_velocity_world, load_angular_vel_world, load_rotation_matrix)
            self._calculate_propensities()

            if self.R_tot <= 1e-12:
                self.time_to_next_event = float("inf")
                break

            r2 = np.random.rand() * self.R_tot

            # --- Attachment (allowed empty only) ---
            if r2 < self.R_att:
                candidates = np.where((self.states == 0) & (~self.blocked))[0]
                if len(candidates) > 0:
                    idx = int(np.random.choice(candidates))
                    self.states[idx] = 1
                    self.reorient_ant(idx, load_rotation_matrix, ant_global_positions[idx], self.sites_normals_local[idx])

            # --- Forgetting ---
            elif r2 < self.R_att + self.R_forget:
                infos = self.informed
                if len(infos) > 0:
                    idx = int(np.random.choice(infos))
                    # Eq. (14) uses p_i · f_loc / F_ind
                    dot_val = float(np.dot(self.p_world[idx], self.f_loc[idx]) / self.F_ind)
                    prob_puller = 1.0 / (1.0 + np.exp(-2.0 * dot_val))
                    self.states[idx] = 2 if (np.random.rand() < prob_puller) else 3

            # --- Role switching ---
            elif r2 < self.R_att + self.R_forget + self.R_c:
                self._execute_switch_event()

            # --- Detachment (uninformed only) ---
            elif r2 < self.R_att + self.R_forget + self.R_c + self.R_det:
                uninf = self.uninformed
                if len(uninf) > 0:
                    idx = int(np.random.choice(uninf))
                    self.states[idx] = 0
                    self.offset_angles[idx] = 0.0

            # --- Reorientation (pullers only) ---
            else:
                pullers = self.pullers
                if len(pullers) > 0:
                    idx = int(np.random.choice(pullers))
                    self.reorient_ant(idx, load_rotation_matrix, ant_global_positions[idx], self.sites_normals_local[idx])

            # Draw next event time and carry over negative remainder
            self._update_normals_and_p_world(load_rotation_matrix)
            self._calculate_f_loc(load_velocity_world, load_angular_vel_world, load_rotation_matrix)
            self._calculate_propensities()

            if self.R_tot <= 1e-12:
                self.time_to_next_event = float("inf")
                break

            tau = -np.log(np.random.rand()) / self.R_tot
            self.time_to_next_event += tau

        # Forces in world: only pullers contribute; lifters = 0
        self._update_normals_and_p_world(load_rotation_matrix)
        forces = np.zeros_like(self.p_world)
        pullers = self.pullers
        if len(pullers) > 0:
            forces[pullers] = self.p_world[pullers] * self.f_0
        return forces