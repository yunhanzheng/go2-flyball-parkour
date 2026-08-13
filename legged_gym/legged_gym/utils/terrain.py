# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
from numpy.random import choice
from scipy import interpolate
import random
from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from scipy import ndimage
from pydelatin import Delatin
import pyfqmr
from scipy.ndimage import binary_dilation
from legged_gym.utils.flyball_wall import (
    convert_heightfield_to_trimesh_delatin,
    flyball_wall_terrain,
    _hurdle_clear_boxes,
)

# A bar shorter than this counts as flat ground and gets no clearance volume.
_HURDLE_CLEAR_MIN_H = 0.01

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width

        cfg.terrain_proportions = np.array(cfg.terrain_proportions) / np.sum(cfg.terrain_proportions)
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]
        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.terrain_type = np.zeros((cfg.num_rows, cfg.num_cols))
        self.goals = np.zeros((cfg.num_rows, cfg.num_cols, cfg.num_goals, 3))
        self.flyball_pit_x0 = np.zeros((cfg.num_rows, cfg.num_cols), dtype=np.float32)
        self.flyball_pit_x1 = np.zeros((cfg.num_rows, cfg.num_cols), dtype=np.float32)
        self.flyball_forbidden_x0 = np.zeros((cfg.num_rows, cfg.num_cols), dtype=np.float32)
        self.flyball_forbidden_x1 = np.zeros((cfg.num_rows, cfg.num_cols), dtype=np.float32)
        # Per-env hurdle x-ranges in world coords, (rows, cols, slots, x0/x1). Slot
        # count max(2, num_goals); add_terrain_to_map pads unused slots with -1.
        self.n_hurdle_slots = max(2, int(cfg.num_goals))
        self.flyball_hurdle_x_ranges = np.full((cfg.num_rows, cfg.num_cols, self.n_hurdle_slots, 2), -1.0, dtype=np.float32)  # -1 sentinel = no hurdle (stays <0 after the world-x offset)
        self.flyball_wall_angle_deg = np.zeros((cfg.num_rows, cfg.num_cols), dtype=np.float32)
        self.flyball_wall_delta_z   = np.zeros((cfg.num_rows, cfg.num_cols), dtype=np.float32)
        self.flyball_liftoff_region = np.zeros((cfg.num_rows, cfg.num_cols, 2, 3), dtype=np.float32)
        self.flyball_wall_region    = np.zeros((cfg.num_rows, cfg.num_cols, 2, 3), dtype=np.float32)
        self.flyball_landing_region = np.zeros((cfg.num_rows, cfg.num_cols, 2, 3), dtype=np.float32)
        self.flyball_air_region     = np.zeros((cfg.num_rows, cfg.num_cols, 2, 3), dtype=np.float32)
        # Per-foot targets (4 feet x xyz) in local metres, FL,FR,RL,RR, interpolated
        # by wall angle. Reward-only — never enters the observation space.
        self.flyball_foot_target    = np.zeros((cfg.num_rows, cfg.num_cols, 4, 3), dtype=np.float32)
        # Front-foot LANDING footholds (FL, FR) for the P4L stage-2 floor landing.
        self.flyball_landing_foot_target = np.zeros((cfg.num_rows, cfg.num_cols, 2, 3), dtype=np.float32)
        # Per-hurdle WoCoCo AABBs (env_class 15 only); zero elsewhere.
        self.flyball_hurdle_liftoff_region = np.zeros((cfg.num_rows, cfg.num_cols, 2, 2, 3), dtype=np.float32)
        self.flyball_hurdle_air_region     = np.zeros((cfg.num_rows, cfg.num_cols, 2, 2, 3), dtype=np.float32)
        self.flyball_hurdle_landing_region = np.zeros((cfg.num_rows, cfg.num_cols, 2, 2, 3), dtype=np.float32)
        # Parabolic CLEARANCE bounds: x = floor roots, y = padded hurdle width,
        # z = floor to apex. Slot k = hurdle k, matching flyball_hurdle_x_ranges (NOT
        # the wococo forward/backward convention). All-zero rows never fire.
        self.flyball_hurdle_clear_region = np.zeros(
            (cfg.num_rows, cfg.num_cols, self.n_hurdle_slots, 2, 3), dtype=np.float32)
        self.num_goals = cfg.num_goals

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            if hasattr(cfg, "max_difficulty"):
                self.curiculum(random=True, max_difficulty=cfg.max_difficulty)
            else:
                self.curiculum(random=True)
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            print("Converting heightmap to trimesh...")
            if cfg.hf2mesh_method == "grid":
                self.vertices, self.triangles, self.x_edge_mask = convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                                self.cfg.horizontal_scale,
                                                                                                self.cfg.vertical_scale,
                                                                                                self.cfg.slope_treshold)
                if self.cfg.simplify_grid:
                    mesh_simplifier = pyfqmr.Simplify()
                    mesh_simplifier.setMesh(self.vertices, self.triangles)
                    mesh_simplifier.simplify_mesh(target_count = int(0.05*self.triangles.shape[0]), aggressiveness=7, preserve_border=True, verbose=10)

                    self.vertices, self.triangles, normals = mesh_simplifier.getMesh()
                    self.vertices = self.vertices.astype(np.float32)
                    self.triangles = self.triangles.astype(np.uint32)
            else:
                assert cfg.hf2mesh_method == "fast", "Height field to mesh method must be grid or fast"
                self.vertices, self.triangles = convert_heightfield_to_trimesh_delatin(self.height_field_raw, self.cfg.horizontal_scale, self.cfg.vertical_scale, max_error=cfg.max_error)
                # Delatin gives only geometry: rebuild the forward-x steep-edge mask.
                raw_threshold = (self.cfg.slope_treshold
                                 * self.cfg.horizontal_scale
                                 / self.cfg.vertical_scale)
                dx = self.height_field_raw[1:, :] - self.height_field_raw[:-1, :]
                move_x = np.zeros_like(self.height_field_raw, dtype=np.int8)
                move_x[:-1, :] += dx > raw_threshold
                move_x[1:, :] -= -dx > raw_threshold
                self.x_edge_mask = move_x != 0
            half_edge_width = int(self.cfg.edge_width_thresh / self.cfg.horizontal_scale)
            structure = np.ones((half_edge_width*2+1, 1))
            self.x_edge_mask = binary_dilation(self.x_edge_mask, structure=structure)
            print("Created {} vertices".format(self.vertices.shape[0]))
            print("Created {} triangles".format(self.triangles.shape[0]))

        if getattr(self.cfg, 'save_vts', False) and self.type == "trimesh":
            print("Saving [Terrain] heightfield as VTS...")

            import os
            import pyvista as pv
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_dir = getattr(self.cfg, 'log_dir', None) or os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs", "terrain"
            )
            path = os.path.join(log_dir, f"terrain_height_{timestamp}.vts")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            H = self.height_field_raw.astype(np.float32) * self.cfg.vertical_scale
            hs = self.cfg.horizontal_scale

            # Decimate the export grid: at horizontal_scale = 1 cm the map is tens of
            # millions of points, which ParaView cannot open. Each cell takes the MAX
            # of its block so thin geometry (7 cm hurdles, the plate lip) survives,
            # placed at the block centre to stay aligned with the goal VTPs.
            export_cell = float(getattr(self.cfg, 'vts_export_cell_size', 0.05))
            stride = max(1, int(round(export_cell / hs)))
            if stride > 1:
                nx, ny = H.shape
                pad_x, pad_y = (-nx) % stride, (-ny) % stride
                if pad_x or pad_y:
                    H = np.pad(H, ((0, pad_x), (0, pad_y)), mode='edge')
                bx, by = H.shape[0] // stride, H.shape[1] // stride
                H = H.reshape(bx, stride, by, stride).max(axis=(1, 3))
                centre_off = (stride - 1) * 0.5 * hs
                x = np.arange(bx) * stride * hs + centre_off
                y = np.arange(by) * stride * hs + centre_off
                print(f"[Terrain] VTS downsampled {nx}x{ny} -> {bx}x{by} "
                      f"({export_cell*100:.0f} cm cells, block max)")
            else:
                x = np.arange(H.shape[0]) * hs
                y = np.arange(H.shape[1]) * hs
            X, Y = np.meshgrid(x, y, indexing='ij')

            grid = pv.StructuredGrid(X, Y, H)

            grid["height"] = H.ravel(order="F")

            grid.save(path)

            print(f"[Terrain] Heightfield saved to {path}")
            goals_path = path.replace("terrain_height_", "terrain_goals_").replace(".vts", ".vtp")

            border_m = float(self.border) * hs  # heightfield grid includes this border
            pts = []
            goal_tile = []
            goal_k = []

            for i in range(self.cfg.num_rows):
                for j in range(self.cfg.num_cols):
                    g = self.goals[i, j]  # (num_goals, 3) in meters, tile/world coords (no border)
                    for k in range(g.shape[0]):
                        xw, yw, zw = float(g[k, 0]), float(g[k, 1]), float(g[k, 2])

                        xg = xw + border_m
                        yg = yw + border_m
                        zg = zw  # use stored z directly — explicit for wall goals, terrain height for floor goals

                        pts.append([xg, yg, zg])
                        goal_tile.append(i * self.cfg.num_cols + j)
                        goal_k.append(k)

            pts = np.asarray(pts, dtype=np.float32)

            goals_pd = pv.PolyData(pts)
            goals_pd["tile_id"] = np.asarray(goal_tile, dtype=np.int32)   # which env tile
            goals_pd["goal_k"]  = np.asarray(goal_k, dtype=np.int32)      # which goal inside the tile

            goals_pd.save(goals_path)
            print(f"[Terrain] Goals (shifted) saved to {goals_path}")

            clearance_path = path.replace(
                "terrain_height_", "terrain_hurdle_clearance_").replace(".vts", ".vtp")
            self._save_hurdle_clearance_surfaces(clearance_path, border_m)

            # Same tile/world-metre frame as self.goals (no border). Foot order
            # 0=FL, 1=FR, 2=RL, 3=RR; all-zero non-wall tiles are skipped.
            _foot_targets = getattr(self, 'flyball_foot_target', None)
            if _foot_targets is not None:
                foot_path = path.replace("terrain_height_", "terrain_footholds_").replace(".vts", ".vtp")
                _foot_names = ["FL", "FR", "RL", "RR"]

                fpts = []
                foot_tile = []
                foot_idx = []
                foot_name = []
                foot_landing = []   # 0 = wall foothold, 1 = P4L floor landing foothold

                for i in range(self.cfg.num_rows):
                    for j in range(self.cfg.num_cols):
                        ft = _foot_targets[i, j]  # (4, 3) tile/world metres
                        for f in range(ft.shape[0]):
                            xw, yw, zw = float(ft[f, 0]), float(ft[f, 1]), float(ft[f, 2])
                            if xw == 0.0 and yw == 0.0 and zw == 0.0:
                                continue

                            fpts.append([xw + border_m, yw + border_m, zw])
                            foot_tile.append(i * self.cfg.num_cols + j)
                            foot_idx.append(f)
                            foot_name.append(_foot_names[f] if f < len(_foot_names) else str(f))
                            foot_landing.append(0)

                # P4L floor touchdowns, tagged is_landing=1, foot_name LAND_FL/LAND_FR.
                _land_targets = getattr(self, 'flyball_landing_foot_target', None)
                if _land_targets is not None:
                    _land_names = ["FL", "FR"]
                    for i in range(self.cfg.num_rows):
                        for j in range(self.cfg.num_cols):
                            lt = _land_targets[i, j]  # (2, 3) tile/world metres
                            for f in range(lt.shape[0]):
                                xw, yw, zw = float(lt[f, 0]), float(lt[f, 1]), float(lt[f, 2])
                                if xw == 0.0 and yw == 0.0 and zw == 0.0:
                                    continue
                                fpts.append([xw + border_m, yw + border_m, zw])
                                foot_tile.append(i * self.cfg.num_cols + j)
                                foot_idx.append(f)   # 0=FL, 1=FR (front feet)
                                foot_name.append("LAND_" + (_land_names[f] if f < len(_land_names) else str(f)))
                                foot_landing.append(1)

                if len(fpts) > 0:
                    fpts = np.asarray(fpts, dtype=np.float32)
                    foot_pd = pv.PolyData(fpts)
                    foot_pd["tile_id"]    = np.asarray(foot_tile, dtype=np.int32)  # which env tile
                    foot_pd["foot_idx"]   = np.asarray(foot_idx, dtype=np.int32)   # 0=FL 1=FR 2=RL 3=RR
                    foot_pd["foot_name"]  = np.asarray(foot_name)                  # FL/FR/RL/RR or LAND_FL/LAND_FR
                    foot_pd["is_landing"] = np.asarray(foot_landing, dtype=np.int32)  # 0=wall, 1=floor landing
                    foot_pd.save(foot_path)
                    print(f"[Terrain] Foothold targets (shifted) saved to {foot_path}")

                else:
                    print("[Terrain] No non-zero foothold targets to save (no wall tiles).")

    def _save_hurdle_clearance_surfaces(self, out_path, border_m, samples=33):
        """Export only the extruded parabolic clearance volumes as VTP surfaces.

        The output is a triangulated, closed boundary (curved top, both lateral
        faces, and floor) sampled from the same root/floor/apex bounds used by
        the exact reward collision test. Other reward AABBs and foothold rings
        are intentionally excluded from the save_vts export.
        """
        import pyvista as pv

        bounds = getattr(self, 'flyball_hurdle_clear_region', None)
        if bounds is None:
            print("[Terrain] No parabolic hurdle-clearance volumes to save.")
            return

        points, faces = [], []
        tile_ids, row_ids, col_ids, slot_ids = [], [], [], []
        surface_ids, boundary_kinds = [], []

        def add_triangle(a, b, c, meta, boundary_kind):
            faces.append([3, a, b, c])
            tile_ids.append(meta[0])
            row_ids.append(meta[1])
            col_ids.append(meta[2])
            slot_ids.append(meta[3])
            surface_ids.append(meta[4])
            boundary_kinds.append(boundary_kind)

        def add_quad(a, b, c, d, meta, boundary_kind):
            add_triangle(a, b, c, meta, boundary_kind)
            add_triangle(a, c, d, meta, boundary_kind)

        next_surface = 0
        bounds = np.asarray(bounds, dtype=np.float32)
        for i in range(self.cfg.num_rows):
            for j in range(self.cfg.num_cols):
                tile = i * self.cfg.num_cols + j
                tile_bounds = bounds[i, j].reshape(-1, 2, 3)
                for slot, prism in enumerate(tile_bounds):
                    lo, hi = prism
                    if hi[0] <= lo[0] or hi[1] <= lo[1] or hi[2] <= lo[2]:
                        continue

                    xs = np.linspace(lo[0], hi[0], samples, dtype=np.float32)
                    center_x = 0.5 * (lo[0] + hi[0])
                    half_span = 0.5 * (hi[0] - lo[0])
                    zs = lo[2] + (hi[2] - lo[2]) * (
                        1.0 - ((xs - center_x) / half_span) ** 2)
                    base = len(points)
                    for x, z in zip(xs, zs):
                        points.extend((
                            [x + border_m, lo[1] + border_m, lo[2]],
                            [x + border_m, hi[1] + border_m, lo[2]],
                            [x + border_m, lo[1] + border_m, z],
                            [x + border_m, hi[1] + border_m, z],
                        ))

                    meta = (tile, i, j, slot, next_surface)
                    for k in range(samples - 1):
                        cur = base + 4 * k
                        nxt = cur + 4
                        add_quad(cur + 2, nxt + 2, nxt + 3, cur + 3,
                                 meta, 0)  # curved top
                        add_quad(cur, nxt, nxt + 2, cur + 2,
                                 meta, 1)  # y-low side
                        add_quad(cur + 1, cur + 3, nxt + 3, nxt + 1,
                                 meta, 2)  # y-high side
                        add_quad(cur, cur + 1, nxt + 1, nxt,
                                 meta, 3)  # floor
                    next_surface += 1

        if not faces:
            print("[Terrain] No parabolic hurdle-clearance volumes to save.")
            return

        pd = pv.PolyData(
            np.asarray(points, dtype=np.float32),
            faces=np.hstack(faces).astype(np.int32))
        pd.cell_data["tile_id"] = np.asarray(tile_ids, dtype=np.int32)
        pd.cell_data["row"] = np.asarray(row_ids, dtype=np.int32)
        pd.cell_data["col"] = np.asarray(col_ids, dtype=np.int32)
        pd.cell_data["slot"] = np.asarray(slot_ids, dtype=np.int32)
        pd.cell_data["surface_id"] = np.asarray(surface_ids, dtype=np.int32)
        pd.cell_data["boundary_kind"] = np.asarray(boundary_kinds, dtype=np.int32)
        pd.save(out_path)
        print(f"[Terrain] Parabolic hurdle clearance ({next_surface} extruded "
              f"surfaces, {len(faces)} triangles) saved to {out_path}")

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            print("choice: ", choice)
            difficulty = np.random.uniform(-0.2, 1.2)
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self, random=False, max_difficulty=False):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / (self.cfg.num_rows - 1)



                choice = j / self.cfg.num_cols + 0.001 
                
                
                
                if random:
                    if max_difficulty:
                        terrain = self.make_terrain(choice, np.random.uniform(0.7, 1), level=i)
                    else:
                        terrain = self.make_terrain(choice, np.random.uniform(0, 1), level=i)
                else:
                    terrain = self.make_terrain(choice, difficulty, level=i)

                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.length_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def add_roughness(self, terrain, difficulty=1):
        max_height = (self.cfg.height[1] - self.cfg.height[0]) * difficulty + self.cfg.height[0]
        height = random.uniform(self.cfg.height[0], max_height)
        terrain_utils.random_uniform_terrain(terrain, min_height=-height, max_height=height, step=0.005, downsampled_scale=self.cfg.downsampled_scale)

    def make_terrain(self, choice, difficulty, level=None):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.length_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.02 + 0.14 * difficulty
        discrete_obstacles_height = 0.03 + difficulty * 0.15
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        
        
        if choice < self.proportions[0]:
            idx = 0
            if choice < self.proportions[0]/ 2:
                idx = 1
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[2]:
            idx = 2
            if choice<self.proportions[1]:
                idx = 3
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            self.add_roughness(terrain)
        elif choice < self.proportions[4]:
            idx = 4
            if choice<self.proportions[3]:
                idx = 5
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
            self.add_roughness(terrain)
        elif choice < self.proportions[5]:
            idx = 6
            num_rectangles = 20
            rectangle_min_size = 0.5
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
            self.add_roughness(terrain)
        elif choice < self.proportions[6]:
            idx = 7
            stones_size = 1.5 - 1.2*difficulty
            half_sloped_terrain(terrain, wall_width=4, start2center=0.5, max_height=0.00)
            stepping_stones_terrain(terrain, stone_size=1.5-0.2*difficulty, stone_distance=0.0+0.4*difficulty, max_height=0.2*difficulty, platform_size=1.2)
            self.add_roughness(terrain)
        elif choice < self.proportions[7]:
            idx = 8
            gap_parkour_terrain(terrain, difficulty, platform_size=4)
            self.add_roughness(terrain)
        elif choice < self.proportions[8]:
            idx = 9
            # "smooth flat": ONLY roughness scales with difficulty; the robot tracks
            # evenly-spaced centerline goals down +x.
            self.add_roughness(terrain, difficulty=difficulty)
            _mid_y = terrain.length // 2
            _plat = round(2.5 / terrain.horizontal_scale)
            _ng = self.num_goals
            _gx = np.linspace(_plat, terrain.width - round(0.5 / terrain.horizontal_scale), _ng)
            _g = np.zeros((_ng, 2))
            _g[:, 0] = _gx
            _g[:, 1] = _mid_y
            terrain.goals = _g * terrain.horizontal_scale
        elif choice < self.proportions[9]:
            idx = 10
            x_range = [-0.1, 0.1 + 0.3 * difficulty]
            y_range = [0.2, 0.3 + 0.1 * difficulty]
            stone_len = [0.9 - 0.3 * difficulty, 1 - 0.2 * difficulty]
            incline_height = 0.25 * difficulty
            last_incline_height = incline_height + 0.1 - 0.1 * difficulty
            parkour_terrain(terrain,
                            num_stones=self.num_goals - 2,
                            x_range=x_range,
                            y_range=y_range,
                            incline_height=incline_height,
                            stone_len=stone_len,
                            last_incline_height=last_incline_height,
                            pad_height=0,
                            pit_depth=[0.2, 1])
            self.add_roughness(terrain)
        elif choice < self.proportions[10]:
            idx = 11
            if self.cfg.all_vertical:
                half_slope_difficulty = 1.0
            else:
                difficulty *= 1.3
                if not self.cfg.no_flat:
                    difficulty -= 0.1
                if difficulty > 1:
                    half_slope_difficulty = 1.0
                elif difficulty < 0:
                    self.add_roughness(terrain)
                    terrain.slope_vector = np.array([1, 0., 0]).astype(np.float32)
                    return terrain
                else:
                    half_slope_difficulty = difficulty
            wall_width = 4 - half_slope_difficulty * 4
            if self.cfg.flat_wall:
                half_sloped_terrain(terrain, wall_width=4, start2center=0.5, max_height=0.00)
            else:
                half_sloped_terrain(terrain, wall_width=wall_width, start2center=0.5, max_height=1.5)
            max_height = terrain.height_field_raw.max()
            top_mask = terrain.height_field_raw > max_height - 0.05
            self.add_roughness(terrain, difficulty=1)
            terrain.height_field_raw[top_mask] = max_height
        elif choice < self.proportions[11]:
            idx = 12
            half_platform_terrain(terrain, max_height=0.1 + 0.4 * difficulty )
            self.add_roughness(terrain, difficulty=1)
        elif choice < self.proportions[13]:
            idx = 13
            height = 0.1 + 0.3 * difficulty
            if choice < self.proportions[12]:
                idx = 14
                height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=1., step_height=height, platform_size=3.)
            self.add_roughness(terrain)
        elif choice < self.proportions[14]:
            idx = 16
            # Curriculum hurdle terrain (env_class 16): height scales 0 -> max with
            # difficulty, so level 0 is a flat corridor and the penalty self-disables.
            _hh_max = getattr(self.cfg, 'hurdle_height_max', 0.15)
            # Height floor: 0.0 = flat at level 0, 0.05 = 5 cm bars on the easiest row.
            _hh_min = getattr(self.cfg, 'hurdle_height_min', 0.0)
            # Thickness and yaw jitter scale with difficulty (level 0 = y-aligned).
            _len_base = getattr(self.cfg, 'hurdle_stone_len_base', 0.1)
            _len_max  = getattr(self.cfg, 'hurdle_thickness_max', 0.20)
            parkour_hurdle_terrain(terrain,
                                   num_stones=self.num_goals - 2,
                                   stone_len=_len_base + (_len_max - _len_base)*difficulty,
                                   hurdle_height_range=[_hh_min, _hh_min + (_hh_max - _hh_min)*difficulty],
                                   pad_height=0,
                                   x_range=getattr(self.cfg, 'hurdle_x_range', [1.2, 2.6]),
                                   y_range=self.cfg.y_range,
                                   half_valid_width=getattr(self.cfg, 'hurdle_half_valid_width', [0.4, 0.8]),
                                   yaw_jitter_max=getattr(self.cfg, 'hurdle_yaw_jitter_max', 0.0)*difficulty,
                                   num_hurdles_range=getattr(self.cfg, 'hurdle_count_range', None),
                                   record_x_ranges=getattr(self.cfg, 'hurdle_record_contact', False),
                                   hurdle_clear_pad_x=getattr(self.cfg, 'flyball_hurdle_clear_pad_x', 0.05),
                                   hurdle_clear_pad_y=getattr(self.cfg, 'flyball_hurdle_clear_pad_y', 0.05),
                                   hurdle_clear_pad_z=getattr(self.cfg, 'flyball_hurdle_clear_pad_z', 0.05),
                                   )
            self.add_roughness(terrain, difficulty=difficulty)
        elif choice < self.proportions[15]:
            parkour_two_hurdles_wall_turn_terrain(
                terrain,
                difficulty=difficulty,
                hurdle_thickness=self.cfg.flyball_hurdle_thickness,
                hurdle_width=self.cfg.flyball_hurdle_width,
                hurdle_height_range=self.cfg.flyball_hurdle_height_range,
                hurdle_gap_range=self.cfg.flyball_hurdle_gap_range,
                inter_hurdle_range=self.cfg.flyball_inter_hurdle_range,
                wall_angle_deg_range=self.cfg.flyball_wall_angle_deg_range,
                pit_width_range=self.cfg.flyball_pit_width_range,
                pit_height_offset_range=self.cfg.flyball_pit_height_offset_range,
                liftoff_x_range=self.cfg.flyball_liftoff_x_range,
                liftoff_y_mid=self.cfg.flyball_liftoff_y_mid,
                liftoff_y_half=self.cfg.flyball_liftoff_y_half,
                liftoff_z_above=self.cfg.flyball_liftoff_z_above,
                wall_y_half=self.cfg.flyball_wall_y_half,
                wall_z_min_above=self.cfg.flyball_wall_z_min_above,
                wall_x_depth_scale=self.cfg.flyball_wall_x_depth_scale,
                landing_x_range=self.cfg.flyball_landing_x_range,
                landing_y_mid=self.cfg.flyball_landing_y_mid,
                landing_y_half=self.cfg.flyball_landing_y_half,
                landing_z_above=self.cfg.flyball_landing_z_above,
                air_x_range=self.cfg.flyball_air_x_range,
                air_y_mid=self.cfg.flyball_air_y_mid,
                air_y_half=self.cfg.flyball_air_y_half,
                air_z_min_above=self.cfg.flyball_air_z_min_above,
                air_z_max_above=self.cfg.flyball_air_z_max_above,
                hurdle_liftoff_back=getattr(self.cfg, 'flyball_hurdle_liftoff_back', 0.6),
                hurdle_liftoff_gap=getattr(self.cfg, 'flyball_hurdle_liftoff_gap', 0.20),
                hurdle_landing_forward=getattr(self.cfg, 'flyball_hurdle_landing_forward', 0.6),
                hurdle_landing_gap=getattr(self.cfg, 'flyball_hurdle_landing_gap', 0.0),
                hurdle_aabb_y_half=getattr(self.cfg, 'flyball_hurdle_aabb_y_half', 0.5),
                hurdle_aabb_z_above=getattr(self.cfg, 'flyball_hurdle_aabb_z_above', 0.10),
                hurdle_air_z_above=getattr(self.cfg, 'flyball_hurdle_air_z_above', 0.30),
                hurdle_clear_pad_x=getattr(self.cfg, 'flyball_hurdle_clear_pad_x', 0.05),
                hurdle_clear_pad_y=getattr(self.cfg, 'flyball_hurdle_clear_pad_y', 0.05),
                hurdle_clear_pad_z=getattr(self.cfg, 'flyball_hurdle_clear_pad_z', 0.05),
                # At level <= easy_lateral_max_level, the g3↔g5 lateral_offset is
                # multiplied by easy_lateral_scale.
                level=level,
                easy_lateral_scale=getattr(self.cfg, 'flyball_easy_lateral_scale', 1.0),
                easy_lateral_max_level=getattr(self.cfg, 'flyball_easy_lateral_max_level', -1),
            )
            idx = 15
        elif choice < self.proportions[16]:
            idx = 17
            parkour_hurdle_terrain(terrain,
                                   num_stones=self.num_goals - 2,
                                   stone_len=0.1+0.3*difficulty,
                                   hurdle_height_range=[0.1+0.1*difficulty, 0.15+0.15*difficulty],
                                   pad_height=0,
                                   y_range=self.cfg.y_range,
                                   half_valid_width=[0.45, 1],
                                   flat=True
                                   )
            # Optional full-width step scales from flat at level zero.
            _step_h = getattr(self.cfg, 'flat_step_height_max', 0.0) * difficulty
            if _step_h > 0:
                _sh_raw = int(round(_step_h / terrain.vertical_scale))
                _sx = terrain.height_field_raw.shape[0] // 2   # mid-corridor (axis-0 = x)
                terrain.height_field_raw[_sx:, :] += _sh_raw
            _hurdle_xr = np.full((2, 2), -1.0, dtype=np.float32)   # sentinels (no hurdle)
            _hurdle_fp = [None, None]   # per-slot footprint for clearance prisms
            _hurdle_h = getattr(self.cfg, 'flat_corridor_hurdle_height_max', 0.0) * difficulty
            # Add one or two corridor-spanning bars and publish their contact ranges.
            _hurdle_cfg_n = int(getattr(self.cfg, 'flat_corridor_hurdle_count', 0))
            if _hurdle_h > 0 and _hurdle_cfg_n > 0:
                _hh_raw = int(round(_hurdle_h / terrain.vertical_scale))
                _nx_h = terrain.height_field_raw.shape[0]
                _plat_h = int(round(2.5 / terrain.horizontal_scale))   # spawn platform_len
                _thk_h = max(1, int(round(getattr(self.cfg, 'flat_corridor_hurdle_thickness', 0.18)
                                          / terrain.horizontal_scale)))
                _xr_frac = getattr(self.cfg, 'flat_corridor_hurdle_x_range', [0.3, 0.7])
                _n_here = 2 if (_hurdle_cfg_n >= 2 and np.random.rand() < 0.5) else 1
                _used_f = []
                for _hk in range(_n_here):
                    _f = float(np.random.uniform(_xr_frac[0], _xr_frac[1]))
                    for _ in range(20):
                        if all(abs(_f - _u) >= 0.15 for _u in _used_f):
                            break
                        _f = float(np.random.uniform(_xr_frac[0], _xr_frac[1]))
                    _used_f.append(_f)
                    _hxc = int(_plat_h + _f * max(1, _nx_h - _plat_h))
                    _hxa = max(0, _hxc - _thk_h // 2)
                    _hxb = min(_nx_h, _hxc + (_thk_h - _thk_h // 2))
                    terrain.height_field_raw[_hxa:_hxb, :] += _hh_raw
                    _hurdle_xr[_hk] = [float(_hxa) * terrain.horizontal_scale,
                                       float(_hxb) * terrain.horizontal_scale]
                    if _hurdle_h > _HURDLE_CLEAR_MIN_H:
                        _hurdle_fp[_hk] = (_hxa, _hxb, 0,
                                           terrain.height_field_raw.shape[1],
                                           float(_hurdle_h))
            terrain.flyball_hurdle_x_ranges_world = _hurdle_xr
            terrain.flyball_hurdle_clear_region = _hurdle_clear_boxes(
                _hurdle_fp, terrain.horizontal_scale, 0.0,
                getattr(self.cfg, 'flyball_hurdle_clear_pad_x', 0.05),
                getattr(self.cfg, 'flyball_hurdle_clear_pad_y', 0.05),
                getattr(self.cfg, 'flyball_hurdle_clear_pad_z', 0.05),
                x_max_m=(terrain.width - 1) * terrain.horizontal_scale)
            _cw_range = getattr(self.cfg, 'flat_corridor_width_range', None)
            if _cw_range is not None:
                _cw = np.random.uniform(_cw_range[0], _cw_range[1])
                _half_px = int(round((_cw / 2.0) / terrain.horizontal_scale))
                _wall_raw = int(round(getattr(self.cfg, 'flat_corridor_wall_height', 0.8)
                                      / terrain.vertical_scale))
                _ny = terrain.height_field_raw.shape[1]        # lateral (y) pixels
                _nx = terrain.height_field_raw.shape[0]        # forward (x) pixels
                _my = _ny // 2                                 # straight centerline (axis-1 = y)
                # Triangle-wave centerlines create sharp turns; width is measured normal to the path.
                _amp_px = (getattr(self.cfg, 'flat_corridor_curve_amplitude', 0.0) * difficulty
                           / terrain.horizontal_scale)
                _plat_px = int(round(2.5 / terrain.horizontal_scale))   # spawn platform_len
                _L = max(1, _nx - _plat_px)                             # curving region length (px)
                # bends == switchback corners == interior goals, capped at num_goals-2.
                _n_goals = terrain.goals.shape[0] if getattr(terrain, 'goals', None) is not None else self.num_goals
                _bends = max(1, min(int(getattr(self.cfg, 'flat_corridor_curve_bends', 6)), _n_goals - 2))
                # Clamp the swing so the WIDENED band never punches through the cell
                # wall at a corner: >=1 px of wall remains on each side.
                _amp_px = min(_amp_px, max(0.0, float(_my - _half_px - 1)))
                if _amp_px > 0:
                    _slope_est = 2.0 * _amp_px * _bends / _L            # |dcy/dx| on a ramp (px/px)
                    _amp_px = min(_amp_px, max(0.0, _my - _half_px * float(np.sqrt(1.0 + _slope_est ** 2)) - 1.0))
                if _amp_px > 0:
                    _dir = float(np.random.choice([-1.0, 1.0]))         # swing left- or right-first
                    _xs = np.arange(_nx)
                    _t = np.clip((_xs - _plat_px) / _L, 0.0, 1.0)
                    _tri = (2.0 / np.pi) * np.arcsin(np.sin(np.pi * _bends * _t))
                    _cy = _my + _dir * _amp_px * _tri
                    # Half-width pinched along x by a difficulty-scaled cosine,
                    # floored so the gap stays walkable.
                    _half_arr = np.full(_nx, float(_half_px))
                    _pinch = getattr(self.cfg, 'flat_corridor_pinch_amount', 0.0) * difficulty
                    if _pinch > 0:
                        _pc = int(getattr(self.cfg, 'flat_corridor_pinch_count', 2))
                        _ph = np.random.uniform(0.0, 2.0 * np.pi)
                        _prof = 1.0 - _pinch * (0.5 + 0.5 * np.cos(2.0 * np.pi * _pc * _t + _ph))
                        _min_half = (getattr(self.cfg, 'flat_corridor_min_width', 0.8) / 2.0
                                     / terrain.horizontal_scale)
                        _half_arr = np.maximum(_half_px * _prof, _min_half)
                    _half_arr[_xs < _plat_px] = _half_px                # never pinch the spawn platform
                    # Widen by sqrt(1+slope^2) to keep perpendicular width on ramps.
                    _dl = np.abs(_cy - np.roll(_cy, 1))
                    _dr = np.abs(np.roll(_cy, -1) - _cy)
                    _slope = np.maximum(_dl, _dr)
                    _slope[0] = _slope[-1] = 0.0
                    _horiz = _half_arr * np.sqrt(1.0 + _slope ** 2)
                    _lo = np.clip(np.round(_cy - _horiz).astype(int), 0, _ny)
                    _hi = np.clip(np.round(_cy + _horiz).astype(int), 0, _ny)
                    _ys = np.arange(_ny)[None, :]
                    _wall_mask = (_ys < _lo[:, None]) | (_ys >= _hi[:, None])
                    terrain.height_field_raw[_wall_mask] = _wall_raw
                    # Goals pinned to the switchback corners (+ start + end) so the
                    # chord between consecutive goals never crosses a wall.
                    _gx = [float(_plat_px - 1)]
                    _gy = [float(_my)]
                    for _k in range(_bends):
                        _gx.append(_plat_px + ((_k + 0.5) / _bends) * _L)
                        _sgn = 1.0 if (_k % 2 == 0) else -1.0           # tri peak alternates +/-1
                        _gy.append(_my + _dir * _amp_px * _sgn)
                    _gx.append(float(_nx - 1))
                    _gy.append(float(_my))
                    _verts = np.stack([np.asarray(_gx), np.asarray(_gy)], axis=1)
                    _goals_px = np.repeat(_verts[-1:], _n_goals, axis=0).astype(float)  # pad tail with end
                    _m = min(_verts.shape[0], _n_goals)
                    _goals_px[:_m] = _verts[:_m]
                    terrain.goals = _goals_px * terrain.horizontal_scale
                else:
                    # Straight corridor: hurdle goals already sit on the centerline.
                    _lo = max(0, _my - _half_px)
                    _hi = _my + _half_px
                    terrain.height_field_raw[:, :_lo] = _wall_raw
                    terrain.height_field_raw[:, _hi:] = _wall_raw
            self.add_roughness(terrain)
            # Applied *after* roughness so the noise doesn't bury the landmark.
            # parkour_flat goals all sit on mid_y, so it proxies the goal path.
            if getattr(self.cfg, 'centerline_strip', False):
                _add_centerline_strip(
                    terrain,
                    width=self.cfg.centerline_strip_width,
                    height=self.cfg.centerline_strip_height,
                    clear_width=getattr(self.cfg, 'centerline_strip_clear_width', None),
                )
        elif choice < self.proportions[17]:
            idx = 18
            parkour_step_terrain(terrain,
                                   num_stones=self.num_goals - 2,
                                   step_height=0.1 + 0.35*difficulty,
                                   x_range=[0.3,1.5],
                                   y_range=self.cfg.y_range,
                                   half_valid_width=[0.5, 1],
                                   pad_height=0,
                                   )
            self.add_roughness(terrain)
        elif choice < self.proportions[18]:
            idx = 19
            flyball_wall_terrain(
                terrain,
                difficulty=difficulty,
                wall_width=getattr(self.cfg, 'flyball_plate_width', 1.5),
                plate_length=getattr(self.cfg, 'flyball_plate_length', None),
                backing_wall_height=getattr(self.cfg, 'flyball_backing_wall_height', 2.0),
                backing_wall_thickness=getattr(self.cfg, 'flyball_backing_wall_thickness', 0.35),
                wall_angle_deg_range=self.cfg.flyball_wall_angle_deg_range,
                wall_angle_jitter_deg=getattr(self.cfg, 'flyball_wall_angle_jitter_deg', 0.0),
                pit_width_range=self.cfg.flyball_pit_width_range,
                pit_height_offset_range=self.cfg.flyball_pit_height_offset_range,
                pit_depth=self.cfg.flyball_pit_depth,
                roughness_height=self.cfg.height,
                roughness_downsampled_scale=self.cfg.downsampled_scale,
                roughness_zero_prob=getattr(self.cfg, 'flyball_roughness_zero_prob', 0.0),
                liftoff_x_range=self.cfg.flyball_liftoff_x_range,
                liftoff_y_mid=self.cfg.flyball_liftoff_y_mid,
                liftoff_y_half=self.cfg.flyball_liftoff_y_half,
                liftoff_z_above=self.cfg.flyball_liftoff_z_above,
                wall_y_half=self.cfg.flyball_wall_y_half,
                wall_z_min_above=self.cfg.flyball_wall_z_min_above,
                wall_x_depth_scale=self.cfg.flyball_wall_x_depth_scale,
                landing_x_range=self.cfg.flyball_landing_x_range,
                landing_y_mid=self.cfg.flyball_landing_y_mid,
                landing_y_half=self.cfg.flyball_landing_y_half,
                landing_z_above=self.cfg.flyball_landing_z_above,
                landing_foot_standoff=getattr(self.cfg, 'flyball_landing_foot_standoff', 0.35),
                landing_foot_y_mid=getattr(self.cfg, 'flyball_landing_foot_y_mid', 0.15),
                landing_foot_stance_half=getattr(self.cfg, 'flyball_landing_foot_stance_half', 0.10),
                landing_foot_x_stagger=getattr(self.cfg, 'flyball_landing_foot_x_stagger', 0.10),
                air_x_range=self.cfg.flyball_air_x_range,
                air_y_mid=self.cfg.flyball_air_y_mid,
                air_y_half=self.cfg.flyball_air_y_half,
                air_z_min_above=self.cfg.flyball_air_z_min_above,
                air_z_max_above=self.cfg.flyball_air_z_max_above,
                runup_dist_range=getattr(self.cfg, 'flyball_runup_dist_range', (4.0, 5.5)),
                runup_jitter=getattr(self.cfg, 'flyball_runup_jitter', 0.5),
                lateral_offset=getattr(self.cfg, 'flyball_lateral_offset', 0.0),
                landing_back_extra=getattr(self.cfg, 'flyball_landing_back_extra', 1.5),
                landing_centerline=getattr(self.cfg, 'flyball_landing_centerline', False),
                randomize_return_goal_x=getattr(self.cfg, 'flyball_randomize_return_goal_x', False),
                add_hurdle=getattr(self.cfg, 'flyball_add_hurdle', False),
                hurdle_height_range=getattr(self.cfg, 'flyball_single_hurdle_height_range', (0.05, 0.20)),
                curriculum_level=level,
                curriculum_max_level=self.cfg.num_rows - 1,
                hurdle_zero_below_level=getattr(self.cfg, 'flyball_single_hurdle_zero_below_level', 0),
                hurdle_height_jitter_m=getattr(self.cfg, 'flyball_hurdle_height_jitter_m', 0.0),
                hurdle_height_decorrelate=getattr(self.cfg, 'flyball_hurdle_height_decorrelate', False),
                hurdle_zero_prob=getattr(self.cfg, 'flyball_hurdle_zero_prob', 0.0),
                hurdle_thickness=getattr(self.cfg, 'flyball_single_hurdle_thickness', 0.18),
                hurdle_thickness_range=getattr(self.cfg, 'flyball_single_hurdle_thickness_range', None),
                hurdle_thickness_range_uniform=getattr(
                    self.cfg, 'flyball_single_hurdle_thickness_range_uniform', False),
                hurdle_thickness_jitter_m=getattr(self.cfg, 'flyball_hurdle_thickness_jitter_m', 0.0),
                hurdle_height_max_m=getattr(self.cfg, 'flyball_single_hurdle_height_max', None),
                hurdle_thickness_max_m=getattr(self.cfg, 'flyball_single_hurdle_thickness_max', None),
                hurdle_width=getattr(self.cfg, 'flyball_single_hurdle_width', 2.0),
                hurdle_width_range=getattr(self.cfg, 'flyball_single_hurdle_width_range', None),
                hurdle_roughness_height=getattr(
                    self.cfg, 'flyball_hurdle_roughness_height', (0.0, 0.0)),
                hurdle_offset_from_platform=getattr(self.cfg, 'flyball_single_hurdle_offset', 0.8),
                hurdle_offset_jitter=getattr(self.cfg, 'flyball_single_hurdle_offset_jitter', 0.3),
                post_hurdle_goal_distance=getattr(self.cfg, 'flyball_post_hurdle_goal_distance', 0.6),
                add_second_hurdle=getattr(self.cfg, 'flyball_add_second_hurdle', False),
                second_hurdle_offset=getattr(self.cfg, 'flyball_second_hurdle_offset', 2.0),
                second_hurdle_offset_jitter=getattr(self.cfg, 'flyball_second_hurdle_offset_jitter', 0.3),
                spawn_x=getattr(self.cfg, 'flyball_spawn_x', 1.0),
                hurdle_front_from_spawn=getattr(self.cfg, 'flyball_hurdle_front_from_spawn', None),
                hurdle_front_from_spawn_jitter=getattr(
                    self.cfg, 'flyball_hurdle_front_from_spawn_jitter', 0.0),
                wall_to_hurdle_rear=getattr(self.cfg, 'flyball_wall_to_hurdle_rear', None),
                wall_to_hurdle_rear_jitter=getattr(
                    self.cfg, 'flyball_wall_to_hurdle_rear_jitter', 0.0),
                wall_to_hurdle_rear_min=getattr(
                    self.cfg, 'flyball_wall_to_hurdle_rear_min', None),
                hurdle_jitter_scale_with_difficulty=getattr(
                    self.cfg, 'flyball_hurdle_jitter_scale_with_difficulty', False),
                hurdle_liftoff_back=getattr(self.cfg, 'flyball_hurdle_liftoff_back', 0.6),
                hurdle_liftoff_gap=getattr(self.cfg, 'flyball_hurdle_liftoff_gap', 0.20),
                hurdle_landing_forward=getattr(self.cfg, 'flyball_hurdle_landing_forward', 0.6),
                hurdle_landing_gap=getattr(self.cfg, 'flyball_hurdle_landing_gap', 0.0),
                hurdle_aabb_y_half=getattr(self.cfg, 'flyball_hurdle_aabb_y_half', 0.5),
                hurdle_aabb_z_above=getattr(self.cfg, 'flyball_hurdle_aabb_z_above', 0.10),
                hurdle_air_z_above=getattr(self.cfg, 'flyball_hurdle_air_z_above', 0.30),
                hurdle_air_x_extra=getattr(self.cfg, 'flyball_hurdle_air_x_extra', 0.0),
                hurdle_air_x_extra_landing=getattr(self.cfg, 'flyball_hurdle_air_x_extra_landing', 0.15),
                hurdle_air_z_min_above=getattr(self.cfg, 'flyball_hurdle_air_z_min_above', 0.10),
                hurdle_clear_pad_x=getattr(self.cfg, 'flyball_hurdle_clear_pad_x', 0.05),
                hurdle_clear_pad_y=getattr(self.cfg, 'flyball_hurdle_clear_pad_y', 0.05),
                hurdle_clear_pad_z=getattr(self.cfg, 'flyball_hurdle_clear_pad_z', 0.05),
                hurdle_clear_box_height=getattr(
                    self.cfg, 'flyball_hurdle_clear_box_height',
                    getattr(self.cfg, 'flyball_hurdle_clear_height', 0.30)),
                hurdle_clear_half_span=getattr(
                    self.cfg, 'flyball_hurdle_clear_half_span', 0.15),
                build_wall=getattr(self.cfg, 'flyball_build_wall', True),
                end_after_hurdles=getattr(self.cfg, 'flyball_end_after_hurdles', False),
                # Pair with flyball_wall_goal_idx=1 / flyball_return_goal_idx=2.
                swimmer_goal_layout=getattr(self.cfg, 'flyball_swimmer_goal_layout', False),
                # Interpolated by wall angle; analytic fallback when out of range.
                foothold_table_path=getattr(self.cfg, 'flyball_foothold_table_path', None),
                foothold_z_offset_m=getattr(self.cfg, 'flyball_foothold_z_offset_m', 0.0),
                rear_foothold_positive_y_shift_m=getattr(
                    self.cfg, 'flyball_rear_foothold_positive_y_shift_m', 0.0),
                rear_foothold_shift_min_angle_deg=getattr(
                    self.cfg, 'flyball_rear_foothold_shift_min_angle_deg', 50.0),
            )
            # roughness applied internally to approach area only — do NOT call add_roughness
        else:
            idx = 20
            demo_terrain(terrain)
            self.add_roughness(terrain)
        terrain.idx = idx
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = i * self.env_length + 1.0
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 0.5) / terrain.horizontal_scale) # within 1 meter square range
        x2 = int((self.env_length/2. + 0.5) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 0.5) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 0.5) / terrain.horizontal_scale)
        if self.cfg.origin_zero_z:
            env_origin_z = 0
        else:
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        self.terrain_type[i, j] = terrain.idx
        pit_x0_local = getattr(terrain, 'flyball_pit_x0_world', 0.0)
        self.flyball_pit_x0[i, j] = pit_x0_local + i * self.env_length
        pit_x1_local = getattr(terrain, 'flyball_pit_x1_world', 0.0)
        self.flyball_pit_x1[i, j] = pit_x1_local + i * self.env_length
        fz_x0_local = getattr(terrain, 'flyball_forbidden_x0_world', 0.0)
        self.flyball_forbidden_x0[i, j] = fz_x0_local + i * self.env_length
        fz_x1_local = getattr(terrain, 'flyball_forbidden_x1_world', 0.0)
        self.flyball_forbidden_x1[i, j] = fz_x1_local + i * self.env_length
        self.flyball_wall_angle_deg[i, j] = getattr(terrain, 'flyball_wall_angle_deg', 0.0)
        self.flyball_wall_delta_z[i, j]   = getattr(terrain, 'wall_delta_z', 0.0)
        hurdle_xr_local = getattr(terrain, 'flyball_hurdle_x_ranges_world', None)
        if hurdle_xr_local is not None:
            # -1 sentinels must stay negative so real_slot holds for rows i>0.
            _xr = np.asarray(hurdle_xr_local, dtype=np.float32).copy()
            if _xr.ndim == 1:
                _xr = _xr.reshape(1, 2)
            _real = _xr[:, 0] >= 0.0
            _xr[_real] += i * self.env_length
            # Writers may emit fewer slots; trailing ones keep their -1 sentinel.
            _k = min(_xr.shape[0], self.n_hurdle_slots)
            self.flyball_hurdle_x_ranges[i, j, :_k] = _xr[:_k]
        for _attr in ('flyball_liftoff_region', 'flyball_wall_region', 'flyball_landing_region', 'flyball_air_region', 'flyball_landing_foot_target'):
            _r = getattr(terrain, _attr, np.zeros((2, 3), dtype=np.float32)).copy()
            _r[:, 0] += i * self.env_length
            _r[:, 1] += j * self.env_width
            getattr(self, _attr)[i, j] = _r
        # Per-foot WoCoCo targets (4,3): same world-frame offset as the AABBs.
        _ft = getattr(terrain, 'flyball_foot_target', np.zeros((4, 3), dtype=np.float32)).copy()
        _ft[:, 0] += i * self.env_length
        _ft[:, 1] += j * self.env_width
        self.flyball_foot_target[i, j] = _ft
        # Per-hurdle AABBs (2 hurdles, 2 endpoints, 3 xyz), same world offset as above.
        for _attr in ('flyball_hurdle_liftoff_region', 'flyball_hurdle_air_region', 'flyball_hurdle_landing_region'):
            _r = getattr(terrain, _attr, np.zeros((2, 2, 3), dtype=np.float32)).copy()
            _r[..., 0] += i * self.env_length
            _r[..., 1] += j * self.env_width
            getattr(self, _attr)[i, j] = _r
        # Offset only the slots the writer filled: an all-zero sentinel row must stay
        # zero-volume, else it becomes a degenerate box at the tile origin.
        _cr = getattr(terrain, 'flyball_hurdle_clear_region', None)
        if _cr is not None:
            _cr = np.asarray(_cr, dtype=np.float32).copy()
            if _cr.ndim == 2:                       # single hurdle emitted as (2, 3)
                _cr = _cr.reshape(1, 2, 3)
            _real = _cr[:, 1, 0] > _cr[:, 0, 0]     # x_hi > x_lo -> a real prism
            _cr[_real, :, 0] += i * self.env_length
            _cr[_real, :, 1] += j * self.env_width
            _k = min(_cr.shape[0], self.n_hurdle_slots)
            self.flyball_hurdle_clear_region[i, j, :_k] = _cr[:_k]
        tg = getattr(terrain, 'goals', np.zeros((self.num_goals, 2), dtype=np.float32))  # (N, 2) or (N, 3) in local metres
        self.goals[i, j, :, :2] = tg[:, :2] + [i * self.env_length, j * self.env_width]
        # An explicit z (col 2) is used directly; otherwise sample the height field.
        has_explicit_z = tg.shape[1] == 3 and np.any(tg[:, 2] != 0.0)
        for g in range(tg.shape[0]):
            if has_explicit_z and tg[g, 2] != 0.0:
                self.goals[i, j, g, 2] = tg[g, 2]
            else:
                world_x, world_y = self.goals[i, j, g, 0], self.goals[i, j, g, 1]
                px = int(np.clip((world_x + self.cfg.border_size) / self.cfg.horizontal_scale, 0, self.height_field_raw.shape[0] - 1))
                py = int(np.clip((world_y + self.cfg.border_size) / self.cfg.horizontal_scale, 0, self.height_field_raw.shape[1] - 1))
                self.goals[i, j, g, 2] = self.height_field_raw[px, py] * self.cfg.vertical_scale

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def gap_parkour_terrain(terrain, difficulty, platform_size=2.):
    gap_width_m = 0.1 + 0.6 * difficulty   # gap width: 0.1m (easy) → 0.7m (hard)
    gap_px = max(1, int(gap_width_m / terrain.horizontal_scale))

    mid_y = terrain.length // 2          # lateral center (pixels)
    cx = terrain.width // 2              # forward center (pixels)

    # single transverse gap across full lateral width, centered in forward direction
    x0 = cx - gap_px // 2
    x1 = x0 + gap_px
    terrain.height_field_raw[:, x0:x1] = -400

    half_gap_px = gap_px // 2
    offset_px = max(1, round(0.5 / terrain.horizontal_scale))  # ~0.5m offset from gap edge

    # goals in pixels [forward_px, lateral_px], then multiplied by horizontal_scale → metres
    goals = np.zeros((8, 2), dtype=np.float32)
    goals[0] = [cx - half_gap_px - offset_px, mid_y]  # before gap
    goals[1] = [cx + half_gap_px + offset_px, mid_y]  # after gap
    goals[2:] = goals[1]
    terrain.goals = goals * terrain.horizontal_scale

def parkour_terrain(terrain, 
                    platform_len=2.5, 
                    platform_height=0., 
                    num_stones=8, 
                    x_range=[1.8, 1.9], 
                    y_range=[0., 0.1], 
                    z_range=[-0.2, 0.2],
                    stone_len=1.0,
                    stone_width=0.6,
                    pad_width=0.1,
                    pad_height=0.5,
                    incline_height=0.1,
                    last_incline_height=0.6,
                    last_stone_len=1.6,
                    pit_depth=[0.5, 1.]):
    # 1st dimension: x, 2nd dimension: y
    goals = np.zeros((num_stones+2, 2))
    terrain.height_field_raw[:] = -round(np.random.uniform(pit_depth[0], pit_depth[1]) / terrain.vertical_scale)
    
    mid_y = terrain.length // 2  # length is actually y width
    stone_len = np.random.uniform(*stone_len)
    stone_len = 2 * round(stone_len / 2.0, 1)
    stone_len = round(stone_len / terrain.horizontal_scale)
    dis_x_min = stone_len + round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = stone_len + round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)
    dis_z_min = round(z_range[0] / terrain.vertical_scale)
    dis_z_max = round(z_range[1] / terrain.vertical_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    stone_width = round(stone_width / terrain.horizontal_scale)
    last_stone_len = round(last_stone_len / terrain.horizontal_scale)

    incline_height = round(incline_height / terrain.vertical_scale)
    last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len - np.random.randint(dis_x_min, dis_x_max) + stone_len // 2
    goals[0] = [platform_len -  stone_len // 2, mid_y]
    left_right_flag = np.random.randint(0, 2)
    dis_z = 0
    
    for i in range(num_stones):
        dis_x += np.random.randint(dis_x_min, dis_x_max)
        pos_neg = round(2*(left_right_flag - 0.5))
        dis_y = mid_y + pos_neg * np.random.randint(dis_y_min, dis_y_max)
        if i == num_stones - 1:
            dis_x += last_stone_len // 4
            heights = np.tile(np.linspace(-last_incline_height, last_incline_height, stone_width), (last_stone_len, 1)) * pos_neg
            terrain.height_field_raw[dis_x-last_stone_len//2:dis_x+last_stone_len//2, dis_y-stone_width//2: dis_y+stone_width//2] = heights.astype(int) + dis_z
        else:
            heights = np.tile(np.linspace(-incline_height, incline_height, stone_width), (stone_len, 1)) * pos_neg
            terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, dis_y-stone_width//2: dis_y+stone_width//2] = heights.astype(int) + dis_z
        
        goals[i+1] = [dis_x, dis_y]

        left_right_flag = 1 - left_right_flag
    final_dis_x = dis_x + 2*np.random.randint(dis_x_min, dis_x_max)
    final_platform_start = dis_x + last_stone_len // 2 + round(0.05 // terrain.horizontal_scale)
    terrain.height_field_raw[final_platform_start:, :] = platform_height
    goals[-1] = [final_dis_x, mid_y]
    
    terrain.goals = goals * terrain.horizontal_scale
    
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height
    
def parkour_gap_terrain(terrain,
                           platform_len=2.5, 
                           platform_height=0., 
                           num_gaps=8,
                           gap_size=0.3,
                           x_range=[1.6, 2.4],
                           y_range=[-1.2, 1.2],
                           half_valid_width=[0.6, 1.2],
                           gap_depth=-200,
                           pad_width=0.1,
                           pad_height=0.5,
                           flat=False):
    goals = np.zeros((num_gaps+2, 2))
    mid_y = terrain.length // 2  # length is actually y width

    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    gap_depth = -round(np.random.uniform(gap_depth[0], gap_depth[1]) / terrain.vertical_scale)
    
    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    
    terrain.height_field_raw[0:platform_len, :] = platform_height

    gap_size = round(gap_size / terrain.horizontal_scale)
    dis_x_min = round(x_range[0] / terrain.horizontal_scale) + gap_size
    dis_x_max = round(x_range[1] / terrain.horizontal_scale) + gap_size

    dis_x = platform_len
    goals[0] = [platform_len - 1, mid_y]
    last_dis_x = dis_x
    for i in range(num_gaps):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        dis_x += rand_x
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        if not flat:
            terrain.height_field_raw[dis_x-gap_size//2 : dis_x+gap_size//2, :] = gap_depth

        terrain.height_field_raw[last_dis_x:dis_x, :mid_y+rand_y-half_valid_width] = gap_depth
        terrain.height_field_raw[last_dis_x:dis_x, mid_y+rand_y+half_valid_width:] = gap_depth
        
        last_dis_x = dis_x
        goals[i+1] = [dis_x-rand_x//2, mid_y + rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]
    
    terrain.goals = goals * terrain.horizontal_scale
    
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height

def parkour_hurdle_terrain(terrain,
                           platform_len=2.5,
                           platform_height=0.,
                           num_stones=8,
                           stone_len=0.3,
                           x_range=[1.5, 2.4],
                           y_range=[-0.4, 0.4],
                           half_valid_width=[0.4, 0.8],
                           hurdle_height_range=[0.2, 0.3],
                           pad_width=0.1,
                           pad_height=0.5,
                           flat=False,
                           yaw_jitter_max=0.0,
                           num_hurdles_range=None,
                           record_x_ranges=False,
                           hurdle_clear_pad_x=0.05,
                           hurdle_clear_pad_y=0.05,
                           hurdle_clear_pad_z=0.05):
    """Forward corridor with raised hurdle bars the robot must clear.

    Curriculum knobs (all backward-compatible — defaults reproduce the
    original behaviour):
      hurdle_height_range : pass [0, h_max*difficulty] so level 0 has
          ZERO-height bars (i.e. flat) and the bars grow with the curriculum.
          When the max height rounds to <= the min, NO bars are drawn and no
          contact x-ranges are recorded, so the hurdle-contact penalty
          self-disables at level 0.
      yaw_jitter_max : per-bar rotation (rad) about world z, so each bar is
          tilted off pure world-y alignment by a random angle in
          [-yaw_jitter_max, +yaw_jitter_max]. Hardens the policy against a
          single perpendicular-bar layout. 0 -> axis-aligned (fast path).
      num_hurdles_range : (min, max) count of bars actually drawn this env
          (clamped to num_stones). A random subset of the num_stones goal
          slots get a bar; the rest are bar-free forward waypoints. None ->
          every slot gets a bar (original behaviour). Combined with the
          per-slot random x-spacing this varies both the count AND the
          inter-hurdle distance per env.
      record_x_ranges : record each drawn bar's world x-band (bounding the
          tilt) into terrain.flyball_hurdle_x_ranges_world so the
          env-driven _reward_hurdle_contact penalty fires on these envs.
          Also emits bounds for the per-bar parabolic clearance volumes that
          drive _reward_hurdle_clearance.
    """
    goals = np.zeros((num_stones+2, 2))

    mid_y = terrain.length // 2  # length is actually y width

    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    hurdle_height_max = round(hurdle_height_range[1] / terrain.vertical_scale)
    hurdle_height_min = round(hurdle_height_range[0] / terrain.vertical_scale)
    # With an upper bound <= 0 no bars are drawn, so the tile is flat and the contact
    # penalty has no x-ranges; a non-zero floor keeps bars enabled at level 0.
    bars_enabled = (not flat) and (hurdle_height_max > 0)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    stone_len = round(stone_len / terrain.horizontal_scale)
    half_len = max(1, stone_len // 2)
    nx = terrain.height_field_raw.shape[0]
    tan_max = np.tan(yaw_jitter_max) if yaw_jitter_max > 0 else 0.0

    # Which of the num_stones slots get a bar (varies count and spacing).
    if bars_enabled and num_hurdles_range is not None:
        n_lo = max(0, int(num_hurdles_range[0]))
        n_hi = min(int(num_hurdles_range[1]), num_stones)
        n_hi = max(n_hi, n_lo)
        n_here = int(np.random.randint(n_lo, n_hi + 1))
        bar_slots = set(np.random.choice(num_stones, size=min(n_here, num_stones), replace=False).tolist()) if n_here > 0 else set()
    else:
        bar_slots = set(range(num_stones)) if bars_enabled else set()

    # World x-bands (metres) of the bars drawn this env, for the contact
    # penalty. -1 sentinel rows = empty slots (never match the in-range test).
    hurdle_xr = np.full((num_stones, 2), -1.0, dtype=np.float32)
    # Per-bar footprint in hurdle_xr slot order; x-bounds include the tilt padding.
    hurdle_fp = [None] * num_stones

    # Phase 1: resolve goal lateral positions BEFORE carving, so the straightening
    # pass runs first and bars land on the final goal line.
    goal_x = np.zeros(num_stones + 2)
    goal_y = np.zeros(num_stones + 2)
    goal_x[0], goal_y[0] = platform_len - 1, mid_y
    bar_x      = [0] * num_stones    # bar centre x (pixels) for slot i
    bar_slot_i = [i in bar_slots for i in range(num_stones)]
    dis_x = platform_len
    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        # parkour_flat has nothing to steer around: keep goals on the centerline.
        rand_y = 0 if flat else np.random.randint(dis_y_min, dis_y_max)
        dis_x += rand_x
        bar_x[i] = dis_x
        goal_x[i+1] = dis_x - rand_x // 2
        goal_y[i+1] = mid_y + rand_y
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goal_x[-1], goal_y[-1] = final_dis_x, mid_y

    # Phase 2: the segment between goal k and k+1 (k >= 1) carries a hurdle iff
    # bar_slot_i[k-1]; the start segment (g0,g1) never does. Each maximal hurdle-free
    # run of >= 3 goals snaps to a single lateral line (the run's first y).
    if not flat:
        seg_hurdle = [False] * (num_stones + 1)       # seg k between goal k, k+1
        for k in range(1, num_stones + 1):
            seg_hurdle[k] = bar_slot_i[k - 1]
        runs = []
        run_start = 0
        for k in range(num_stones + 1):               # k = segment index
            if seg_hurdle[k]:
                runs.append((run_start, k))            # run ends at goal k (hurdle past it)
                run_start = k + 1                      # next run starts past the hurdle
        runs.append((run_start, num_stones + 1))       # final run reaches the last goal
        for a, b in runs:
            if b - a + 1 >= 3:                         # >= 3 collinear goals -> snap to one line
                goal_y[a:b + 1] = goal_y[a]

    # Phase 3: carve the bars on the final goal line, record x-bands.
    for i in range(num_stones):
        if not bar_slot_i[i]:
            continue
        h = np.random.randint(hurdle_height_min, hurdle_height_max) if hurdle_height_max > hurdle_height_min else hurdle_height_max
        cx = int(bar_x[i])
        yc = int(round(goal_y[i+1]))                  # bar follows its approach goal's (straightened) y
        y_lo = max(0, yc - half_valid_width)
        y_hi = yc + half_valid_width
        slope = float(np.tan(np.random.uniform(-yaw_jitter_max, yaw_jitter_max))) if yaw_jitter_max > 0 else 0.0
        if slope == 0.0:
            terrain.height_field_raw[cx-half_len:cx+half_len, y_lo:y_hi] = h
        else:
            # Tilted bar: shift each lateral column's x-slice by slope*(y-yc).
            for y in range(y_lo, y_hi):
                xc = int(round(cx + slope * (y - yc)))
                terrain.height_field_raw[max(0, xc-half_len):min(nx, xc+half_len), y] = h
        # Bounding world x-band of the (possibly tilted) bar, for the penalty.
        x_pad = abs(slope) * half_valid_width
        hurdle_xr[i] = [float(cx - half_len - x_pad) * terrain.horizontal_scale,
                        float(cx + half_len + x_pad) * terrain.horizontal_scale]
        # Bars stand on the tile's zero baseline, so the clearance floor z = 0.
        if h * terrain.vertical_scale > _HURDLE_CLEAR_MIN_H:
            hurdle_fp[i] = (cx - half_len - x_pad, cx + half_len + x_pad,
                            y_lo, y_hi, float(h) * terrain.vertical_scale)

    goals[:, 0] = goal_x
    goals[:, 1] = goal_y
    terrain.goals = goals * terrain.horizontal_scale
    if record_x_ranges:
        terrain.flyball_hurdle_x_ranges_world = hurdle_xr
        terrain.flyball_hurdle_clear_region = _hurdle_clear_boxes(
            hurdle_fp, terrain.horizontal_scale, 0.0,
            hurdle_clear_pad_x, hurdle_clear_pad_y, hurdle_clear_pad_z,
            x_max_m=(terrain.width - 1) * terrain.horizontal_scale)

    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height

def _add_centerline_strip(terrain, width=0.05, height=0.05, clear_width=None):
    """Raise a thin rectangular prism along the corridor centerline.

    Spans the full x-extent of the tile (length-wise) at y = mid_y, with
    the requested lateral `width` and `height` (both in metres). The
    height-field cells in the strip's footprint — and optionally a
    surrounding `clear_width`-wide band — are *zeroed* before the strip
    is written, so the strip sits on a clean baseline instead of inheriting
    the roughness pattern. Defaults clear_width = width (only the strip
    footprint is cleared); pass a larger value to also flatten a margin
    around it.
    """
    mid_y = terrain.length // 2  # 'length' is the y (lateral) extent
    w_cells = max(1, int(round(width / terrain.horizontal_scale)))
    h_units = int(round(height / terrain.vertical_scale))
    if clear_width is None:
        clear_width = width
    clear_cells = max(w_cells, int(round(clear_width / terrain.horizontal_scale)))
    clear_half = clear_cells // 2
    cy_lo = mid_y - clear_half
    cy_hi = cy_lo + clear_cells
    # Wipe roughness under the strip so the marker sits on a flat baseline.
    terrain.height_field_raw[:, cy_lo:cy_hi] = 0
    half = w_cells // 2
    y_lo = mid_y - half
    y_hi = y_lo + w_cells
    terrain.height_field_raw[:, y_lo:y_hi] = h_units

def parkour_step_terrain(terrain,
                           platform_len=2.5, 
                           platform_height=0., 
                           num_stones=8,
                            x_range=[0.2, 0.4],
                           y_range=[-0.15, 0.15],
                           half_valid_width=[0.45, 0.5],
                           step_height = 0.2,
                           pad_width=0.1,
                           pad_height=0.5):
    goals = np.zeros((num_stones+2, 2))
    mid_y = terrain.length // 2  # length is actually y width

    dis_x_min = round( (x_range[0] + step_height) / terrain.horizontal_scale)
    dis_x_max = round( (x_range[1] + step_height) / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    step_height = round(step_height / terrain.vertical_scale)

    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    

    dis_x = platform_len
    last_dis_x = dis_x
    stair_height = 0
    goals[0] = [platform_len - round(1 / terrain.horizontal_scale), mid_y]
    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        if i < num_stones // 2:
            stair_height += step_height
        elif i > num_stones // 2:
            stair_height -= step_height
        terrain.height_field_raw[dis_x:dis_x+rand_x, ] = stair_height
        dis_x += rand_x
        terrain.height_field_raw[last_dis_x:dis_x, :mid_y+rand_y-half_valid_width] = 0
        terrain.height_field_raw[last_dis_x:dis_x, mid_y+rand_y+half_valid_width:] = 0
        
        last_dis_x = dis_x
        goals[i+1] = [dis_x-rand_x//2, mid_y+rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]
    
    terrain.goals = goals * terrain.horizontal_scale
    
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height

def demo_terrain(terrain):
    goals = np.zeros((8, 2))
    mid_y = terrain.length // 2
    
    platform_length = round(2 / terrain.horizontal_scale)
    hurdle_depth = round(np.random.uniform(0.35, 0.4) / terrain.horizontal_scale)
    hurdle_height = round(np.random.uniform(0.3, 0.36) / terrain.vertical_scale)
    hurdle_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[0] = [platform_length + hurdle_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+hurdle_depth, round(mid_y-hurdle_width/2):round(mid_y+hurdle_width/2)] = hurdle_height
    
    platform_length += round(np.random.uniform(1.5, 2.5) / terrain.horizontal_scale)
    first_step_depth = round(np.random.uniform(0.45, 0.8) / terrain.horizontal_scale)
    first_step_height = round(np.random.uniform(0.35, 0.45) / terrain.vertical_scale)
    first_step_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[1] = [platform_length+first_step_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+first_step_depth, round(mid_y-first_step_width/2):round(mid_y+first_step_width/2)] = first_step_height
    
    platform_length += first_step_depth
    second_step_depth = round(np.random.uniform(0.45, 0.8) / terrain.horizontal_scale)
    second_step_height = first_step_height
    second_step_width = first_step_width
    goals[2] = [platform_length+second_step_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+second_step_depth, round(mid_y-second_step_width/2):round(mid_y+second_step_width/2)] = second_step_height
    
    platform_length += second_step_depth
    gap_size = round(np.random.uniform(0.5, 0.8) / terrain.horizontal_scale)
    
    platform_length += gap_size
    third_step_depth = round(np.random.uniform(0.25, 0.6) / terrain.horizontal_scale)
    third_step_height = first_step_height
    third_step_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[3] = [platform_length+third_step_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+third_step_depth, round(mid_y-third_step_width/2):round(mid_y+third_step_width/2)] = third_step_height
    
    platform_length += third_step_depth
    forth_step_depth = round(np.random.uniform(0.25, 0.6) / terrain.horizontal_scale)
    forth_step_height = first_step_height
    forth_step_width = third_step_width
    goals[4] = [platform_length+forth_step_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+forth_step_depth, round(mid_y-forth_step_width/2):round(mid_y+forth_step_width/2)] = forth_step_height
    
    platform_length += forth_step_depth
    gap_size = round(np.random.uniform(0.1, 0.4) / terrain.horizontal_scale)
    platform_length += gap_size
    
    left_y = mid_y + round(np.random.uniform(0.15, 0.3) / terrain.horizontal_scale)
    right_y = mid_y - round(np.random.uniform(0.15, 0.3) / terrain.horizontal_scale)
    
    slope_height = round(np.random.uniform(0.15, 0.22) / terrain.vertical_scale)
    slope_depth = round(np.random.uniform(0.75, 0.85) / terrain.horizontal_scale)
    slope_width = round(1.0 / terrain.horizontal_scale)
    
    platform_height = slope_height + np.random.randint(0, 0.2 / terrain.vertical_scale)

    goals[5] = [platform_length+slope_depth/2, left_y]
    heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * 1
    terrain.height_field_raw[platform_length:platform_length+slope_depth, left_y-slope_width//2: left_y+slope_width//2] = heights.astype(int) + platform_height
    
    platform_length += slope_depth + gap_size
    goals[6] = [platform_length+slope_depth/2, right_y]
    heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * -1
    terrain.height_field_raw[platform_length:platform_length+slope_depth, right_y-slope_width//2: right_y+slope_width//2] = heights.astype(int) + platform_height
    
    platform_length += slope_depth + gap_size + round(0.4 / terrain.horizontal_scale)
    goals[-1] = [platform_length, left_y]
    terrain.goals = goals * terrain.horizontal_scale

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

def half_sloped_terrain(terrain, wall_width=4, start2center=0.7, max_height=1):
    wall_width_int = max(int(wall_width / terrain.horizontal_scale), 1)
    max_height_int = int(max_height / terrain.vertical_scale)
    slope_start = int(start2center / terrain.horizontal_scale + terrain.length // 2)
    terrain_length = terrain.length
    height2width_ratio = max_height_int / wall_width_int
    xs = np.arange(slope_start, terrain_length)
    heights = (height2width_ratio * (xs - slope_start)).clip(max=max_height_int).astype(np.int16)
    terrain.height_field_raw[slope_start:terrain_length, :] = heights[:, None]
    terrain.slope_vector = np.array([wall_width_int*terrain.horizontal_scale, 0., max_height]).astype(np.float32)
    terrain.slope_vector /= np.linalg.norm(terrain.slope_vector)
    print(terrain.slope_vector, wall_width)
    import matplotlib.pyplot as plt
    plt.imsave('test.png', terrain.height_field_raw, cmap='gray')

def half_platform_terrain(terrain, start2center=2, max_height=1):
    max_height_int = int(max_height / terrain.vertical_scale)
    slope_start = int(start2center / terrain.horizontal_scale + terrain.length // 2)
    terrain_length = terrain.length
    terrain.height_field_raw[:, :] = max_height_int
    terrain.height_field_raw[-slope_start:slope_start, -slope_start:slope_start] = 0

def stepping_stones_terrain(terrain, stone_size, stone_distance, max_height, platform_size=1., depth=-1):
    """
    Generate a stepping stones terrain

    Parameters:
        terrain (terrain): the terrain
        stone_size (float): horizontal size of the stepping stones [meters]
        stone_distance (float): distance between stones (i.e size of the holes) [meters]
        max_height (float): maximum height of the stones (positive and negative) [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
        depth (float): depth of the holes (default=-10.) [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    def get_rand_dis_int(scale):
        return np.random.randint(int(- scale / terrain.horizontal_scale + 1), int(scale / terrain.horizontal_scale))
    stone_size = int(stone_size / terrain.horizontal_scale)
    stone_distance = int(stone_distance / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    height_range = np.arange(-max_height-1, max_height, step=1)

    start_x = 0
    start_y = 0
    terrain.height_field_raw[:, :] = int(depth / terrain.vertical_scale)
    if terrain.length >= terrain.width:
        while start_y < terrain.length:
            stop_y = min(terrain.length, start_y + stone_size)
            start_x = np.random.randint(0, stone_size)
            stop_x = max(0, start_x - stone_distance - get_rand_dis_int(0.2))
            terrain.height_field_raw[0: stop_x, start_y: stop_y] = np.random.choice(height_range)
            while start_x < terrain.width:
                stop_x = min(terrain.width, start_x + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_x += stone_size + stone_distance + get_rand_dis_int(0.2)
            start_y += stone_size + stone_distance + get_rand_dis_int(0.2)
    elif terrain.width > terrain.length:
        while start_x < terrain.width:
            stop_x = min(terrain.width, start_x + stone_size)
            start_y = np.random.randint(0, stone_size)
            stop_y = max(0, start_y - stone_distance)
            terrain.height_field_raw[start_x: stop_x, 0: stop_y] = np.random.choice(height_range)
            while start_y < terrain.length:
                stop_y = min(terrain.length, start_y + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_y += stone_size + stone_distance
            start_x += stone_size + stone_distance

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain


def parkour_two_hurdles_wall_turn_terrain(
    terrain,
    difficulty: float,
    platform_len: float = 2.5,
    platform_height: float = 0.0,
    hurdle_thickness: float = 0.18,
    hurdle_width: float = 2.0,
    hurdle_height_range: tuple = (0.10, 0.30),
    hurdle_gap_range: tuple = (1.1, 2.4),
    inter_hurdle_range: tuple = (1.5, 2.8),
    # ---- wall (mirrors flyball_wall_terrain: no yaw, no y-offset, no lip) ----
    wall_height: float = 1.0,
    wall_width: float = 3.0,
    wall_thickness: float = 0.35,
    wall_angle_deg_range: tuple = (50.0, 90.0),
    wall_goal_height: float = 0.26,
    pit_width_range: tuple = (0.10, 0.30),
    pit_height_offset_range: tuple = (0.05, 0.15),
    runup_dist_range: tuple = (3.0, 4.5),
    runup_jitter: float = 0.5,
    lateral_offset: float = 0.4,
    landing_extra_lateral_offset: float = -0.10,  # g_land_y = yC + lateral_offset + this. -0.10 shifts the landing waypoint 15 cm toward centre relative to the prior +0.05 default.
    roughness_height: tuple = (0.02, 0.06),
    roughness_downsampled_scale: float = 0.075,
    corridor_extra_clearance: float = 0.25,
    # ---- WoCoCo AABB regions (relative to wall_front_x and corridor centre) ----
    liftoff_x_range: tuple = (-0.9500, -0.7500),
    liftoff_y_mid: float   = -0.5600,
    liftoff_y_half: float  =  0.5300,
    liftoff_z_above: float =  0.0500,
    wall_y_half: float     =  0.8200,
    wall_z_min_above: float =  0.1600,
    wall_x_depth_scale: float = 2.0,
    landing_x_range: tuple = (-0.9600, -0.4700),
    landing_y_mid: float   =  0.6500,
    landing_y_half: float  =  0.6300,
    landing_z_above: float =  0.0500,
    air_x_range: tuple     = (-0.8500, -0.1000),
    air_y_mid: float       =  0.6500,
    air_y_half: float      =  1.0000,
    air_z_min_above: float =  0.1000,
    air_z_max_above: float =  0.8000,
    hurdle_liftoff_back: float    = 0.60,
    hurdle_liftoff_gap: float     = 0.20,
    hurdle_landing_forward: float = 0.60,
    hurdle_landing_gap: float     = 0.0,
    hurdle_aabb_y_half: float     = 0.50,
    hurdle_aabb_z_above: float    = 0.10,
    hurdle_air_z_above: float     = 0.30,
    hurdle_clear_pad_x: float     = 0.05,
    hurdle_clear_pad_y: float     = 0.05,
    hurdle_clear_pad_z: float     = 0.05,
    level: int                    = None,
    easy_lateral_scale: float     = 1.0,
    easy_lateral_max_level: int   = -1,
):
    """
    start platform → hurdle1 → hurdle2 → run-up → slanted wall.

    The wall portion mirrors flyball_wall_terrain exactly: same ramp construction (no yaw,
    no y-offset, no lip), delta_z elevated base, analytic goal-on-surface placement, and
    the same metadata + AABB regions stored on the terrain object so wococo gates can be
    enabled in Phase 2 without touching this file.

    Goals (stored as (8, 3) in local metres, z explicit):
      0) before hurdle 1, center
      1) between hurdles, center
      2) just after hurdle 2 (run-up entry), center
      3) wp_takeoff: far run-up, yC − lateral_offset (mirrors flyball g1)
      4) wp_contact: on wall front surface at delta_z + wall_goal_height (analytic)
      5) wp_land: far run-up x, yC + lateral_offset + landing_extra_lateral_offset
      6) between hurdles — return path (center)
      7) before hurdle 1 — return to start (center)
    """
    hs = terrain.horizontal_scale
    vs = terrain.vertical_scale

    def m2px(m): return int(round(m / hs))
    def m2h(m):  return int(round(m / vs))
    def clamp(v, lo, hi): return max(lo, min(hi, v))
    def lerp(a, b, t): return a + (b - a) * t

    d = clamp(float(difficulty), 0.0, 1.0)

    delta_x = lerp(pit_width_range[0], pit_width_range[1], d)
    delta_z = lerp(pit_height_offset_range[0], pit_height_offset_range[1], d)

    base_h_i = m2h(platform_height)
    terrain.height_field_raw[:] = base_h_i
    mid_y = terrain.length // 2
    p_len = clamp(m2px(platform_len), 1, terrain.width)

    # ---------- approach roughness (applied globally; wall area cleared later) ----------
    rough_h = random.uniform(roughness_height[0], roughness_height[1])
    terrain_utils.random_uniform_terrain(
        terrain,
        min_height=-rough_h, max_height=rough_h,
        step=0.005, downsampled_scale=roughness_downsampled_scale,
    )

    half_corr = max(1, m2px(hurdle_width / 2.0))
    y1a = clamp(mid_y - half_corr, 0, terrain.length - 1)
    y1b = clamp(mid_y + half_corr, 0, terrain.length)
    half_clear = max(half_corr, m2px(hurdle_width / 2.0 + corridor_extra_clearance))
    yc0 = clamp(mid_y - half_clear, 0, terrain.length - 1)
    yc1 = clamp(mid_y + half_clear, 0, terrain.length)

    h_easy, h_hard = hurdle_height_range
    hurdle_h1_m = lerp(h_easy, h_hard, d)
    hurdle_h2_m = lerp(h_easy, h_hard, d)

    jitter_h = 0.02 * d
    hurdle_h1_m = max(0.0, hurdle_h1_m + np.random.uniform(-jitter_h, jitter_h))
    hurdle_h2_m = max(0.0, hurdle_h2_m + np.random.uniform(-jitter_h, jitter_h))

    hurdle_h1_i = m2h(hurdle_h1_m)
    hurdle_h2_i = m2h(hurdle_h2_m)

    g_easy, g_hard = hurdle_gap_range
    gap1_m = lerp(g_easy, g_hard, d)
    gap_j = 0.02 * g_hard * d
    gap1_m = max(0.2, gap1_m + np.random.uniform(-gap_j, gap_j))
    gap2_m = float(np.random.uniform(inter_hurdle_range[0], inter_hurdle_range[1]))

    gap1_px = max(1, m2px(gap1_m))
    gap2_px = max(1, m2px(gap2_m))
    hurdle_thk_px = max(1, m2px(hurdle_thickness))

    x = p_len + gap1_px
    x1a = clamp(x - hurdle_thk_px // 2, 0, terrain.width - 1)
    x1b = clamp(x + (hurdle_thk_px - hurdle_thk_px // 2), 1, terrain.width)
    terrain.height_field_raw[x1a:x1b, yc0:yc1] = base_h_i
    terrain.height_field_raw[x1a:x1b, y1a:y1b] = hurdle_h1_i

    x = x + gap2_px
    x2a = clamp(x - hurdle_thk_px // 2, 0, terrain.width - 1)
    x2b = clamp(x + (hurdle_thk_px - hurdle_thk_px // 2), 1, terrain.width)
    terrain.height_field_raw[x2a:x2b, yc0:yc1] = base_h_i
    terrain.height_field_raw[x2a:x2b, y1a:y1b] = hurdle_h2_i

    r_easy, r_hard = runup_dist_range
    runup_m = lerp(r_easy, r_hard, d)
    runup_m = max(0.4, runup_m + np.random.uniform(-runup_jitter, runup_jitter) * d)
    runup_px = max(1, m2px(runup_m))

    theta_deg = lerp(wall_angle_deg_range[0], wall_angle_deg_range[1], d)
    theta = np.deg2rad(theta_deg)
    tan_t = max(np.tan(theta), 1e-6)

    wall_bottom_len_m = min(wall_height / tan_t, 6.0)
    dx_px = max(2, m2px(wall_bottom_len_m))
    depth_px = max(2, m2px(wall_thickness))

    wall_half_w = max(1, m2px(wall_width / 2.0))
    wall_x0 = clamp(x2b + runup_px, 0, terrain.width - dx_px - 2)
    wall_front_px = wall_x0 - (depth_px - 1)

    terrain.height_field_raw[wall_front_px:, :] = base_h_i

    wall_base_h = base_h_i + int(round(delta_z / vs))
    wall_top_h = wall_base_h + int(round(wall_height / vs))
    ramp = np.linspace(float(wall_base_h), float(wall_top_h), dx_px, endpoint=True)
    XI, YI = np.meshgrid(np.arange(terrain.width), np.arange(terrain.length), indexing='ij')
    for depth in range(depth_px):
        u = XI - wall_x0 + depth
        in_wall = (u >= 0) & (u < dx_px) & (np.abs(YI - mid_y) <= wall_half_w)
        u_c = np.clip(u, 0, dx_px - 1 - 1e-9).astype(np.float64)
        u_lo = np.floor(u_c).astype(np.int32)
        u_hi = np.minimum(u_lo + 1, dx_px - 1)
        h_interp = ramp[u_lo] * (1.0 - (u_c - u_lo)) + ramp[u_hi] * (u_c - u_lo)
        h_int = np.round(h_interp).astype(np.int16)
        mask = in_wall & (h_int > terrain.height_field_raw)
        terrain.height_field_raw[mask] = h_int[mask]

    goals = np.zeros((8, 3), dtype=np.float32)
    yC_m = mid_y * hs
    base_z_m = float(platform_height)
    x_max_m = (terrain.width - 1) * hs

    # 0: before hurdle 1
    g0_x = clamp((x1a - m2px(0.6)) * hs, 0.0, x_max_m)
    goals[0] = [g0_x, yC_m, base_z_m]

    # 1: between hurdles (midpoint of hurdle centres)
    h1c_px = 0.5 * (x1a + x1b)
    h2c_px = 0.5 * (x2a + x2b)
    g1_x = clamp(0.5 * (h1c_px + h2c_px) * hs, 0.0, x_max_m)
    goals[1] = [g1_x, yC_m, base_z_m]

    # 2: just after hurdle 2 (run-up entry)
    g2_x = clamp((x2b + m2px(0.8)) * hs, 0.0, x_max_m)
    goals[2] = [g2_x, yC_m, base_z_m]

    # 3: far run-up just before the wall footprint, at yC − lateral_offset. On easy
    # levels lateral_offset is scaled so the g3↔g5 pair sits wider apart.
    eff_lateral_offset = lateral_offset
    if level is not None and easy_lateral_max_level >= 0 and level <= easy_lateral_max_level:
        eff_lateral_offset = lateral_offset * float(easy_lateral_scale)
    pit_width_px = max(1, m2px(delta_x))
    g3_x = clamp((wall_front_px - pit_width_px - m2px(1.0)) * hs, 0.0, x_max_m)
    g3_y = clamp(yC_m - eff_lateral_offset, 0.0, (terrain.length - 1) * hs)
    goals[3] = [g3_x, g3_y, base_z_m]

    # 4: wall contact — analytic, on the front surface at delta_z + wall_goal_height.
    wall_front_x = wall_front_px * hs
    contact_x = clamp(wall_front_x + wall_goal_height / tan_t, 0.0, x_max_m)
    goals[4] = [contact_x, yC_m, delta_z + wall_goal_height]

    # 5: landing goal — same x as g3 takeoff, opposite lateral side.
    g5_y = clamp(yC_m + eff_lateral_offset + landing_extra_lateral_offset, 0.0, (terrain.length - 1) * hs)
    goals[5] = [g3_x, g5_y, base_z_m]

    # 6–7: return path through between-hurdles → start (centered).
    goals[6] = goals[1]
    goals[7] = goals[0]

    terrain.goals = goals  # already in local metres; no pixel→metre conversion

    terrain.flyball_pit_x0_world = float(wall_front_x)
    terrain.flyball_pit_x1_world = float(wall_front_x)
    terrain.flyball_forbidden_x0_world = float((wall_front_px - pit_width_px) * hs)
    terrain.flyball_forbidden_x1_world = float(wall_front_x)
    terrain.flyball_wall_angle_deg = float(theta_deg)
    terrain.wall_theta_deg = float(theta_deg)
    terrain.wall_goal_height = float(wall_goal_height)
    terrain.wall_delta_z = float(delta_z)
    terrain.wall_delta_x = float(delta_x)
    terrain.runup_m = float(runup_m)
    terrain.hurdle_heights = (float(hurdle_h1_m), float(hurdle_h2_m))
    # Local-frame x-ranges (metres) that _reward_hurdle_contact penalises.
    terrain.flyball_hurdle_x_ranges_world = np.array([
        [x1a * hs, x1b * hs],
        [x2a * hs, x2b * hs],
    ], dtype=np.float32)

    terrain.flyball_liftoff_region = np.array([
        [clamp(wall_front_x + liftoff_x_range[0], 0.0, x_max_m),
         yC_m + liftoff_y_mid - liftoff_y_half,
         base_z_m - 0.1],
        [clamp(wall_front_x + liftoff_x_range[1], 0.0, x_max_m),
         yC_m + liftoff_y_mid + liftoff_y_half,
         base_z_m + liftoff_z_above],
    ], dtype=np.float32)

    wall_region_z_min = delta_z + wall_z_min_above
    wall_region_x_min = clamp(wall_front_x + wall_z_min_above / tan_t, 0.0, x_max_m)
    wall_top_x = clamp(wall_front_x + wall_height / tan_t, 0.0, x_max_m)
    wall_region_x_max = clamp(
        wall_region_x_min + wall_x_depth_scale * (wall_top_x - wall_region_x_min), 0.0, x_max_m)
    terrain.flyball_wall_region = np.array([
        [wall_region_x_min, yC_m - wall_y_half, wall_region_z_min],
        [wall_region_x_max, yC_m + wall_y_half, delta_z + wall_height],
    ], dtype=np.float32)

    terrain.flyball_landing_region = np.array([
        [clamp(wall_front_x + landing_x_range[0], 0.0, x_max_m),
         yC_m + landing_y_mid - landing_y_half,
         base_z_m - 0.1],
        [clamp(wall_front_x + landing_x_range[1], 0.0, x_max_m),
         yC_m + landing_y_mid + landing_y_half,
         base_z_m + landing_z_above],
    ], dtype=np.float32)

    terrain.flyball_air_region = np.array([
        [clamp(wall_front_x + air_x_range[0], 0.0, x_max_m),
         yC_m + air_y_mid - air_y_half,
         base_z_m + air_z_min_above],
        [clamp(wall_front_x + air_x_range[1], 0.0, x_max_m),
         yC_m + air_y_mid + air_y_half,
         base_z_m + air_z_max_above],
    ], dtype=np.float32)

    # Per hurdle: liftoff = rear-feet stance zone before the front face, air =
    # clearance above the footprint, landing = front-feet zone past the back face.
    h_x_pairs_px = [(x1a, x1b), (x2a, x2b)]
    h_heights_m  = [float(hurdle_h1_m), float(hurdle_h2_m)]
    hurdle_liftoff = np.zeros((2, 2, 3), dtype=np.float32)
    hurdle_air     = np.zeros((2, 2, 3), dtype=np.float32)
    hurdle_landing = np.zeros((2, 2, 3), dtype=np.float32)
    y_min = clamp(yC_m - hurdle_aabb_y_half, 0.0, (terrain.length - 1) * hs)
    y_max = clamp(yC_m + hurdle_aabb_y_half, 0.0, (terrain.length - 1) * hs)
    for h, ((xa_px, xb_px), h_top_m) in enumerate(zip(h_x_pairs_px, h_heights_m)):
        x_front_m = float(xa_px) * hs
        x_back_m  = float(xb_px) * hs
        # Far edge sits `hurdle_liftoff_gap` m back from the front face, so feet
        # stancing right at the hurdle don't satisfy the gate.
        hurdle_liftoff[h, 0] = [clamp(x_front_m - hurdle_liftoff_gap - hurdle_liftoff_back, 0.0, x_max_m),
                                y_min, base_z_m - 0.1]
        hurdle_liftoff[h, 1] = [clamp(x_front_m - hurdle_liftoff_gap, 0.0, x_max_m),
                                y_max, base_z_m + hurdle_aabb_z_above]
        hurdle_air[h, 0]     = [clamp(x_front_m, 0.0, x_max_m),
                                y_min, base_z_m + h_top_m]
        hurdle_air[h, 1]     = [clamp(x_back_m, 0.0, x_max_m),
                                y_max, base_z_m + h_top_m + hurdle_air_z_above]
        # hurdle_landing_gap shifts the AABB past the back face; 0.0 = flush with it.
        hurdle_landing[h, 0] = [clamp(x_back_m + hurdle_landing_gap, 0.0, x_max_m),
                                y_min, base_z_m - 0.1]
        hurdle_landing[h, 1] = [clamp(x_back_m + hurdle_landing_gap + hurdle_landing_forward, 0.0, x_max_m),
                                y_max, base_z_m + hurdle_aabb_z_above]
    terrain.flyball_hurdle_liftoff_region = hurdle_liftoff
    terrain.flyball_hurdle_air_region     = hurdle_air
    terrain.flyball_hurdle_landing_region = hurdle_landing

    # Slot order matches flyball_hurdle_x_ranges_world above (slot 0 = hurdle 1).
    terrain.flyball_hurdle_clear_region = _hurdle_clear_boxes(
        [(x1a, x1b, y1a, y1b, hurdle_h1_m) if hurdle_h1_m > _HURDLE_CLEAR_MIN_H else None,
         (x2a, x2b, y1a, y1b, hurdle_h2_m) if hurdle_h2_m > _HURDLE_CLEAR_MIN_H else None],
        hs, base_z_m,
        hurdle_clear_pad_x, hurdle_clear_pad_y, hurdle_clear_pad_z,
        x_max_m=x_max_m)

    return terrain



def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols-1)*horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows-1)*horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:

        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[:num_rows-1, :] += (hf[1:num_rows, :] - hf[:num_rows-1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[:num_rows-1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, :num_cols-1] += (hf[:, 1:num_cols] - hf[:, :num_cols-1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, :num_cols-1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[:num_rows-1, :num_cols-1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows-1, :num_cols-1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows-1, :num_cols-1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners*(move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners*(move_y == 0)) * horizontal_scale

    vertices = np.zeros((num_rows*num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2*(num_rows-1)*(num_cols-1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols-1) + i*num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2*i*(num_cols-1)
        stop = start + 2*(num_cols-1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start+1:stop:2, 0] = ind0
        triangles[start+1:stop:2, 1] = ind2
        triangles[start+1:stop:2, 2] = ind3

    return vertices, triangles, move_x != 0
