from scipy.spatial import Voronoi
import random
from seed_lines import generate_seed_line,seedline_presets
from biases import assign_bias, bias_presets
import numpy as np

def build_continent_presets(bias_presets, seedline_presets):
    continent_presets = {}
    for bias_name, bias_fn in bias_presets.items():
        for line_name, line_fn in seedline_presets.items():
            combo_name = f"{bias_name} {line_name}"
            continent_presets[combo_name] = dict(
                bias_fn=bias_fn,
                bias_name=bias_name,
                line_fn=line_fn,
                line_name=line_name
            )
    return continent_presets

def generate_micro_voronoi(num_points=600, width=1000, height=1000, lloyd_iterations=3, seed=42):
    np.random.seed(seed)
    points = np.random.rand(num_points,2) * np.array([width,height])
    for _ in range(lloyd_iterations):
        vor = Voronoi(points)
        new_points = []
        for i, region_index in enumerate(vor.point_region):
            vertices_idx = vor.regions[region_index]
            if -1 in vertices_idx or len(vertices_idx)==0:
                new_points.append(points[i])
                continue
            polygon = vor.vertices[vertices_idx]
            centroid = polygon.mean(axis=0)
            centroid = np.clip(centroid, [0,0], [width, height], out=centroid)
            new_points.append(centroid)
        points = np.array(new_points)
    vor = Voronoi(points)
    return points, vor

def preset_chooser(num_continents, preset_catalog, seed=42):
    random.seed(seed)
    preset_names = list(preset_catalog.keys())
    chosen = random.choices(preset_names, k=num_continents)

    # map continent IDs (0..num_continents-1) to chosen presets
    return {i: chosen[i] for i in range(num_continents)}

def pareto_distribution(num_plates, num_land_cells, size_irregularity):
    raw_weights = np.random.power(a=2.5, size=num_plates) + 1
    raw_weights = raw_weights ** (1 + 2 * size_irregularity)
    weights = raw_weights / np.sum(raw_weights)
    plate_sizes = (weights * num_land_cells).astype(int)
    plate_sizes[-1] += num_land_cells - np.sum(plate_sizes) < 1
    return plate_sizes

def get_voronoi_adjacency(vor):
    """Build adjacency map for Voronoi points"""
    adjacency = {i: set() for i in range(len(vor.points))}
    for (p1, p2) in vor.ridge_points:
        adjacency[p1].add(p2)
        adjacency[p2].add(p1)
    return adjacency

def grow_plate(adjacency, seed_line, biases, target_size):
    """
    Grows a single continent (plate) from its seed line.
    Expands based on bias until reaching the target size.
    
    Returns:
        set of cell indices that belong to this plate.
    """
    assigned = set(seed_line)
    frontier = [(cell, bias) for cell, bias in zip(seed_line, biases)]
    
    while len(assigned) < target_size and frontier:
        new_frontier = []
        
        # sort frontier by bias (higher bias = stronger expansion)
        frontier.sort(key=lambda x: x[1], reverse=True)
        
        for cell, bias in frontier:
            # get all unassigned neighbors
            neighbors = adjacency[cell] - assigned
            if not neighbors:
                continue
            
            # bias-influenced expansion probability
            for n in neighbors:
                if len(assigned) >= target_size:
                    break
                
                if random.random() < bias:
                    assigned.add(n)
                    new_frontier.append((n, bias * random.uniform(0.8, 1.2)))
        
        frontier = new_frontier

    return assigned

def compute_bounding_box(vor, cell_indices):
    """
    Compute the axis-aligned bounding box for a set of Voronoi cells.
    
    Returns:
        (min_x, min_y, max_x, max_y)
    """
    if not cell_indices:
        return (0, 0, 0, 0)

    coords = vor.points[list(cell_indices)]
    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)
    return (min_x, min_y, max_x, max_y)

def place_continents(bounding_boxes, width=1000, height=1000, padding=50, max_tries=1000):
    """
    Finds non-overlapping placements for continent bounding boxes.

    Args:
        bounding_boxes: dict of {continent_id: (min_x, min_y, max_x, max_y)}
        width, height: map dimensions
        padding: minimum spacing between continents
        max_tries: maximum attempts per continent

    Returns:
        dict of {continent_id: (offset_x, offset_y)} translation offsets
    """
    placements = {}
    placed_areas = []

    for cid, (min_x, min_y, max_x, max_y) in bounding_boxes.items():
        bbox_w = max_x - min_x
        bbox_h = max_y - min_y
        placed = False

        for _ in range(max_tries):
            # Try a random position within map bounds
            ox = random.uniform(padding, width - bbox_w - padding)
            oy = random.uniform(padding, height - bbox_h - padding)
            new_rect = (ox, oy, ox + bbox_w, oy + bbox_h)

            # Check overlap with existing placements
            overlap = False
            for (x1, y1, x2, y2) in placed_areas:
                if not (new_rect[2] + padding < x1 or
                        new_rect[0] - padding > x2 or
                        new_rect[3] + padding < y1 or
                        new_rect[1] - padding > y2):
                    overlap = True
                    break

            if not overlap:
                placements[cid] = (ox - min_x, oy - min_y)  # offset relative to original coords
                placed_areas.append(new_rect)
                placed = True
                break

        if not placed:
            print(f"⚠️ Warning: Could not find space for continent {cid} after {max_tries} tries.")

    return placements

def offset_continent(vor, continent, offset):
    """
    Translates all cells in a continent by a given offset.

    Args:
        vor: The Voronoi diagram object.
        continent: dict with keys {"id", "cells", "seed_line"}.
        offset: (dx, dy) offset tuple from placement step.
    """
    dx, dy = offset
    cell_indices = continent["cells"]

    # Apply offset to points corresponding to the continent cells
    vor.points[cell_indices] += np.array([dx, dy])

    # Optional: also move seed line positions for visualization or debugging
    if "seed_line" in continent:
        for idx in range(len(continent["seed_line"])):
            continent["seed_line"][idx] = int(continent["seed_line"][idx])
        vor.points[continent["seed_line"]] += np.array([dx, dy])



def generate_map():
  num_voronoi = 1200
  width = 1000
  height = 1000
  lloyd_iterations = 3
  num_continents = 5
  water_fraction = 0.7
  size_irregularity = 0.1
  seed = random.randint(0, 10000)

  points, vor = generate_micro_voronoi(num_voronoi, width, height, lloyd_iterations, seed)
  
  num_cells = len(vor.points)
  num_land_cells = int(num_cells * (1 - water_fraction))
  adjacency = get_voronoi_adjacency(vor)

  plate_sizes = pareto_distribution(num_continents, num_land_cells, size_irregularity)
  
  continent_presets = build_continent_presets(bias_presets, seedline_presets)
  assignments = preset_chooser(num_continents, continent_presets, seed)

  grown_continents = []
  for plate_id, preset_name in assignments.items():
        bias_name = continent_presets[preset_name]['bias_name']
        line_name = continent_presets[preset_name]['line_name']

        line = generate_seed_line(vor, plate_sizes[plate_id], line_name)
        bias = assign_bias(line, bias_name)
        cells = grow_plate( adjacency, line, bias, plate_sizes[plate_id])
        grown_continents.append({"id": plate_id, "cells": cells, "seed_line": line})

  # Compute bounding boxes and place
  bounding_boxes = {c["id"]: compute_bounding_box(vor, c["cells"]) for c in grown_continents}
  placements = place_continents(bounding_boxes)

  # Apply placements
  for c in grown_continents:
      offset_continent(vor, c, placements[c["id"]])

  

