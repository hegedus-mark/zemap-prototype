import numpy as np
import random

def generate_path(points, start_idx, line_length, segment_length, angle_fn):
    """
    Generic path generator.
    - points: Voronoi points
    - start_idx: starting index
    - line_length: number of segments
    - segment_length: average distance per segment
    - angle_fn: function (i, prev_angle, state) -> new_angle, new_state
    """
    x, y = points[start_idx]
    line = [start_idx]
    angle = random.uniform(0, 360)
    state = {}  # preset-specific memory (e.g., for spirals)

    for i in range(line_length - 1):
        # let the preset decide the next angle
        angle, state = angle_fn(i, angle, state)

        # compute new position
        new_x = np.clip(x + np.cos(np.radians(angle)) * segment_length, 0, 1000)
        new_y = np.clip(y + np.sin(np.radians(angle)) * segment_length, 0, 1000)

        # find nearest Voronoi cell
        new_idx = np.argmin(np.linalg.norm(points - np.array([new_x, new_y]), axis=1))
        line.append(int(new_idx))

        # move position
        x, y = points[new_idx]

    return line


def straight_seed_line(points, start_idx, size, max_size):
    """Mostly straight line with small deviations."""
    line_length = int(np.interp(size / max_size, [0, 1], [4, 8]))
    segment_length = np.interp(size / max_size, [0, 1], [30, 60])

    def angle_fn(i, prev_angle, state):
        delta = random.uniform(-5, 5)
        return prev_angle + delta, state

    return generate_path(points, start_idx, line_length, segment_length, angle_fn)


def snake_seed_line(points, start_idx, size, max_size):
    """Wavy snake-like line."""
    line_length = int(np.interp(size / max_size, [0, 1], [6, 12]))
    segment_length = np.interp(size / max_size, [0, 1], [30, 60])

    def angle_fn(i, prev_angle, state):
        delta = random.uniform(-40, 40)
        return prev_angle + delta, state

    return generate_path(points, start_idx, line_length, segment_length, angle_fn)


def spiral_seed_line(points, start_idx, size, max_size):
    """Spiral-like seed line (growing radius and changing angle)."""
    line_length = int(np.interp(size / max_size, [0, 1], [8, 16]))
    segment_length = np.interp(size / max_size, [0, 1], [20, 40])
    base_angle = random.uniform(0, 360)
    angle_step = random.uniform(15, 30)

    def angle_fn(i, prev_angle, state):
        # store cumulative angle in state
        state.setdefault("angle_base", base_angle)
        return base_angle + i * angle_step, state

    return generate_path(points, start_idx, line_length, segment_length, angle_fn)


def chaotic_seed_line(points, start_idx, size, max_size):
    """Totally random walk."""
    line_length = int(np.interp(size / max_size, [0, 1], [8, 16]))
    segment_length = np.interp(size / max_size, [0, 1], [20, 40])

    def angle_fn(i, prev_angle, state):
        return random.uniform(0, 360), state

    return generate_path(points, start_idx, line_length, segment_length, angle_fn)



seedline_presets = {
    "straight": straight_seed_line,
    "snake": snake_seed_line,
    "spiral": spiral_seed_line,
    "chaotic": chaotic_seed_line,
}

def generate_seed_lines_with_presets(vor, plate_sizes, line_presets):
    seed_lines = []
    points = vor.points
    num_cells = len(points)
    max_size = max(plate_sizes)

    for i, size in enumerate(plate_sizes):
        preset_name = line_presets.get(i, "straight")
        generator = seedline_presets.get(preset_name, straight_seed_line)

        start_idx = random.randint(0, num_cells - 1)
        line = generator(points, start_idx, size, max_size)
        seed_lines.append(line)

    return seed_lines

def generate_seed_line(vor, plate_size, preset_name):
    """Generate a single seed line for testing."""
    points = vor.points
    num_cells = len(points)
    max_size = plate_size  # single plate

    generator = seedline_presets.get(preset_name, straight_seed_line)
    start_idx = random.randint(0, num_cells - 1)
    line = generator(points, start_idx, plate_size, max_size)
    return line