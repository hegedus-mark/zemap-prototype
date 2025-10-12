import random
from dash import Input, Output, dcc, html
import dash
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys
import plotly.graph_objects as go
from scipy.spatial import Voronoi, voronoi_plot_2d, distance

def plot_voronoi(vor, width=1000, height=1000):
    """Plot Voronoi diagram with map boundaries"""
    fig, ax = plt.subplots(figsize=(8,8))
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black')
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    plt.show()


def darken_color_hls(color, factor=0.6):
    """Darken a color while preserving hue and saturation using HLS space."""
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0, l * factor)  # reduce lightness
    r_new, g_new, b_new = colorsys.hls_to_rgb(h, l, s)
    return (r_new, g_new, b_new)

def plot_plates_with_seeds(vor, plate_assignment, seed_lines, seed_line_to_plate, num_continental_plates, seed_biases):
    # Use tab10 or tab20b for better contrast
    colors = plt.cm.get_cmap("tab10", num_continental_plates)
    
    fig, ax = plt.subplots(figsize=(8,8))

    # Draw plates (Voronoi regions)
    for i, region_index in enumerate(vor.point_region):
        vertices_idx = vor.regions[region_index]
        if -1 in vertices_idx or len(vertices_idx) == 0:
            continue
        polygon = vor.vertices[vertices_idx]
        if np.any((polygon < -50) | (polygon > 1050)):
            continue
        plate_id = plate_assignment[i]
        color = "lightblue" if plate_id == -1 else colors(plate_id)
        ax.fill(*zip(*polygon), color=color, edgecolor='k', linewidth=0.3)

    # Draw seed lines with perceptually darkened color
    for line, bias in zip(seed_lines, seed_biases):
        plate_id = seed_line_to_plate.get(id(line), -1)
        base_color = "red" if plate_id == -1 else colors(plate_id)

        coords = vor.points[line]
        # Plot segment by segment with color intensity = bias
        for j in range(len(coords)-1):
            c = darken_color_hls(base_color, factor=0.3 + 0.7*bias[j])  # bias controls darkness
            ax.plot(coords[j:j+2,0], coords[j:j+2,1], color=c, linewidth=2)

        # Optional: label each point with bias percentage
        for pt, b in zip(coords, bias):
            ax.text(pt[0], pt[1], f"{b*100:.0f}%", fontsize=6, color='black', ha='center', va='center')

    ax.set_aspect('equal')
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    plt.show()

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


def get_voronoi_adjacency(vor):
    """Build adjacency map for Voronoi points"""
    adjacency = {i: set() for i in range(len(vor.points))}
    for (p1, p2) in vor.ridge_points:
        adjacency[p1].add(p2)
        adjacency[p2].add(p1)
    return adjacency

def pareto_distribution(num_plates, num_land_cells, size_irregularity):
    raw_weights = np.random.power(a=2.5, size=num_plates) + 1
    raw_weights = raw_weights ** (1 + 2 * size_irregularity)
    weights = raw_weights / np.sum(raw_weights)
    plate_sizes = (weights * num_land_cells).astype(int)
    plate_sizes[-1] += num_land_cells - np.sum(plate_sizes) < 1
    return plate_sizes

def generate_seed_lines(vor, plate_sizes, max_cum_angle=90, max_step=10, 
                        segment_length_range=(30, 60), chaotic=False):
    seed_lines = []
    num_cells = len(vor.points)

    for size in plate_sizes:
        start_idx = random.randint(0, num_cells - 1)
        points = vor.points
        start_point = points[start_idx]

        # derive line length & segment length from plate size (soft scaling)
        relative_size = size / max(plate_sizes)
        line_length = int(np.interp(relative_size, [0, 1], [4, 9])) + random.choice([-1, 0, 1])
        segment_length = np.interp(relative_size, [0, 1], segment_length_range)
        segment_length *= random.uniform(0.8, 1.2)

        # direction setup
        base_angle = random.uniform(0, 360)
        current_angle = base_angle
        total_angle_change = 0

        line = [start_idx]
        x, y = start_point

        for _ in range(line_length - 1):
            if chaotic:
                delta_angle = random.uniform(-180, 180)
            else:
                delta_angle = random.uniform(-max_step, max_step)
                if abs(total_angle_change + delta_angle) > max_cum_angle:
                    delta_angle *= -1
            total_angle_change += delta_angle
            current_angle += delta_angle

            new_x = x + np.cos(np.radians(current_angle)) * segment_length
            new_y = y + np.sin(np.radians(current_angle)) * segment_length
            new_x = np.clip(new_x, 0, 1000)
            new_y = np.clip(new_y, 0, 1000)
            new_idx = np.argmin(np.linalg.norm(points - np.array([new_x, new_y]), axis=1))
            line.append(new_idx.astype(int))
            x, y = points[new_idx]

        seed_lines.append(line)
    return seed_lines


def assign_biases(seed_lines, chaotic=False):
    biases = []
    for line in seed_lines:
        length = len(line)
        if chaotic:
            b = np.random.rand(length)
        else:
            start = random.uniform(0.1, 0.3)
            end = random.uniform(0.5, 0.8)
            b = np.linspace(start, end, length)
            b += np.random.normal(0, 0.05, size=length)
            b = np.clip(b, 0, 1)
        biases.append(b)
    return biases


def grow_plate(vor, adjacency, seed_line, biases, target_size, unassigned, plate_id, plate_assignment):
    """Expands a plate from a seed line based on bias influence."""
    frontier = [(cell, bias) for cell, bias in zip(seed_line, biases)]
    for cell, _ in frontier:
        if cell in unassigned:
            plate_assignment[cell] = plate_id
            unassigned.remove(cell) 

    while len(np.where(plate_assignment == plate_id)[0]) < target_size and frontier:
        new_frontier = []
        # Weighted expansion based on bias
        frontier.sort(key=lambda x: x[1], reverse=True)
        for cell, bias in frontier:
            neighbors = list(adjacency[cell] & unassigned)
            random.shuffle(neighbors)
            for n in neighbors:
                if len(np.where(plate_assignment == plate_id)[0]) >= target_size:
                    break
                # bias-based probability of expanding
                if random.random() < bias:
                    plate_assignment[n] = plate_id
                    unassigned.remove(n)
                    new_frontier.append((n, bias * random.uniform(0.8, 1.2)))
        frontier = new_frontier

def placement_strategy(seed_lines, vor, gap=150, width=1000, height=1000):
    """Translates existing seed lines to reduce overlap or group continents."""
    placed_lines = []
    used_regions = []
    cx, cy = 0, 0

    for line in seed_lines:
        # random center placement with spacing
        placed = False
        while not placed:
            cx = random.uniform(gap, width - gap)
            cy = random.uniform(gap, height - gap)
            if all(np.linalg.norm(np.array([cx, cy]) - np.array(r)) > gap for r in used_regions):
                used_regions.append((cx, cy))
                placed = True

        # translate seed line around center
        coords = vor.points[line]
        offset = np.array([cx, cy]) - coords.mean(axis=0)
        new_line = [np.argmin(np.linalg.norm(vor.points - (p + offset), axis=1)) for p in coords]
        placed_lines.append(new_line)
    return placed_lines

def dedicate_plates(vor, num_continental_plates=6, water_fraction=0.7,
                    size_irregularity=0.1, seed=None, chaotic=False):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    num_cells = len(vor.points)
    adjacency = get_voronoi_adjacency(vor)

    num_land_cells = int(num_cells * (1 - water_fraction))
    plate_sizes = pareto_distribution(num_continental_plates, num_land_cells, size_irregularity)

    # Stage 1: Generate seed lines
    seed_lines = generate_seed_lines(vor, plate_sizes, chaotic=chaotic)

    # Stage 2: Assign biases
    biases = assign_biases(seed_lines, chaotic=chaotic)

    # Stage 3: Placement adjustment
    seed_lines = placement_strategy(seed_lines, vor)

    # Stage 4: Grow plates
    plate_assignment = np.full(num_cells, -1, dtype=int)
    unassigned = set(range(num_cells))


    seed_line_to_plate = {}
    for plate_id, (line, b, target_size) in enumerate(zip(seed_lines, biases, plate_sizes)):
        grow_plate(vor, adjacency, line, b, target_size, unassigned, plate_id, plate_assignment)
        seed_line_to_plate[id(line)] = plate_id  # associate the specific seed line object

    return seed_lines, plate_assignment, seed_line_to_plate, biases

num_voronoi = 1200
width = 1000
height = 1000
lloyd_iterations = 3
seed = random.randint(0, 10000)

# Pre-generate points so we can just reassign plates interactively
points, vor = generate_micro_voronoi(num_voronoi, width, height, lloyd_iterations, seed)

# --- Dash App ---
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Interactive Voronoi Plate Generator"),
    
    html.Div([
        html.Label("Number of Continental Plates:"),
        dcc.Slider(id='num_plates', min=1, max=12, step=1, value=6, marks={i:str(i) for i in range(1,13)}),
    ], style={'margin':'20px'}),
    
    html.Div([
        html.Label("Water Fraction:"),
        dcc.Slider(id='water_fraction', min=0.0, max=0.9, step=0.05, value=0.7, marks={i/10:str(i/10) for i in range(0,10)}),
    ], style={'margin':'20px'}),
    
    html.Div([
        html.Label("Size Irregularity:"),
        dcc.Slider(id='size_irregularity', min=0.0, max=1.0, step=0.05, value=0.7),
    ], style={'margin':'20px'}),
    
    html.Div([
        html.Label("Chaotic Seed Lines:"),
        dcc.Checklist(id='chaotic', options=[{'label':'Enable', 'value':1}], value=[]),
    ], style={'margin':'20px'}),
    
    dcc.Graph(id='voronoi_graph', style={'height':'800px'}),
])

# --- Callback to update plot ---
@app.callback(
    Output('voronoi_graph', 'figure'),
    [Input('num_plates', 'value'),
     Input('water_fraction', 'value'),
     Input('size_irregularity', 'value'),
     Input('chaotic', 'value')]
)
def update_graph(num_plates, water_fraction, size_irregularity, chaotic):
    chaotic_flag = bool(chaotic)
    seed_lines, plate_assignment, seed_line_to_plate, seed_biases = dedicate_plates(
        vor, num_plates, water_fraction, size_irregularity, seed, chaotic=chaotic_flag
    )
    
    # Plot using Plotly
    cmap = plt.get_cmap("tab10")
    colors = [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for c in [cmap(i) for i in range(num_plates)]]
    
    fig = go.Figure()

    # Draw plates
    for i, region_index in enumerate(vor.point_region):
        vertices_idx = vor.regions[region_index]
        if -1 in vertices_idx or len(vertices_idx) == 0:
            continue
        polygon = vor.vertices[vertices_idx]
        if np.any((polygon < -50) | (polygon > 1050)):
            continue
        plate_id = plate_assignment[i]
        color = "lightblue" if plate_id == -1 else colors[plate_id]
        fig.add_trace(go.Scatter(
            x=polygon[:,0].tolist() + [polygon[0,0]],
            y=polygon[:,1].tolist() + [polygon[0,1]],
            fill="toself",
            fillcolor=color,
            line=dict(color='black', width=0.5),
            hoverinfo="skip",
            showlegend=False
        ))
    
    # Draw seed lines
    for line, bias in zip(seed_lines, seed_biases):
        plate_id = seed_line_to_plate.get(id(line), -1)
        base_color = "red" if plate_id == -1 else colors[plate_id]
        coords = vor.points[line]
        for j in range(len(coords)-1):
            alpha = 0.3 + 0.7*bias[j]
            if plate_id >= 0:
                r, g, b, _ = plt.get_cmap("tab10")(plate_id)
                color = f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})"
            else:
                color = f"rgba(255,0,0,{alpha})"
            fig.add_trace(go.Scatter(
                x=[coords[j,0], coords[j+1,0]],
                y=[coords[j,1], coords[j+1,1]],
                mode='lines',
                line=dict(color=color, width=3),
                hoverinfo='text',
                text=f"Bias: {bias[j]*100:.0f}%",
                showlegend=False
            ))
    
    fig.update_layout(
        xaxis=dict(range=[0,1000], showgrid=False, zeroline=False),
        yaxis=dict(range=[0,1000], showgrid=False, zeroline=False),
        plot_bgcolor='white',
        hovermode='closest',
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)