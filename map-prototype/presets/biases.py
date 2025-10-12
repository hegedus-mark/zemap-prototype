import numpy as np

def random_bias(length):
    vals = np.random.rand(length)
    vals = np.clip(vals, 0, None)
    return vals / vals.sum()

def start_low_bias(length, jaggedness):
    vals = np.linspace(0.1, 1.0, length)
    vals += np.random.normal(0, jaggedness, size=length)
    vals = np.clip(vals, 0, None)
    return vals / vals.sum()

def middle_high_bias(length, jaggedness):
    x = np.linspace(0,1,length)
    vals = np.exp(-((x-0.5)**2)/(0.08))  # peak in middle
    vals += np.random.normal(0, jaggedness, size=length)
    vals = np.clip(vals, 0, None)
    return vals / vals.sum()

def middle_low_bias(length, jaggedness):
    x = np.linspace(0,1,length)
    vals = 1 - np.exp(-((x-0.5)**2)/(0.08))  # low in middle
    vals += np.random.normal(0, jaggedness, size=length)
    vals = np.clip(vals, 0, None)
    return vals / vals.sum()

bias_presets = {
    "random": random_bias,
    "start_low": start_low_bias,
    "middle_high": middle_high_bias,
    "middle_low": middle_low_bias
}

def assign_biases(seed_lines, seed_line_to_preset, jaggedness=0.05):
    """
    Assign biases to each seed line based on a preset mapping.

    Parameters:
        seed_lines: list of seed lines
        seed_line_to_preset: dict mapping id(seed_line) -> preset_name
        jaggedness: controls randomness for each generator

    Returns:
        biases: list of arrays, each sums to 1
    """
    biases = []

    for line in seed_lines:
        length = len(line)
        preset_name = seed_line_to_preset.get(id(line), "random")  # default to 'random'
        bias_func = bias_presets[preset_name]
        bias_array = bias_func(length, jaggedness)
        biases.append(bias_array)

    return biases

def assign_bias(seed_line, preset_name, jaggedness=0.05):
    """
    Assign bias to a single seed line based on a preset.

    Parameters:
        seed_line: the seed line (list of points)
        preset_name: name of the bias preset
        jaggedness: controls randomness for each generator

    Returns:
        bias_array: array summing to 1
    """
    length = len(seed_line)
    bias_func = bias_presets.get(preset_name, random_bias)  # default to 'random'
    bias_array = bias_func(length, jaggedness)
    return bias_array