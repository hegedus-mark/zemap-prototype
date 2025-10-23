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