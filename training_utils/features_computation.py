import numpy as np
import torch

from snntorch import spikegen

def applyRMS(data):
    return np.sqrt(np.mean(np.square(data), axis=0))


def compute_features(data, window, increment, features):
    half_window = int(window / 2)
    # TODO look at https://github.com/pydata/bottleneck for faster approaches
    sliding_win_view = np.array([data[i:i + window] for i in range(0, len(data) - window + 1, increment)])
    diff = np.diff(data, axis=0, prepend=0)
    derivative = np.array([diff[i:i + window] for i in range(0, len(diff) - window + 1, increment)])

    # Mean Square (MSQ)
    if 'ms' in features or 'rms' in features or 'all' in features:
        ms = np.sum(np.square(sliding_win_view), axis=1) / window

    # Root Mean Square
    if 'rms' in features or 'all' in features:
        rms = np.sqrt(ms)

    # Mean Absolute Value
    if 'mav' in features or 'all' in features:
        mav = np.mean(np.abs(sliding_win_view), axis=1)

    # Zero Crossing
    if 'zc' in features or 'all' in features:
        z = data[1:] * data[:-1] < 0
        z_range = range(0, len(z) - window + 1, increment)
        if not z_range:
            z_range = [0]
        zc = np.array([sum(z[i:i + window]) for i in z_range])

    # Slope Sign changes
    if 'ssc' in features or 'all' in features:
        s = ((data[2:] - data[1:-1]) * (data[1:-1] - data[:-2])) < 0
        s_range = range(0, len(s) - window + 1, increment)
        if not s_range:
            s_range = [0]
        ssc = np.array([sum(s[i:i + window]) for i in s_range])

    # Waveform length
    if 'wl' in features or 'all' in features:
        wl = np.sum(np.abs(derivative), axis=1)

    # V-order 3  (V3)
    if 'v3' in features or 'all' in features:
        v3 = np.cbrt(np.sum(np.power(sliding_win_view, 3), axis=1) / window)

    # Log Detector (LD)
    if 'ld' in features or 'all' in features:
        abs_val = np.abs(sliding_win_view, dtype=float)
        abs_val[abs_val == 0] += 1e-6  # Avoid log10(0)
        ld = np.exp(np.sum(np.log10(abs_val), axis=1) / window)

    # Mean Absolute Value Slope
    if 'mavs' in features or 'all' in features:
        mavs = (np.sum(np.abs(sliding_win_view[:, :half_window, :]), axis=1) -
                np.sum(np.abs(sliding_win_view[:, half_window:, :]), axis=1)) / half_window

    # # Wilson Amplitude
    if 'wa' in features or 'all' in features:
        try:
            # TODO: check if use the standard deviation or fixed threshold
            wa = np.sum((np.abs(derivative) -
                         np.expand_dims(np.std(sliding_win_view, axis=1), -1)) > 0, axis=1)
        except ValueError:
            wa = np.sum((np.abs(derivative) -
                         np.expand_dims(np.std(sliding_win_view, axis=1), 1)[:len(derivative)]) > 0, axis=1)

    # Difference Absolute Standard Deviation (DABS)
    if 'dabs' in features or 'all' in features or 'mfl' in features:
        dabs = np.sqrt((np.sum(np.square(derivative), axis=1)) / (window - 1))

    # Maximum Fractal Length (MFL)
    if 'mfl' in features or 'all' in features:
        nonzero_dabs = dabs.copy()
        nonzero_dabs[nonzero_dabs == 0] += 1e-6  # Avoid log10(0)
        mfl = np.log10(nonzero_dabs) - 0.5 * np.log10(window - 1)  # TODO: check the usability and its information
        mfl[np.where(np.isinf(mfl))] = min(mfl[np.where(np.isfinite(mfl))])

    # Myopulse percentage rate (MPR)
    if 'mpr' in features or 'all' in features:
        mpr = np.sum((np.abs(sliding_win_view) -
                      np.expand_dims(np.std(sliding_win_view, axis=1), 1)) > 0, axis=1)

    variables = locals()
    features_dict = {
        'data': {x: variables[x] for x in features},
        'window': window,
        'increment': increment
    }
    min_len = min([len(features_dict['data'][x]) for x in features_dict['data']])

    for feat in features_dict['data']:
        features_dict['data'][feat] = features_dict['data'][feat][:min_len]

    return features_dict

def convert_tensor_to_spikes(data, threshold=.05):
    data -= torch.mean(data)
    data /= torch.std(data)

    spikes = spikegen.delta(data, threshold=threshold, padding=False, off_spike=True)
    all_spikes = torch.zeros((spikes.shape[0], 2, spikes.shape[1]))
    all_spikes[:, 0, :] = torch.where(spikes == 1, 1, 0)
    all_spikes[:, 1, :] = torch.where(spikes == -1, 1, 0)
    all_spikes = torch.unsqueeze(all_spikes, -1)
    return all_spikes
