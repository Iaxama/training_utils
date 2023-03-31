import numpy as np

from tqdm import tqdm
import scipy.linalg
from scipy.special import legendre


def applyRMS(data):
    return np.sqrt(np.mean(np.square(data), axis=0))


def compute_features(data, window, increment, features_list):
    features = []
    for i in range(0, len(data) - (window - increment), increment):
        if features_list is not None:
            features.append(compute_features_window(
                data[i: i + window], features_list=features_list))
    features = np.array(features)
    return {'data': {f: features[:, i] for i, f in enumerate(features_list)}}


def compute_features_window(data, features_list):
    derivative = np.diff(data, axis=0)
    window = len(data)
    half_window = int(window / 2)

    # Mean Square (MSQ)
    if 'ms' in features_list or 'rms' in features_list or 'all' in features_list:
        ms = np.sum(np.square(data), axis=0) / window

    # Root Mean Square
    if 'rms' in features_list or 'all' in features_list:
        rms = np.sqrt(ms)

    # Mean Absolute Value
    if 'mav' in features_list or 'all' in features_list:
        mav = np.mean(np.abs(data), axis=0)

    # Zero Crossing
    if 'zc' in features_list or 'all' in features_list:
        z = data[1:] * data[:-1] < 0
        zc = np.array(sum(z))

    # Slope Sign changes
    if 'ssc' in features_list or 'all' in features_list:
        s = ((data[2:] - data[1:-1]) * (data[1:-1] - data[:-2])) < 0
        ssc = np.array(sum(s))

    # Waveform length
    if 'wl' in features_list or 'all' in features_list:
        wl = np.sum(np.abs(derivative), axis=0)

    # V-order 3  (V3)
    if 'v3' in features_list or 'all' in features_list:
        v3 = np.cbrt(np.sum(np.power(data, 3), axis=0) / window)

    # Log Detector (LD)
    if 'ld' in features_list or 'all' in features_list:
        abs_val = np.abs(data, dtype=float)
        abs_val[abs_val == 0] += 1e-6  # Avoid log10(0)
        ld = np.exp(np.sum(np.log10(abs_val), axis=0) / window)

    # Mean Absolute Value Slope
    if 'mavs' in features_list or 'all' in features_list:
        mavs = (np.sum(np.abs(data[:half_window]), axis=0) -
                np.sum(np.abs(data[half_window:]), axis=0)) / half_window

    # # Wilson Amplitude
    if 'wa' in features_list or 'all' in features_list:
        # TODO: check if use the standard deviation or fixed threshold
        wa = np.sum((np.abs(derivative) - np.std(data, axis=0)) > 0, axis=0)

    # Difference Absolute Standard Deviation (DABS)
    if 'dabs' in features_list or 'all' in features_list or 'mfl' in features_list:
        dabs = np.sqrt((np.sum(np.square(derivative), axis=0)) / (window - 1))

    # Maximum Fractal Length (MFL)
    if 'mfl' in features_list or 'all' in features_list:
        nonzero_dabs = dabs.copy()
        nonzero_dabs[nonzero_dabs == 0] = 1e-6  # Avoid log10(0)
        # TODO: check the usability and its information
        mfl = np.log10(nonzero_dabs) - 0.5 * np.log10(window - 1)
        mfl[np.where(np.isinf(mfl))] = min(mfl[np.where(np.isfinite(mfl))])

    # Myopulse percentage rate (MPR)
    if 'mpr' in features_list or 'all' in features_list:
        mpr = np.sum((np.abs(data) - np.std(data, axis=0)) > 0, axis=0)

    variables = locals()
    return np.array([variables[x] for x in features_list])


def convert_nparray_to_spikes(data, threshold=.05, output_off_spikes=False):
    # data -= np.mean(data)
    # data /= np.std(data)

    diff = np.zeros(data[:, 0].shape)
    spikes = np.zeros(data.shape)
    off_spikes = np.zeros(data.shape)

    last_time_step = data[:, 0]
    for n, time_step in enumerate(data.swapaxes(0, 1)):
        diff += time_step - last_time_step
        last_time_step = time_step
        is_spike = diff > threshold
        spikes[:, n][is_spike] = 1
        diff[is_spike] = 0
        if output_off_spikes:
            is_off_spike = diff < -threshold
            off_spikes[:, n][is_off_spike] = 1
            diff[is_off_spike] = 0
    if output_off_spikes:
        spikes = np.stack((spikes, off_spikes), axis=1)
    else:
        spikes = np.expand_dims(spikes, axis=1)
    return spikes.astype(np.float32)


class LDN():
    def __init__(self, theta, q, size_in=1):
        # number of internal state dimensions (polynomials) per input
        self.q = q
        self.theta = theta      # size of time window (in seconds)
        self.size_in = size_in  # number of inputs (channels)
        # Do Aaron's math to generate the matrices A and B so that
        #  dx/dt = Ax + Bu will convert u into a legendre representation over a window theta
        #  https://github.com/arvoelke/nengolib/blob/master/nengolib/synapses/analog.py#L536
        A = np.zeros((q, q))
        B = np.zeros((q, 1))
        for i in range(q):
            B[i] = (-1.)**i * (2*i+1)
            for j in range(q):
                # DON'T CHANGE THIS! TRUST IT
                A[i, j] = (2*i+1)*(-1 if i < j else (-1.)**(i-j+1))
        self.A = A / theta
        self.B = B / theta

    def make_step(self, dt):
        state = np.zeros((self.q, self.size_in))
        # Handle the fact that we're discretizing the time step
        #  https://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models
        Ad = scipy.linalg.expm(self.A*dt)
        Bd = np.dot(np.dot(np.linalg.inv(self.A), (Ad-np.eye(self.q))), self.B)
        # this code will be called every timestep

        def step_legendre(x, state=state):
            state[:] = np.dot(Ad, state) + np.dot(Bd, x[None, :])
            return state.T.flatten()
        return step_legendre

    def get_weights_for_delays(self, r):
        # compute the weights needed to extract the value at time r
        # from the network (r=0 is right now, r=1 is theta seconds ago)
        r = np.asarray(r)
        m = np.asarray([legendre(i)(2*r - 1) for i in range(self.q)])
        return m.reshape(self.q, -1).T


def compute_legendre_polynomials(data, dt, q, theta):
    ldn = LDN(theta, q)
    step = ldn.make_step(dt)

    compressed_data = []
    for channel in tqdm(data.swapaxes(0, 1)):
        compressed_data.append([step(x) for x in channel.reshape(-1, 1)])
    out = np.array(compressed_data).swapaxes(0, 1)
    out = (out - out.min()) / (out.max() - out.min())
    return out
