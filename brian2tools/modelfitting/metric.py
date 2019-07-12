import abc
from numpy import (shape, array, sum, square, reshape, abs, amin, digitize,
                   rint, arange, atleast_2d, NaN, float64)
from brian2 import Hz

def firing_rate(spikes):
    '''
    Rate of the spike train.
    '''
    if len(spikes) < 2:
        return NaN
    return (len(spikes) - 1) / (spikes[-1] - spikes[0])


def get_gamma_factor(source, target, delta, dt):
    """Calculate gamma factor between source and tagret spike trains"""
    source = array(source)
    target = array(target)
    target_rate = firing_rate(target) * Hz

    source = array(rint(source / dt), dtype=int)
    target = array(rint(target / dt), dtype=int)
    delta_diff = int(rint(delta / dt))

    source_length = len(source)
    target_length = len(target)

    if (source_length > 1):
        bins = .5 * (source[1:] + source[:-1])
        indices = digitize(target, bins)
        diff = abs(target - source[indices])
        matched_spikes = (diff <= delta_diff)
        coincidences = sum(matched_spikes)
    else:
        indices = [amin(abs(source - target[i])) <= delta_diff for i in arange(target_length)]
        coincidences = sum(indices)

    # Normalization of the coincidences count
    NCoincAvg = 2 * delta * target_length * target_rate
    norm = .5*(1 - 2 * target_rate * delta)
    gamma = (coincidences - NCoincAvg)/(norm*(source_length + target_length))
    
    return gamma


class Metric(object):
    """
    Metic acstract class to define functions required for a custom metric
    To be used with modelfitting fit_traces.
    """
    __metaclass__= abc.ABCMeta

    def __init__(self, **kwds):
        """Initialize the metric"""
        pass

    @abc.abstractmethod
    def traces_to_features(self, traces, output):
        """Function calculates features or errors for each of the traces"""
        pass

    @abc.abstractmethod
    def features_to_errors(self, features, n_traces):
        """
        Function weights features/multiple errors into one final error fed
        back to the optimization algorithm
        """
        pass

    @abc.abstractmethod
    def calc(traces, output, n_traces):
        """Performs the error calculation"""
        pass


class MSEMetric(Metric):
    def __init__(self):
        super(Metric, self).__init__()

    def traces_to_features(self, traces, output, n_traces):
        mse_list = []
        output = atleast_2d(output)

        for i in arange(n_traces):
            temp_out = output[i]
            temp_traces = traces[i::n_traces]

            for trace in temp_traces:
                mse = sum(square(temp_out - trace))
                mse_list.append(mse)

        return mse_list

    def features_to_errors(self, features, n_traces):
        feat_arr = reshape(array(features), (n_traces,
                           int(len(features)/n_traces)))
        return feat_arr.mean(axis=0)

    def calc(self, traces, output, n_traces):
        mse = self.traces_to_features(traces, output, n_traces)
        errors = self.features_to_errors(mse, n_traces)

        return errors


class GammaFactor(Metric):
    def __init__(self, dt, delta=None):
        super(Metric, self)
        if delta is None:
            raise Exception('delta (time window for gamma factor), \
                             has to be set')
        self.delta = delta
        self.dt = dt

    def traces_to_features(self, traces, output, n_traces):
        gamma_factors = []
        if type(output[0]) == float64:
            output = atleast_2d(output)

        for i in arange(n_traces):
            temp_out = output[i]
            temp_traces = traces[i::n_traces]

            for trace in temp_traces:
                gf = get_gamma_factor(trace, temp_out, self.delta, self.dt)
                gamma_factors.append(1 - gf)

        return gamma_factors

    def features_to_errors(self, features, n_traces):
        feat_arr = reshape(array(features), (n_traces,
                           int(len(features)/n_traces)))
        return feat_arr.mean(axis=0)

    def calc(self, traces, output, n_traces):
        gamma_factors = self.traces_to_features(traces, output, n_traces)
        errors = self.features_to_errors(gamma_factors, n_traces)

        return errors
