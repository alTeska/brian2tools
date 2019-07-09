import abc
from numpy import shape, array, sum, square, reshape, abs, amin, digitize, rint
from brian2 import Hz

def firing_rate(spikes):
    '''
    Rate of the spike train.
    '''
    if len(spikes)<2:
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
        indices = [amin(abs(source - target[i])) <= delta_diff for i in xrange(target_length)]
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
    def traces_to_features(self, traces, output_traces):
        """Function calculates features or errors for each of the traces"""
        pass

    @abc.abstractmethod
    def features_to_errors(self, features, Ntraces):
        """
        Function weights features/multiple errors into one final error fed
        back to the optimization algorithm
        """
        pass

    @abc.abstractmethod
    def calc(traces, output_traces, Ntraces):
        """Performs the error calculation"""
        pass


class MSEMetric(Metric):
    def __init__(self):
        super(Metric, self).__init__()

    def traces_to_features(self, traces, output_traces):
        mse = []

        for trace in traces:
            mse.append(sum(square(output_traces - trace)))

        return mse

    def features_to_errors(self, features, n_traces):
        feat_arr = reshape(array(features), (int(len(features)/n_traces),
                           n_traces))
        return feat_arr.mean(axis=1)

    def calc(self, traces, output_traces, Ntraces):
        mse = self.traces_to_features(traces, output_traces)
        errors = self.features_to_errors(mse, Ntraces)

        return errors


class GammaFactor(Metric):
    def __init__(self):
        super(Metric, self, delta, dt)
        self.dt = dt
        self.delta = delta

    def traces_to_features(self, traces, output_traces):
        gamma_factors = []

        for trace in traces:
            gf = get_gamma_factor(trace, output_traces,)
            gamma_factors.append(g_factor)

        return gamma_factors

    def features_to_errors(self, features, n_traces):
        feat_arr = reshape(array(features), (int(len(features)/n_traces),
                           n_traces))
        return feat_arr.mean(axis=1)

    def calc(self, traces, output_traces, Ntraces):
        gamma_factors = self.traces_to_features(traces, output_traces)
        errors = self.features_to_errors(gamma_factors, Ntraces)

        return errors
