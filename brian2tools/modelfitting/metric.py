import abc
from numpy import shape, array, sum, square, reshape


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
