import abc


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
    def features_to_error(self, features):
        """
        Function weights features/multiple errors into one final error fed
        back to the optimization algorithm
        """
        pass


class RMSMetric(Metric):
    def __init__(self):
        super(Metric, self).__init__()

    def traces_to_features(self, traces, output_traces):
        pass

    def features_to_error(self, features):
        pass
