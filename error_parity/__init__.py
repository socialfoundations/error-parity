from ._version import __version__, __version_info__
from .threshold_optimizer import RelaxedThresholdOptimizer

# For backwards compatibility we'll maintain the previous naming here
# NOTE: please use the current naming of `RelaxedThresholdOptimizer`
from .threshold_optimizer import RelaxedThresholdOptimizer as RelaxedEqualOdds
