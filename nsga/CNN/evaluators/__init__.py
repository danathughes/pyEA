## evaluators
##
## Different methods to evaluate CNN individuals

from .DummyEvaluator import DummyEvaluator
from .ProxyEvaluator import ProxyEvaluator
from .SingleNetworkEvaluator import SingleNetworkEvaluator
from .SingleNetworkEvaluatorKFold import SingleNetworkEvaluatorKFold
from .MultiNetworkEvaluator import MultiNetworkEvaluator
from .MultiNetworkEvaluatorKFold import MultiNetworkEvaluatorKFold
from .ThreadPoolEvaluator import ThreadPoolEvaluator


