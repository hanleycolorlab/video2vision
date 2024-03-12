import unittest
import warnings
# Raise all warnings as errors
warnings.filterwarnings('error')

from auto_operator_tests import *
from elementwise_tests import *
from io_tests import *
from notebook_tests import *
from operator_tests import *
from pipeline_tests import *
from warp_tests import *
from utils_tests import *


if __name__ == '__main__':
    unittest.main()
