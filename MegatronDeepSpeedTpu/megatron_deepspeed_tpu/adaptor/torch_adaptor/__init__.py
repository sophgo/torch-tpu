import os
from distutils.util import strtobool
if strtobool(os.environ.get('DS_USE_HOST_GLOO', "0")):
    from . import distributed_gloo
else:
    from . import distributed
from . import cuda