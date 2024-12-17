import os
from distutils.util import strtobool
if strtobool(os.environ.get('ENABLE_TPU_WORKAROUNDS', '1')):
    from . import workaround_impl