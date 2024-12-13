import os
import netifaces
from distutils.util import strtobool

def set_comm_socket():
    for iface in netifaces.interfaces():
        if 'lo' in iface:
            continue
        iface_info = netifaces.ifaddresses(iface)
        if netifaces.AF_INET in iface_info:
            os.environ['NCCL_SOCKET_IFNAME'] = iface
            os.environ['GLOO_SOCKET_IFNAME'] = iface
            os.environ['SOPHON_SOCKET_IFNAME'] = iface
            os.environ['OMPI_MCA_btl_tcp_if_include'] = iface
            return iface
    raise RuntimeError("No interface found")

def environ_flag(var, reverse_default=False, default_override=None):
    '''
    if TPU is enabled, the default value of the flag is 1, otherwise 0
    '''
    disable_tpu = strtobool(os.environ.get('DS_DISABLE_TPU', '0'))
    if var == 'DS_DISABLE_TPU':
        return disable_tpu
    if default_override is not None:
        default_value = default_override
    else:
        default_value = disable_tpu if reverse_default else 1 - disable_tpu
    return strtobool(os.environ.get(var, str(default_value)))