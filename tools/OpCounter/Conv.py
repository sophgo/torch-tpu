from .Utils import *

def get_bytes(shape: list):
    if len(shape) == 0: return 0
    bytes = 4 # f32
    for s in shape:
        bytes *= s
    return bytes

def get_out_dma_bytes(I_shape : list, W_shape : list, O_shape : list, Bias_shape: list):
    # O = CONV(I, weight)
    OBytes = get_bytes(O_shape)
    WBytes = get_bytes(W_shape)
    IBytes = get_bytes(I_shape)
    Bbytes = get_bytes(Bias_shape)
    return OBytes + WBytes + IBytes + Bbytes

def ConvDmaTimeUs(I_shape : list, W_shape : list, O_shape : list, Bias_shape: list):
    allDMA_Bytes = get_out_dma_bytes(I_shape, W_shape, O_shape, Bias_shape)
    time_us = (allDMA_Bytes / GB) / BW * US
    return time_us 

def get_conv_flops(I_shape : list, W_shape : list, O_shape : list, Bias_shape: list):
    # O = CONV(I, weight)
    # FLOPS = IN*IC*IW*IH * (OC*KH*KW) * 2
    IN,IC,IH,IW = I_shape[0], I_shape[1], I_shape[2], I_shape[3]
    OC, KH, KW  = W_shape[0], W_shape[2], W_shape[3]
    return IN * IC * IH * IW * (OC*KH*KW) * 2

def ConvTiuTimeUs(I_shape : list, W_shape : list, O_shape : list, Bias_shape: list):
    flops =get_conv_flops(I_shape, W_shape, O_shape, Bias_shape)
    return flops / F16_FLOPS * US
