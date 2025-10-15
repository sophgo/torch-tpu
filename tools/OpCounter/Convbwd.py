from .Utils import *

def get_bytes(shape: list):
    if len(shape) == 0: return 0
    bytes = 4 # f32
    for s in shape:
        bytes *= s
    return bytes

def get_di_dma_bytes(GradO_shape : list, GradI_shape : list, W_shape : list):
    # di = DCONV(grad_out, weight)
    GOBytes = get_bytes(GradO_shape)
    WBytes = get_bytes(W_shape)
    GIBytes = get_bytes(GradI_shape)
    return GOBytes + WBytes + GIBytes

def get_dw_dma_bytes(GradO_shape : list, I_shape : list, W_shape : list):
    # dw = CONV(grad_out, i)
    GOBytes = get_bytes(GradO_shape)
    WBytes = get_bytes(W_shape)
    GIBytes = get_bytes(I_shape)
    return GOBytes + WBytes + GIBytes

def get_db_dma_bytes(GradO_shape : list):
    # di = reduce_sum(GradO)
    GOBytes = get_bytes(GradO_shape)
    GBBytes = get_bytes([GradO_shape[1]])
    return GOBytes + GBBytes


def ConvBwdDmaTimeUs(GradO_shape : list, GradI_shape : list, W_shape : list,
                 DI_enable : bool, DW_enable : bool, DB_enable : bool):
    DI_DMA_BYTES = 0 if not DI_enable else get_di_dma_bytes(GradO_shape, GradI_shape, W_shape)
    DW_DMA_BYTES = 0 if not DW_enable else get_dw_dma_bytes(GradO_shape, GradI_shape, W_shape)
    DB_DMA_BYTES = 0 if not DB_enable else get_db_dma_bytes(GradO_shape)
    allDMA_Bytes = DW_DMA_BYTES + DI_DMA_BYTES + DB_DMA_BYTES
    time_us = (allDMA_Bytes / GB) / BW * US
    return time_us 

def get_di_flops(GradO_shape : list, GradI_shape : list, W_shape : list):
    # di = DCONV(grad_out, weight)
    # FLOPS = IN*IC*IW*IH * (OC*KH*KW) * 2
    IN,IC,IH,IW = GradI_shape[0], GradI_shape[1], GradI_shape[2], GradI_shape[3]
    OC = GradO_shape[1]
    KH,KW = W_shape[2], W_shape[3],
    return IN * IC * IH * IW * (OC*KH*KW) * 2

def get_dw_flops(GradO_shape : list, I_shape : list, W_shape : list):
    # dw = CONV(i, grad_out)
    # FLOPS = OC * IC * KH * KW * (ON * OH * OW) * 2
    OC,IC,KH,KW = W_shape[0], W_shape[1], W_shape[2], W_shape[3]
    ON,OH,OW    = GradO_shape[0], GradO_shape[2], GradO_shape[3] 
    return OC * IC * KH * KW * (ON * OH * OW) * 2

def get_db_flops(GradO_shape : list):
    # di = reduce_sum(GradO, dim=1)
    N, OC, OH, OW = GradO_shape[0], GradO_shape[1], GradO_shape[2], GradO_shape[3] 
    return OC * (OH * OW * N - 1)

def ConvBwdTiuTimeUs(GradO_shape : list, GradI_shape : list, W_shape : list,
                DI_enable : bool, DW_enable : bool, DB_enable : bool):
    DI_FLOPS = 0 if not DI_enable else get_di_flops(GradO_shape, GradI_shape, W_shape)
    DW_FLOPS = 0 if not DW_enable else get_dw_flops(GradO_shape, GradI_shape, W_shape)
    DB_FLOPS = 0 if not DB_enable else get_db_flops(GradO_shape)
    flops = DI_FLOPS + DW_FLOPS + DB_FLOPS
    return flops / F16_FLOPS * US
