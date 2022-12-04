#!/usr/bin/env python3
import ctypes as ct
import os
import sys
import argparse

def parseArgs():
  parser = argparse.ArgumentParser(description="bmneto", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--firmware', type=str, required=True, help='firmware path')
  parser.add_argument('--firmware_tcm', type=str, default=None, required=False, help='tcm firmware path')
  parser.add_argument("--device_id", type=int, default=-1, help="device which firmware will be loaded on, default 0")
  args = parser.parse_args()
  if args.device_id < 0:
    args.device_id = os.environ.get("DEVICE_ID", 0)
  print("args:", args)
  return args

def load_firmware(args):
  device_id, ddr_firmware, tcm_firmware= args.device_id, args.firmware, args.firmware_tcm
  assert os.path.exists(ddr_firmware)

  lib = ct.cdll.LoadLibrary("libbmlib.so")

  handle = ct.c_void_p(0)
  bm_dev_request = lib.bm_dev_request
  bm_dev_request.restype = ct.c_int
  status = bm_dev_request(ct.byref(handle), ct.c_int32(device_id))
  assert status == 0

  c_ddr_firmware =ct.c_char_p(bytes(ddr_firmware, encoding="utf-8"))

  c_tcm_firmware = ct.c_char_p(0)
  if tcm_firmware:
    c_tcm_firmware =ct.c_char_p(bytes(tcm_firmware, encoding="utf-8"))
  bm_load_firmware = lib.bm_load_firmware
  bm_load_firmware.restype = ct.c_int
  status = bm_load_firmware(handle, c_tcm_firmware, c_ddr_firmware)
  assert status == 0, "failed to load firmware {} on device_id={}".format(args.firmware, args.device_id)

  bm_dev_free = lib.bm_dev_free
  bm_dev_free.restype=None
  bm_dev_free(handle)
  print("succeed to load firmware {} on device_id={}".format(args.firmware, args.device_id))

if __name__ == '__main__':
  args = parseArgs()
  load_firmware(args)

