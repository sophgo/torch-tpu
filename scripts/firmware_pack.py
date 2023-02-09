import ctypes as ct
import zlib
import sys
import os
import struct
import time

from firmware_info import (FirmwareHeader, FIRMWARE_MAGIC, print_fw_header)


def pack_firmware(filename, outdir):

    header = FirmwareHeader()
    header.magic = (ct.c_byte*4)(*bytes(FIRMWARE_MAGIC, encoding="ascii"))

    version = os.environ.get("FIRMWARE_VERSION", "0.0.0")
    major, minor, patch = [int(v) for v in version.split(".") ]
    header.major = type(header.major)(major)
    header.minor = type(header.minor)(minor)
    header.patch = type(header.patch)(patch)


    chip_id_str = os.environ.get("FIRMWARE_CHIPID", "0000").lower()

    if chip_id_str == "1684x":
        chip_id_str = "1686"
    header.chip_id = type(header.chip_id)(int(chip_id_str, base=16))

    commit_id = os.environ.get("FIRMWARE_COMMIT", "00000000")
    if commit_id == "00000000":
        result = os.popen("git rev-parse HEAD")
        commit_id = result.read().strip()
        commit_id = commit_id[0:8]
    assert len(commit_id)==8
    header.commit = type(header.commit)(int(commit_id, base=16))

    date_str = os.environ.get("FIRMWARE_DATE", None)
    if not date_str:
        date_str = time.strftime("%y%m%d")
    full_data_str = date_str + "00"
    header.date= type(header.date)(int(full_data_str, base=16))

    with open(filename, "rb") as f:
        firmware_data=f.read()

    header.fw_size = type(header.fw_size)(len(firmware_data))

    crc32 = zlib.crc32(firmware_data)&0xFFFFFFFF
    header.fw_crc32 = type(header.fw_crc32)(crc32)

    header_len = ct.sizeof(header)
    byte_array = ct.cast(ct.pointer(header), ct.POINTER(ct.c_ubyte*header_len))
    header_content = bytes(byte_array.contents[:])

    firmware_suffix = "_v{}-{}-{}".format(version, commit_id, date_str)
    print_fw_header(header)

    full_path = os.path.join(outdir, os.path.basename(firmware_file+firmware_suffix))

    with open(full_path, "wb") as f:
        f.write(header_content)
        f.write(firmware_data)
    print("Pack {} to {} successfully".format(firmware_file, full_path))

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage:\n  python3 {} firmware".format(sys.argv[0]))
        sys.exit(-1)
    firmware_file = sys.argv[1]
    outdir = os.path.dirname(firmware_file)
    if len(sys.argv) == 3:
        outdir = sys.argv[2]
    pack_firmware(sys.argv[1], outdir)
