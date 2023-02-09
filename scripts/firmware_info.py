import ctypes as ct
import sys
import zlib

"""
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef struct {
  char magic[4];       // 字符串"spfw"，sophon-firmware缩写，如果前四个字符不是这个，就按裸的firmware进行load, 其他信息全用0填充
  u8 major;            // 主版本号
  u8 minor;            // 次版本号
  u16 patch;           // patch版本号
  u32 date;            // 0xYYMMDD--
  u32 commit_hash;     // hash前8个十六进制，commit_id="AABBCCDD"
  u32 chip_id;         // 设备id
  u32 fw_size;         // firmware数据大小
  u32 fw_crc32;        // firmware实际数据校验码
  u8 fw_data[0];       // firmware实际数据
} firmware_header_t;
"""

FIRMWARE_MAGIC="sgfw"
class FirmwareHeader(ct.Structure):
    _fields_ = [
    ("magic", ct.c_byte*4),
    ("major", ct.c_ubyte),
    ("minor", ct.c_ubyte),
    ("patch", ct.c_ushort),
    ("date", ct.c_uint),
    ("commit", ct.c_uint),
    ("chip_id", ct.c_uint),
    ("fw_size", ct.c_uint),
    ("fw_crc32", ct.c_uint),
    ]
def print_fw_header(header):
    print("Header Size = {}".format(ct.sizeof(header)))
    print("Header Content:")
    lines = [
        "magic: {:s}".format(str(bytes(header.magic[:]))),
        "version: {}.{}.{}".format(header.major, header.minor, header.patch),
        "commit: {:x}".format(header.commit),
        "date: {:02x}-{:02x}-{:02x}".format((header.date>>24)&0xFF, (header.date>>16)&0xFF, (header.date>>8)&0xFF),
        "chip_id: {:#x}".format(header.chip_id),
        "crc32: {:#x}".format(header.fw_crc32),
    ]
    print("\n".join(lines))

def check_firmware(filename):
    with open(filename, "rb") as f:
        firmware_data = f.read()
        hlen = ct.sizeof(FirmwareHeader)
        header_data = firmware_data[0:hlen]
        bin_data = firmware_data[hlen:]
        header = ct.cast(header_data, ct.POINTER(FirmwareHeader)).contents
        print_fw_header(header)
        magic_str = (bytes(header.magic[:]))
        if bytes(header.magic[:]) != bytes(FIRMWARE_MAGIC, encoding="ascii"):
            print("Firmware {} has no header".format(filename))
            return False
        if header.fw_size != len(bin_data):
            print("firmware size is invalid: expect={}, real={}".format(header.fw_size, len(bin_data)))
            return False
        real_crc = zlib.crc32(bin_data)&0xFFFFFFFF
        expect_crc = header.fw_crc32
        if real_crc != expect_crc:
            print("firmware size is invalid: expect={:#x}, real={:#x}".format(expect_crc, real_crc))
            return False

if __name__ == "__main__":
    if len(sys.argv)==1:
        print("Usage:\n  python3 {} firmware_file".format(sys.argv[0]))
        sys.exit(-1)
    check_firmware(sys.argv[1])
