import struct
import numpy as np
from reedsolo import RSCodec

CRC_LENGTH = 4
PACKET_TEXT_LENGTH = 12
codec = RSCodec(CRC_LENGTH)

def binarray2barray(binarray):
    """ 01 np.array to bytesarray """
    bin_str = ''.join(str(i) for i in binarray)
    return bytearray(int(bin_str[i*8:i*8+8],2) for i in range(len(bin_str)//8))

def rsencode(text):
    """ string to bytesarray """
    return codec.encode(str.encode(text))

def rsdecode(barray):
    """ bytesarray to string """
    try:
        res = codec.decode(barray)[0].decode()
    except:
        res = barray[:PACKET_TEXT_LENGTH].decode(errors='replace')

    return res