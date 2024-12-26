import struct
import numpy as np
from reedsolo import RSCodec

CRC_LENGTH = 4
PACKET_TEXT_LENGTH = 12
codec = RSCodec(CRC_LENGTH)

def text2binarray(text):
    """ text to 01 np.array """
    bin_str =  ''.join(format(ord(char), '08b') for char in text)
    return np.array([int(s) for s in bin_str])

def binarray2text(binarray):
    """ 01 np.array to text """
    bin_str = ''.join(str(i) for i in binarray)
    return ''.join(chr(int(bin_str[i*8:i*8+8],2)) for i in range(len(bin_str)//8))

def shortarray_unpack(barray):
    """ bytesarray to np.array (short type, little endian) """
    tuple_of_shorts = struct.unpack('<'+'h'*(len(barray)//2), barray)
    return np.array(tuple_of_shorts)

def barray2binarray(barray):
    """ bytesarray to 01 np.array """
    bin_str = ''.join(format(i, '08b') for i in barray)
    return np.array([int(s) for s in bin_str]).astype(np.int32)

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