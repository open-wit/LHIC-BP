#
# Automatically generated file, do not edit!
#

"""range Asymmetric Numeral System python bindings"""
from __future__ import annotations
import cbench.rans
import typing
import numpy
_Shape = typing.Tuple[int, ...]

__all__ = [
    "BufferedRansEncoder",
    "RansDecoder",
    "RansEncoder",
    "pmf_to_quantized_cdf",
    "pmf_to_quantized_cdf_np"
]


class BufferedRansEncoder():
    def __init__(self) -> None: ...
    def encode_with_indexes(self, arg0: list[int], arg1: list[int], arg2: list[list[int]], arg3: list[int], arg4: list[int]) -> None: ...
    def encode_with_indexes_np(self, arg0: numpy.ndarray[numpy.int32], arg1: numpy.ndarray[numpy.int32], arg2: numpy.ndarray[numpy.int32], arg3: numpy.ndarray[numpy.int32], arg4: numpy.ndarray[numpy.int32]) -> None: ...
    def flush(self) -> bytes: ...
    pass
class RansDecoder():
    def __init__(self) -> None: ...
    def decode_stream(self, arg0: list[int], arg1: list[list[int]], arg2: list[int], arg3: list[int]) -> list[int]: ...
    def decode_stream_np(self, arg0: numpy.ndarray[numpy.int32], arg1: numpy.ndarray[numpy.int32], arg2: numpy.ndarray[numpy.int32], arg3: numpy.ndarray[numpy.int32]) -> numpy.ndarray[numpy.int32]: ...
    def decode_with_indexes(self, arg0: str, arg1: list[int], arg2: list[list[int]], arg3: list[int], arg4: list[int]) -> list[int]: 
        """
        Decode a string to a list of symbols
        """
    def decode_with_indexes_np(self, arg0: str, arg1: numpy.ndarray[numpy.int32], arg2: numpy.ndarray[numpy.int32], arg3: numpy.ndarray[numpy.int32], arg4: numpy.ndarray[numpy.int32]) -> numpy.ndarray[numpy.int32]: 
        """
        Decode a string to a list of symbols
        """
    def set_stream(self, arg0: str) -> None: ...
    pass
class RansEncoder():
    def __init__(self) -> None: ...
    def encode_with_indexes(self, arg0: list[int], arg1: list[int], arg2: list[list[int]], arg3: list[int], arg4: list[int]) -> bytes: ...
    def encode_with_indexes_np(self, arg0: numpy.ndarray[numpy.int32], arg1: numpy.ndarray[numpy.int32], arg2: numpy.ndarray[numpy.int32], arg3: numpy.ndarray[numpy.int32], arg4: numpy.ndarray[numpy.int32]) -> bytes: ...
    pass
def pmf_to_quantized_cdf(pmf: list[float], precision: int = 16) -> list[int]:
    """
    Return quantized CDF for a given PMF
    """
def pmf_to_quantized_cdf_np(pmf: numpy.ndarray[numpy.float32], precision: int = 16) -> numpy.ndarray[numpy.uint32]:
    """
    Return quantized CDF for a given PMF
    """
