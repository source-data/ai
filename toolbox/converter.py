# -*- coding: utf-8 -*-
#T. Lemberger, 2018


import argparse
import torch
import os
from functools import lru_cache
from copy import copy
from typing import List

NBITS = int(os.getenv('NBITS'))

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class TStringTypeError(Error):
    """
    Exception raised when TString is initialized with something else than a str, a StringList or a Tensor.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, x):
        super().__init__(f"Wrong type: only str, StringList or torch.Tensor allowed whereas {x} is of type {type(x)}")

class HeterogenousWordLengthError(Error):
    """
    Exception raised when StringList is initialized with a list of words that are of various length.
    """
    def __init__(self, message):
        super().__init__(message)

class ConcatenatingTStringWithUnequalDepthError(Error):
    """
    Exception raised when 2 TStrings with different depth (number of examples) are concatenated.
    """
    def __init__(self, d1, d2):
        super().__init__(f"Depths of the 2 concatenated TString are not identical ({d1} != {d2}).")

class RepeatError(Error):
    """
    Exception raised when TString.repeat(0) is called. Rather than returning an empty tensor, raising an exception is preferred as repeating zero times is probably unintended.
    """
    def __init__(self, N):
        super().__init__(f"repeat with N={N} as argument is not allowed. N must be int and N > 0")

class Converter():
    """
    Conversion operations between strings and tensors.
    """
    def __init__(self,):
        self.dtype = dtype

    def encode(self, input_string:str) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, t: torch.Tensor) -> str:
        raise NotImplementedError


class ConverterNBITS(Converter):


    def __init__(self, nbits):
        self.nbits = nbits

    @lru_cache(maxsize=1024)
    def code2bits(self, code):
        bits = torch.Tensor([code >> i & 1 for i in range(self.nbits)])
        return bits

    def encode(self, input_string: str, dtype:torch.dtype=torch.float) -> torch.Tensor:
        """
        Encodes an input string into a 3D tensor.

        Args:
            input_string (str): string to convert

        Returns:
            (torch.Tensor): 3D tensor 1 x NBITS x L, (1 example x NBITS bits x L characters) representing characters as NBITS features
        """

        L = len(input_string)
        t = torch.Tensor(0)
        if L > 0:
            t = torch.zeros(1, self.nbits, L, dtype=dtype)
            for i in range(L):
                code = ord(input_string[i])
                try:
                    t[0, : , i] = self.code2bits(code)
                except IndexError:
                    t[0, : , i] = self.code2bits(ord('?'))
        return t

    def decode(self, t: torch.Tensor) -> str:
        """
        Decodes a 3D tensor into a unicode string.

        Args:
            t (torch.Tensor): 3D tensor 1xNBITSxL (1 example x NBITS bits x L characters) representing characters as NBITS features
        
        Returns:
            (str): resulting string
        """

        L = t.size(2)
        str = ""
        for i in range(L):
            code = 0
            for j in range(self.nbits):
                bit = t[0, j, i]
                try:
                    bit = int(bit)
                except Exception as e:
                    print(f"{e} in converter module.")
                    print(e)
                    bit = 0
                code += bit*(2**j)
            try:
                str += chr(code) #python 2: unichr()
            except ValueError:
                str += '?'
            except OverflowError:
                print("Error: code too large", code)
                print(type(code))
                print(bin(code))
                print(t[0, : , i].view(-1))
                str += '?'
        return str

CONVERTER = ConverterNBITS(NBITS)

class StringList:
    """
    A class to represent a list of strings that MUST all be of identical length.
    The identical length lends itself for straight forward conversion to a tensor once encoded.

    Raises:
        HeterogenousWordLengthError: when the strings are of heterogenous lengths or all empty.
    """


    def __init__(self, x: List[str]=[]):
        self._N = 0
        self._L = 0
        self._list = []
        if x:
            self._N = len(x)
            x_0 = len(x[0])
            total = len("".join([e for e in x]))
            if total != self._N * x_0 or x_0 == 0:
                raise HeterogenousWordLengthError(f"{x}: all the words have to have the same length in a StringList so that they can be stacked into same tensor when converted.")
            self._L = x_0
            self._list = x

    @property
    def words(self) -> List:
        return self._list

    def __len__(self) -> int:
        return self._L

    @property
    def depth(self) -> int:
        return self._N

    def __add__(self, x: 'StringList') -> 'StringList':
        result = StringList([a + b for a, b in zip(self.words, x.words)])
        return result

    def __getitem__(self, i: int) -> str:
        result = self.words[i]
        return result

    def __repr__(self) -> str:
        result = " | ".join(self.words)
        return result

    def __nonzero__(self) -> bool:
        return len(self) > 0

    def clone(self) -> 'StringList':
        cloned = StringList(copy(self.words))
        return cloned


class TString:
    '''
    Class to represent strings simultaneously as Tensor as a list of str. 
    The number of feature used to encode one character is NBITS.
    A list of N strings of homogenous length L is encoded into a  N x NBITS x L3D Tensor.

    Args:
        x: either a list of strings, in in which case it is converted into the corresonding Tensor;
        or a Tensor, in which case it does not need conversion but needs to be 3D with N x NBITS x L.
        If no argument is provided, TString is initialized with an empty string.

    Methods:
        toStringList: string list representation of TString
        __len__(): length with len(TString) and returns int
        __add__(TString): concatenates TString and returns a TString; allow operation like tstring_1 + tstring_2
        __getitem(i): gets the i-th element of each string of the list and of the underlying tensor and returns a TString; allows to slice with tstring[start:stop]
        repeat(N): repeats the TString N time
        tensor: returns the 3D (N x NBITS x L) torch.Tensor representation of the encoded list of strings
    '''

    def __init__(self, x = StringList(), dtype:torch.dtype=torch.float):
        self.dtype = dtype
        self.nbits = CONVERTER.nbits
        self._t = torch.zeros([], dtype=self.dtype) # empty tensor
        self._s = []
        self._L = 0 # length
        self._N = 0 # number of strings in the list, or depth
        if isinstance(x, str):
            x = StringList([x]) if x else StringList()
        if isinstance(x, torch.Tensor):
            assert x.dim() == 3 and x.size(1) == self.nbits
            self._t = x
            self._L = self._t.size(2)
            self._N = self._t.size(0)
            for i in range(self.depth):
                s = CONVERTER.decode(x[i:i+1, :, : ])
                self._s.append(s)
        elif isinstance(x, StringList):
            if x:
                self._s = x.words
                self._N = x.depth
                t_list = [CONVERTER.encode(ex, dtype=dtype) for ex in x]
                self._t = torch.cat(t_list, 0)
                self._L = self._t.size(2)
                assert self._N == self._t.size(0)
        else:
            raise TStringTypeError(x)



    def __str__(self):
        return "\n".join(self._s)
    
    def toStringList(self) -> StringList:
        return StringList(self._s) # slight overhead due to checks of homogenous length

    @property
    def words(self) -> List[str]:
        return self._s # more direct

    @property
    def stringList(self) -> StringList:
        return self.toStringList()

    def __len__(self) -> int:
        return self._L

    @property
    def depth(self) -> int:
        return self._N

    def __add__(self, x: 'TString') -> 'TString':
        # overwrites tensor adding operator to make it a tensor concatenation like for strings
        # what to do when both are empty?
        if len(x) == 0:
            return self # or should it return a cloned self?
        elif len(self) == 0:
            return x # or should it return a cloned x?
        else:
            try:
                assert self.depth == x.depth
            except AssertionError:
                raise ConcatenatingTStringWithUnequalDepthError(self.depth, x.depth)
            concatenated = TString(dtype=self.dtype)
            concatenated._t = torch.cat((self.tensor, x.tensor), 2)
            concatenated._s = [a + b for a, b in zip(self.words, x.words)]
            concatenated._L = len(self) + len(x)
            concatenated._N = self._N
            return concatenated

    def __getitem__(self, i: int) -> 'TString':
        if len(self) == 0:
            return TString()
        else:
            item = TString(dtype=self.dtype)
            item._s = [s[i] for s in self.words]
            item._t = self.toTensor()[ : , : , i]
            item._L = 1
            item._N = self._N
            return item

    def repeat(self, N: int) -> 'TString':
        if N == 0 or not isinstance(N, int): 
            raise RepeatError(N)
        if N == 1:
            return self
        else:
            repeated = TString(dtype=self.dtype)
            repeated._t = self.toTensor().repeat(1, 1, N) 
            repeated._s = [w * N for w in self.words]
            repeated._L = len(self) * N
            repeated._N = self._N
        return repeated

    @property
    def tensor(self) -> torch.Tensor:
        return self._t

    def toTensor(self) -> torch.Tensor: # legacy method
        return self._t


def self_test(input_string: str):
    encoded = CONVERTER.encode(input_string)
    decode_encoded = CONVERTER.decode(encoded)
    assert input_string == decode_encoded, f"{input_string}<>{decode_encoded}"
    print("the decoded of the encoded:", TString(TString(StringList([input_string, input_string])).tensor).toStringList())

    a = TString("a")
    b = TString("b")
    assert (a + b).toStringList().words == StringList(["ab"]).words
    print(TString(StringList(['12345', '67890', 'abcde'])))
    print("It seems it works!")

def main():
    # more systematic tests in test.test_converter
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description="conversion", formatter_class=formatter_class)
    parser.add_argument('input_string', nargs='?', default= "αβγ∂this is so ☾😎 😎 L ‼️" + u'\uE000', help="The string to convert")
    args = parser.parse_args()
    input_string = args.input_string
    self_test(input_string)

if __name__ == "__main__":
    main()
