from ctypes import *
lib = CDLL('./build/libtest.so')
lib.myround(20)
