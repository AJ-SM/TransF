import numpy as np
import torch

def encode(input, dictonary):
    outputE = []
    for item in input:
        idx = dictonary.index(item)
        outputE.append(idx)
    return outputE


def decode(input,dictonary):
    outputD = []
    for idx in input:
        value = dictonary[idx]
        outputD.append(value)
    return outputD

def generate(data,dictonary):
    pars = np.array(data).flatten()

    lst = []
    for msg in pars:
        ms = encode(msg,dictonary=dictonary)
        if(ms != []):
            lst.extend(ms)

    # lst1 = np.array(lst)
    # print(len(lst1))
    tens = torch.tensor(lst)
    return tens




# Test
# oupt = encode("Hellow there ")
# outpd = decode([41, 70, 77, 77, 80, 88, 1, 85, 73, 70, 83, 70, 1])
# print(outpd)
