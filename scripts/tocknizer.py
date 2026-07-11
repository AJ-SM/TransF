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
    return "".join(outputD)

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


def genIO(data, start_index, context_size):
    # Grab a chunk for the input (Length = context_size)
    x = data[start_index: start_index + context_size]

    # Grab the exact same sized chunk, but shifted right by 1
    y = data[start_index + 1: start_index + context_size + 1]

    return x, y


# Test
# oupt = encode("Hellow there ")
# outpd = decode([41, 70, 77, 77, 80, 88, 1, 85, 73, 70, 83, 70, 1])
# print(outpd)
