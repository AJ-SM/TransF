import numpy as np
import os
from dotenv import load_dotenv

env = load_dotenv()
chunk = os.getenv("CHUNK")
data= open("D:\\Storeage-1\\MlNewBorn\\TransF\\res\\data.txt","r",encoding="utf-8").read()

def cleanup(chunk=chunk,data=data):
    if chunk == None:
        raise ValueError("Chunk not Found Pls enter the chunk ")
    if int(chunk) == 1:
        clean(data)
        return data
    else:
        print('I Am SORRY BABU ')


def clean(data):
    clean_d = []
    text = data.split('\n')
    for i in text:
        net = i.split(':')[-1]

        if net != ' <Media omitted>':
            print(net)
            clean_d.append(net)
    # print(text)