from scripts import tocknizer
from scripts import decoder

data,dictonary = decoder.cleanup()


conv = decoder.clean()
t = len(dictonary)

dataset = tocknizer.generate(conv,dictonary)
print(dataset.shape)
# print("For the vocab dict ", dictonary, "we are having total size of dict : ",t)
