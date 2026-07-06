from scripts import decoder

data = decoder.cleanup()

dictonary = sorted(list(set(data)))
t = len(dictonary)
print("For the vocab dict ", dictonary, "we are having total size of dict : ",t)
