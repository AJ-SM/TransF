from scripts import tocknizer
from scripts import decoder
from scripts import BiLangModel


data,dictonary = decoder.cleanup()

conv = decoder.clean()
t = len(dictonary)

model = BiLangModel.BiLangModel(t)

dataset = tocknizer.generate(conv,dictonary)
x,y = tocknizer.genIO(dataset[:800],1,8)

y= y.view(1, -1)
x = x.view(1, -1)
print(x,y, "x,y pairs ")
print(f"Input shape: {x.shape}")
print(f"Target shape: {y.shape}")
logits, loss= model(x,y)
print(logits)
print(loss)

# print("For the vocab dict ", dictonary, "we are having total size of dict : ",t)
