# from scripts import tocknizer
# from scripts import decoder
# from scripts import BiLangModel
# import torch
#
# data,dictonary = decoder.cleanup()
#
# conv = decoder.clean()
# t = len(dictonary)
#
# model = BiLangModel.BiLangModel(t)
#
# dataset = tocknizer.generate(conv,dictonary)
# x,y = tocknizer.genIO(dataset[:800],1,8)
#
# y= y.view(1, -1)
# x = x.view(1, -1)
#
# # Debuging prints
# # print(x,y, "x,y pairs ")
# # print(f"Input shape: {x.shape}")
# # print(f"Target shape: {y.shape}")
#
# logits, loss= model(x,y)
# # print(logits[:,-1])
# idx = torch.zeros((1,1), dtype=torch.long)
#
#
# # print("For the vocab dict ", dictonary, "we are having total size of dict : ",t)
#
#
# def trainer(model,dataset, epoch ):
#     optimizer= torch.optim.AdamW(model.parameters(),lr=0.001)
#     x,y=tocknizer.genIO(dataset,1,100)
#     y= y.view(1, -1)
#     x = x.view(1, -1)
#     for epoch in range(epoch):
#         logits,loss = model(x,y)
#         optimizer.zero_grad(set_to_none=True)
#         loss.backward()
#         optimizer.step()
#         print(loss.item())
#
# print("         Before Preds             ")
# print(tocknizer.decode(model.generate(idx,max_tocken_len=300)[0].tolist(),dictonary=dictonary))
# print("         After Preds             ")
# trainer(model,dataset,3)
# print(tocknizer.decode(model.generate(idx,max_tocken_len=300)[0].tolist(),dictonary=dictonary))
# print("         Done Preds             ")

from scripts import tocknizer
from scripts import decoder
from scripts import BiLangModel
import torch

data,dictonary = decoder.cleanup()

conv = decoder.clean()
t = len(dictonary)

model = BiLangModel.BiLangModel(t)

dataset = tocknizer.generate(conv,dictonary)
x,y = tocknizer.genIO(dataset[:800],1,8)

y= y.view(1, -1)
x = x.view(1, -1)

# Debuging prints
# print(x,y, "x,y pairs ")
# print(f"Input shape: {x.shape}")
# print(f"Target shape: {y.shape}")




# Bilang Training
logits, loss= model(x,y)
idx = torch.zeros((1,1), dtype=torch.long)
# print(logits[:,-1])
# print("For the vocab dict ", dictonary, "we are having total size of dict : ",t)
print("         Before Preds             ")
print(tocknizer.decode(model.generate(idx,max_tocken_len=300)[0].tolist(),dictonary=dictonary))
print("         After Preds             ")
BiLangModel.trainer(model,dataset,3)
print(tocknizer.decode(model.generate(idx,max_tocken_len=300)[0].tolist(),dictonary=dictonary))
print("         Done Preds             ")
