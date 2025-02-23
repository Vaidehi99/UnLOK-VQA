import torch

x = torch.load("/nas-ssd2/vaidehi/MMMEdit/data/model_preeedit.pt")
y = torch.load("/nas-ssd2/vaidehi/MMMEdit/data/model_posteedit.pt")
# z = torch.load("/nas-ssd2/vaidehi/MMMEdit/data/model_post_preeedit.pt")

print(x.keys()==y.keys())
# print(x.keys()==z.keys())

for key in x.keys():
    if x[key].shape!=y[key].shape:
            print("shape not same")
            print(key)

for key in x.keys():
    if x[key].shape==y[key].shape and x[key].dtype!=y[key].dtype:
            print("dtype not same")
            print(key)

for key in x.keys():
    if x[key].shape==y[key].shape and x[key].dtype==y[key].dtype:
        if not torch.equal(x[key], y[key]):
            print("dtype and shape same")
            print(key)