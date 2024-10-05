import torch
from torch import nn
from functools import partial
from functorch import make_fx

data0 = (torch.LongTensor([0, 1, 2, 3]), torch.LongTensor([0]))
data1 = (torch.LongTensor([0, 2, 3]), torch.LongTensor([0]))


def try_(sparse):
    e = nn.EmbeddingBag(10, 2, mode="sum", sparse=sparse)

    def fwd(model, data):
        out = model(*data)
        grad = nn.MSELoss()(out, torch.Tensor([[0, 0]]))
        grad.backward()
        return grad
    gm = make_fx(partial(fwd, e))(data0)

    # Call the model
    print(gm(data0))
    print(gm(data1))


try_(sparse=False)
print("---------")
try_(sparse=True)


# print(gm.graph)
# print(list(gm.graph.nodes))
# list(gm.graph.nodes)[-1].replace_input_with(
#     list(gm.graph.nodes)[11], list(gm.graph.nodes)[14])
# gm.recompile()
