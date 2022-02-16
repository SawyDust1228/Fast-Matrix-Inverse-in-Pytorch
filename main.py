
from OneColInv import OneColInv
import torch


seed = 1234567890
torch.manual_seed(seed=seed)
def getInverse(a):
    b = torch.mm(a.T, a)
    return b.inverse()

def testDel():
    l  = [4,3,2,1,0]
    a = torch.randn(5, 5)
    print(f"[a] : {a}")
    b = torch.matmul(a.T, a)
    b = b.inverse()
    print(f"[b]: {b}")
    a_del = a[:, :4]
    print(f"[a_del] : {a_del}")
    b_del = getInverse(a_del)
    print(f"[b_del] : {b_del}")


    answer = OneColInv(b, a, l, 4, add = False)
    print(f"[Answer] : {answer}")


def testAdd():
    a = torch.randn(5, 5)
    print(f"[a] : {a}")
    l = [i for i in range(a.shape[1])]
    b = torch.matmul(a.T, a)
    b = b.inverse()
    print(f"[b]: {b}")

    v = torch.randn(5, 1)
    print(f"[v] : {v}")
    a_add = torch.hstack([a, v])
    print(f"[a_add] : {a_add}")


    b_add = getInverse(a_add)
    print(f"[b_add] : {b_add}")


    answer = OneColInv(b, a_add, l, 5, add = True)
    print(f"[Answer] : {answer}")


testDel()