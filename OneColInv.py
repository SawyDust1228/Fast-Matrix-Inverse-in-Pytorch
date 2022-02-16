from typing import List
import torch


def OneColInv(B, X0, col : List, posX0 : int, add = True):
    col.sort()
    X = X0[:, col]
    if add == True:
        col.append(posX0)
        col.sort()
        pos = col.index(posX0)
        v = X0[:, posX0]
        u1 = torch.matmul(X.T, v)
        u2 = torch.matmul(B, u1)
        F22inv = 1 / (torch.matmul(v.T, v) - torch.matmul(u1.T, u2))
        u3 = F22inv * u2
        F11inv = B + F22inv * torch.matmul(u2.T, u2)
        Bnew = torch.hstack([F11inv, -u3.reshape(F11inv.shape[0], -1)])
        Bnew = torch.vstack([Bnew, torch.hstack([-u3.T, F22inv])])

        l = []
        for i in range(0, pos):
            l.append(i)
        l.append(Bnew.shape[1]- 1)
        for i in range(pos, Bnew.shape[1]- 1):
            l.append(i)
        Bnew = Bnew[:, l]

        l.clear()    
        for i in range(0, pos):
            l.append(i)
        l.append(Bnew.shape[0]- 1)
        for i in range(pos, Bnew.shape[0]- 1):
            l.append(i)
        Bnew = Bnew[l, :]   

        # print(Bnew)
        return Bnew
    else:
        pos = col.index(posX0)

        l = []
        for i in range(0, pos):
            l.append(i)
        for i in range(pos + 1, B.shape[0]):
            l.append(i)
        l.append(pos)

        B = B[l,:]

        l.clear()
        for i in range(0, pos):
            l.append(i)
        for i in range(pos + 1, B.shape[1]):
            l.append(i)
        l.append(pos)

        B = B[:,l]

        F11inv = B[0 : B.shape[0] - 1, 0 : B.shape[1] - 1]
        d = B[B.shape[0] - 1, B.shape[1] - 1]
        u3 = -B[0 : B.shape[0] -1, B.shape[1] - 1]
        u2 = u3/d
        Bnew = F11inv - d * torch.matmul(u2, u2.T)
        return Bnew


