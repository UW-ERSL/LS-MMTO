import torch
import numpy as np
from torch_sparse_solve import solve
#--------------------------#
class TorchSolver:
  def __init__(self, mesh, material, bc):
    self.mesh = mesh
    self.material = material
    self.bc = bc
    # self.D0elem = torch.tensor(self.material.getD0elemMatrix(self.mesh.meshProp))
    self.D0elem = self.material.getD0elemMatrix(self.mesh.meshProp).clone()
    self.f = torch.tensor(self.bc['force']).unsqueeze(0)

    V = np.zeros((mesh.meshProp['ndof'], mesh.meshProp['ndof']));
    V[self.bc['fixed'],self.bc['fixed']] = 1.
    V = torch.tensor(V[np.newaxis])
    indices = torch.nonzero(V).t()
    values = V[indices[0], indices[1], indices[2]] # modify this based on dimensionality
    penal = 1000000000000.
    self.fixedBCPenaltyMatrix = \
        penal*torch.sparse_coo_tensor(indices, values, V.size())
    
  #--------------------------#

  def assembleK(self, Eeffctv):
    sK = torch.einsum('e,ejk->ejk', Eeffctv, self.D0elem).flatten()
    Kasm = torch.sparse_coo_tensor(self.mesh.connectivity['nodeIdx'], sK, \
            (1, self.mesh.meshProp['ndof'], self.mesh.meshProp['ndof']))
    return Kasm;
  #--------------------------#
  def computeEffectiveModulus(self, E0, rho):
    penalRho = rho**self.material.matProp['penal']
    Eeffctv = torch.einsum('m,m->m',E0, penalRho)
    return Eeffctv
  #--------------------------#
  def solveFE(self, E0, rho):
    Eeffctv = self.computeEffectiveModulus(E0, rho)
    Kasm = self.assembleK(Eeffctv);
    K = (Kasm + self.fixedBCPenaltyMatrix).coalesce()
    u = solve(K, self.f).flatten()
    return u;
  #--------------------------#

  def computeCompliance(self, u):
    J = torch.einsum('i,i->i', u, self.f.view(-1)).sum()
    return J;
  #--------------------------#