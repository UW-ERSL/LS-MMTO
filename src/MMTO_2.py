from networks import MaterialNetwork, TopologyNetwork
from fea import TorchSolver
import torch
import torch.optim as optim
import numpy as np
from projections import Projections
import itertools
import matplotlib.pyplot as plt
from utilFuncs import to_np
import torch.nn.functional as F
from matplotlib import colors
import time
from matplotlib.colors import LinearSegmentedColormap
from mmaOptimize import optimize
torch.random.manual_seed(42)
import torch.nn.functional as F

class TopologyOptimizer:
  #-----------------------------#

  def __init__(self, mesh, material, bc, materialEncoder, constraints):
    self.materialEncoder = materialEncoder
    self.FE = TorchSolver(mesh, material, bc)
    self.constraints = constraints
    self.numConstraints = 2 if self.constraints['mass']['isOn'] and self.constraints['volume']['isOn'] else 1
    print('numConstraints', self.numConstraints)
    # self.numConstraints = 3 if self.constraints['mass']['isOn'] and self.constraints['volume']['isOn'] and self.constraints['distance']['isOn'] else 2
    self.mesh = mesh
    self.obj0 = 1.0
  ############################################################################################################

  def map_to_ellipse(self, arr):
    last_n = self.mesh.meshProp['numElems']
    arr_copy = np.copy(arr)
    z0 = arr_copy[last_n:2*last_n]
    z1 = arr_copy[2*last_n:]

    cx = self.constraints['distance']['center'][0]
    cy = self.constraints['distance']['center'][1]
    a = self.constraints['distance']['a']
    b = self.constraints['distance']['b']
    theta = self.constraints['distance']['theta']
    r = np.sqrt(z0)  # Uniform distribution in the ellipse
    phi = 2 * np.pi * z1
    # Unrotated ellipse coordinates

    x_e = r * a * np.cos(phi)
    y_e = r * b * np.sin(phi)

    # Apply rotation
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x_r = x_e * cos_theta - y_e * sin_theta
    y_r = x_e * sin_theta + y_e * cos_theta

    # Translate to ellipse center
    x = x_r + cx
    y = y_r + cy

    arr_copy[last_n:2*last_n] = x
    arr_copy[2*last_n:] = y

    return arr_copy
  
  #-----------------------------#
  def map_to_ellipse_torch(self, arr):
    last_n = self.mesh.meshProp['numElems']
    arr_copy = arr.clone()
    arr_copy.retain_grad()
    z0 = arr_copy[last_n:2 * last_n]
    z1 = arr_copy[2 * last_n:]

    cx = self.constraints['distance']['center'][0]
    cy = self.constraints['distance']['center'][1]
    a = self.constraints['distance']['a']
    b = self.constraints['distance']['b']
    theta = self.constraints['distance']['theta']

    # Uniform distribution in the ellipse
    r = torch.sqrt(z0)
    phi = 2 * torch.pi * z1

    # Unrotated ellipse coordinates
    x_e = r * a * torch.cos(phi)
    y_e = r * b * torch.sin(phi)

    # Apply rotation
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    x_r = x_e * cos_theta - y_e * sin_theta
    y_r = x_e * sin_theta + y_e * cos_theta

    # Translate to ellipse center
    x = x_r + cx
    y = y_r + cy

    # Update the input tensor in place
    arr_copy[last_n:2 * last_n] = x
    arr_copy[2 * last_n:] = y

    return arr_copy

  def getMaterialProperties(self, decoded):
    def unlognorm(x, scaleMax, scaleMin):
      return 10**(x*(scaleMax-scaleMin) + scaleMin)
    youngModulus = unlognorm(decoded[:,self.materialEncoder.dataInfo['ElasticModulus']['idx']], 
                              self.materialEncoder.dataInfo['ElasticModulus']['scaleMax'],
                              self.materialEncoder.dataInfo['ElasticModulus']['scaleMin'])
    physicalDensity = unlognorm(decoded[:,self.materialEncoder.dataInfo['MassDensity']['idx']],
                        self.materialEncoder.dataInfo['MassDensity']['scaleMax'],
                        self.materialEncoder.dataInfo['MassDensity']['scaleMin'])
    return youngModulus, physicalDensity

  #-----------------------------#
  def testMMA(self, optimizationParams = {'maxIters':30,'minIters':10,'relTol':0.05}):
    def objFunc(x):

      xT = torch.tensor(x).float()
      xT.requires_grad = True
      xTensor = self.map_to_ellipse_torch(xT)
      rho = xTensor[0:self.mesh.meshProp['numElems']]
      zD = xTensor[self.mesh.meshProp['numElems']:]
      zDomain = zD.view(2,-1).T

      def computeObjective(youngModulus, rho_design):
        uv_displacement = self.FE.solveFE(youngModulus, rho_design)
        objective = self.FE.computeCompliance(uv_displacement)
        return objective

      decoded = self.materialEncoder.vaeNet.decoder(zDomain)
      youngModulus, _ = self.getMaterialProperties(decoded) 
      objective = computeObjective(youngModulus, rho)

      # print('objective', objective)
      # print('------------')
      objective.backward()
      dJ = xTensor.grad.detach().numpy()
      J = objective.detach().numpy()/self.obj0


      return J, dJ
    ############################################################################################################
    #  Can combine the conFunc and objFunc
    ############################################################################################################
    def conFunc(x):      
      xT = torch.tensor(x).float()
      xT.requires_grad = True
      xTensor = self.map_to_ellipse_torch(xT)
      rho = xTensor[0:self.mesh.meshProp['numElems']]
      zDomain = xTensor[self.FE.mesh.meshProp['numElems']:].reshape(2,-1).T

      if self.constraints['volume']['isOn']:
        def computeVolumeInfo(rho):
          volumeFraction = torch.mean(rho)
          volConstraint = ((volumeFraction/self.constraints['volume']['desiredVolumeFraction']) - 1.0)
          return volumeFraction, volConstraint

        volumeFraction, volConstraint = computeVolumeInfo(rho)
        volConstraint.backward(retain_graph=True)

        dcVol = xTensor.grad.detach().numpy()[np.newaxis]
        c = volConstraint.detach().numpy()[np.newaxis]
        dc = dcVol
        print('volumeFraction', volumeFraction)

############################################################################################################
      if self.constraints['mass']['isOn']:
        if xTensor.grad is not None:
          xTensor.grad.zero_()

        def computeMassInfo(massDensity, rho): # TODO: Add if condition for mass constraint
          totalMass = torch.einsum('m,m->m',massDensity, rho).sum()
          massConstraint = ((totalMass/self.constraints['mass']['maxMass']) - 1.0)
          return totalMass, massConstraint


        decoded = self.materialEncoder.vaeNet.decoder(zDomain)
        _, massDensity = self.getMaterialProperties(decoded) 
        totalMass, massConstraint = computeMassInfo(massDensity, rho)
        # print('totalMass', totalMass)
        print('massConstraint', massConstraint)
        massConstraint.backward()
        dcMass = xTensor.grad.detach().numpy()[np.newaxis]

        c = massConstraint.detach().numpy()[np.newaxis]
        dc = dcMass
############################################################################################################
      if self.constraints['distance']['isOn']:
        if xTensor.grad is not None:
          xTensor.grad.zero_()


      if self.constraints['volume']['isOn'] and self.constraints['mass']['isOn']:
        c = np.concatenate((volConstraint.detach().numpy()[np.newaxis], massConstraint.detach().numpy()[np.newaxis])).reshape(-1,1)
        dc = np.concatenate((dcVol, dcMass))
        print('volume constraint', volConstraint.detach().numpy()[np.newaxis])
        print('mass constraint', massConstraint.detach().numpy()[np.newaxis])

      if self.constraints['volume']['isOn'] and self.constraints['distance']['isOn']:
        c = np.concatenate((volConstraint.detach().numpy()[np.newaxis], sdf_constraint.detach().numpy()[np.newaxis])).reshape(-1,1)
        dc = np.concatenate((dcVol, dcSDF))
        print('volume constraint', volConstraint.detach().numpy()[np.newaxis])
        print('distance constraint', sdf_constraint.detach().numpy()[np.newaxis])

      if self.constraints['volume']['isOn'] and self.constraints['mass']['isOn'] and self.constraints['distance']['isOn']:
        c = np.concatenate((volConstraint.detach().numpy()[np.newaxis], massConstraint.detach().numpy()[np.newaxis], sdf_constraint.detach().numpy()[np.newaxis])).reshape(-1,1)
        dc = np.concatenate((dcVol, dcMass, dcSDF))
        print('volume constraint', volConstraint.detach().numpy()[np.newaxis])
        print('mass constraint', massConstraint.detach().numpy()[np.newaxis])
        print('distance constraint', sdf_constraint.detach().numpy()[np.newaxis])

      return c, dc
    ############################################################################################################
    x0 = 1*np.ones(self.FE.mesh.meshProp['numElems']*3)
    J, dJ = objFunc(x0)
    c, dc = conFunc(x0)

    self.obj0 = J.copy()
    filterRadius = 3
    H, Hs = self.mesh.computeFilter(filterRadius)
    ft = {'type':1, 'H':H, 'Hs':Hs}
    res = optimize(self.mesh, optimizationParams, ft, \
             objFunc, conFunc, self.numConstraints)
    return res
#-----------------------------#
