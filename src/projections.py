import torch
import numpy as np


class Projections:

  def __init__(self, projectionSettings, device):
    self.projectionSettings = projectionSettings
    if(self.projectionSettings['fourierEncoding']['isOn']):
      coordnMap = np.zeros((2, projectionSettings['fourierEncoding']['numTerms']))
      for i in range(coordnMap.shape[0]):
        for j in range(coordnMap.shape[1]):
          coordnMap[i, j] = np.random.choice([-1., 1.]) * \
                np.random.uniform(1./(2*projectionSettings['fourierEncoding']['maxRadius']),\
                                  1./(2*projectionSettings['fourierEncoding']['minRadius']))

      self.projectionSettings['fourierEncoding']['map'] = \
          torch.tensor(coordnMap).float().to(device)
  #-------------------------#

  def applyFourierEncoding(self, x):
    if(self.projectionSettings['fourierEncoding']['isOn']):
      c = torch.cos(2*np.pi*torch.matmul(x, self.projectionSettings['fourierEncoding']['map']))
      s = torch.sin(2*np.pi*torch.matmul(x, self.projectionSettings['fourierEncoding']['map']))
      x = torch.cat((c, s), axis=1)
    return x
  #--------------------------#

  def applyDensityProjection(self, x):
    if(self.projectionSettings['densityProjection']['isOn']):
      b = self.projectionSettings['densityProjection']['sharpness']
      nmr = np.tanh(0.5*b) + torch.tanh(b*(x-0.5))
      x = 0.5*nmr/np.tanh(0.5*b)
    return x
  #--------------------------#

  def applySymmetry(self, x):
    if(self.projectionSettings['symmetry']['YAxis']['isOn']):
      xv = self.projectionSettings['symmetry']['YAxis']['midPt'] + \
            torch.abs(x[:, 0] - self.projectionSettings['symmetry']['YAxis']['midPt'])
    else:
      xv = x[:, 0]
    if(self.projectionSettings['symmetry']['XAxis']['isOn']):
      yv = self.projectionSettings['symmetry']['XAxis']['midPt'] + \
            torch.abs(x[:, 1] - self.projectionSettings['symmetry']['XAxis']['midPt'])
    else:
      yv = x[:, 1]
    x = torch.transpose(torch.stack((xv, yv)), 0, 1)
    return x
  #--------------------------#
    