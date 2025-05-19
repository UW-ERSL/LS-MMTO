from networks import VariationalAutoencoder
import torch
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
import numpy as np

class MaterialEncoder:
  def __init__(self, trainingData, dataInfo, dataIdentifier, vaeSettings):
    self.trainingData, self.dataInfo = trainingData, dataInfo
    self.dataIdentifier = dataIdentifier
    self.vaeSettings = vaeSettings
    self.vaeNet = VariationalAutoencoder(vaeSettings)
  
  def loadAutoencoderFromFile(self, fileName):
    self.vaeNet.load_state_dict(torch.load(fileName))
    self.vaeNet.eval()
    
  def trainAutoencoder(self, numEpochs, klFactor, savedNet, learningRate):
    opt = torch.optim.Adam(self.vaeNet.parameters(), learningRate)
    convgHistory = {'reconLoss':[], 'klLoss':[], 'loss':[]}
    self.vaeNet.encoder.isTraining = True
    for epoch in range(numEpochs):
      opt.zero_grad()
      predData = self.vaeNet(self.trainingData)
      klLoss = klFactor*self.vaeNet.encoder.kl
      reconLoss =  ((self.trainingData - predData)**2).sum()
      loss = reconLoss + klLoss 
      loss.backward()
      convgHistory['reconLoss'].append(reconLoss)
      convgHistory['klLoss'].append(klLoss/klFactor) # save unscaled loss
      convgHistory['loss'].append(loss)
      opt.step()
      if(epoch%500 == 0):
        print('Iter {:d} reconLoss {:.3F} klLoss {:.3F} loss {:.3F}'.\
              format(epoch, reconLoss.item(), klLoss.item(), loss.item()))
    self.vaeNet.encoder.isTraining = False
    torch.save(self.vaeNet.state_dict(), savedNet)
    return convgHistory
  
  def plotLatent(self, ltnt1, ltnt2, plotHull, annotateHead, saveFileName):
    clrs = ['purple', 'green', 'orange', 'pink', 'yellow', 'black', 'violet', 'cyan', 'red', 'blue']
    colorcol = self.dataIdentifier['classID']
    ptLabel = self.dataIdentifier['name']
    autoencoder = self.vaeNet
    z = autoencoder.encoder.z.to('cpu').detach().numpy()
    fig, ax = plt.subplots()

    for i in range(np.max(colorcol)+1): 
      zMat = np.vstack((z[colorcol == i,ltnt1], z[colorcol == i,ltnt2])).T
      ax.scatter(zMat[:, 0], zMat[:, 1], c = 'black', s = 4)#clrs[i]
      if(i == np.max(colorcol)): #removed for last class TEST
        break # END TEST
      if(plotHull):
        hull = ConvexHull(zMat)
        cent = np.mean(zMat, 0)
        pts = []
        for pt in zMat[hull.simplices]:
            pts.append(pt[0].tolist())
            pts.append(pt[1].tolist())
  
        pts.sort(key=lambda p: np.arctan2(p[1] - cent[1],
                                        p[0] - cent[0]))
        pts = pts[0::2]  # Deleting duplicates
        pts.insert(len(pts), pts[0])
        poly = Polygon(1.1*(np.array(pts)- cent) + cent,
                       facecolor= 'black', alpha=0.1, edgecolor = 'black')
        poly.set_capstyle('round')
        plt.gca().add_patch(poly)
        ax.annotate(self.dataIdentifier['className'][i], (cent[0], cent[1]), size = 16)
    for i, txt in enumerate(ptLabel):
      if(annotateHead == False or ( annotateHead == True and  i<26)):
        ax.annotate(txt, (z[i,ltnt1], z[i,ltnt2]), size = 12)

  #   plt.axis('off')
    # ticks = [-1.5, -1., -0.5, 0., 0.5, 1., 1.5]
    # ticklabels = ['-1.5', '-1', '-0.5', '0','0.5', '1', '1.5']
    # plt.xticks(ticks, ticklabels, fontsize=18)
    # plt.yticks(ticks, ticklabels, fontsize=18)
    plt.xlabel('z{:d}'.format(ltnt1), size = 18)
    plt.ylabel('z{:d}'.format(ltnt2), size = 18)
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(saveFileName)
    
    return fig, ax
  
    
