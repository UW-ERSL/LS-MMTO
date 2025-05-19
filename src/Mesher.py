import numpy as np
import numpy.matlib
import torch
import matplotlib.pyplot as plt

class RectangularGridMesher:
  #--------------------------#
  def __init__(self, meshProp, physics):

    self.meshProp = meshProp;
    self.physics = physics

    self.meshProp['numElems'] = meshProp['nelx']*meshProp['nely']
    self.meshProp['numNodes'] = (meshProp['nelx']+1)*(meshProp['nely']+1)
    self.meshProp['elemArea'] = self.meshProp['elemSize'][0]*\
    self.meshProp['elemSize'][1]*torch.ones((self.meshProp['numElems']))
    self.meshProp['elemVolume'] = meshProp['thickness']*self.meshProp['elemArea']
    self.meshProp['totalMeshArea'] = torch.sum(self.meshProp['elemArea'] )
    self.meshProp['totalMeshVolume'] = meshProp['thickness']*self.meshProp['totalMeshArea']
    self.meshProp['nodesPerElem'] = 4
    self.meshProp['ndof'] = self.meshProp['dofsPerNode']*self.meshProp['numNodes']
    
    edofMat, nodeIdx, elemNodes, nodeXY, bb = self.getMeshStructure();
    self.meshProp['nodeXY'] = nodeXY
    self.meshProp['boundBox'] = bb
    self.connectivity = {'edofMat':edofMat, 'elemNodes':elemNodes, 'nodeIdx':nodeIdx}

    self.meshProp['elemCenters'] = self.generatePoints();
    self.topFig, self.topAx = plt.subplots()
    self.H, self.Hs = self.computeFilter(1.5)
  #--------------------------#
  def getMeshStructure(self):
    # returns edofMat: array of size (numElemsX8) with
    # the global dof of each elem
    # idx: A tuple informing the position for assembly of computed entries
    n = self.meshProp['dofsPerNode']*self.meshProp['nodesPerElem']
    edofMat=np.zeros((self.meshProp['nelx']*self.meshProp['nely'],n),dtype=int)
    if(self.physics == 'structural'):
      for elx in range(self.meshProp['nelx']):
        for ely in range(self.meshProp['nely']):
          el = ely+elx*self.meshProp['nely']
          n1=(self.meshProp['nely']+1)*elx+ely
          n2=(self.meshProp['nely']+1)*(elx+1)+ely
          edofMat[el,:]=np.array([2*n1+2, 2*n1+3, 2*n2+2,\
                          2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])
                
    elif(self.physics == 'thermal'): # as in thermal
      nodenrs = np.reshape(np.arange(0, self.meshProp['ndof']), \
                     (1+self.meshProp['nelx'], 1+self.meshProp['nely'])).T
      edofVec = np.reshape(nodenrs[0:-1,0:-1]+1, \
                           self.meshProp['numElems'],'F');
      edofMat = np.matlib.repmat(edofVec,4,1).T + \
            np.matlib.repmat(np.array([0, self.meshProp['nely']+1,\
                                       self.meshProp['nely'], -1]),\
                             self.meshProp['numElems'],1);
    
    iK = tuple(np.kron(edofMat,np.ones((n,1))).flatten().astype(int))
    jK = tuple(np.kron(edofMat,np.ones((1,n))).flatten().astype(int))
    bK = tuple(np.zeros((len(iK))).astype(int)) #batch values
    nodeIdx = [bK,iK,jK]


    elemNodes = np.zeros((self.meshProp['numElems'],\
                          self.meshProp['nodesPerElem']));
    for elx in range(self.meshProp['nelx']):
      for ely in range(self.meshProp['nely']):
        el = ely+elx*self.meshProp['nely']
        n1=(self.meshProp['nely']+1)*elx+ely
        n2=(self.meshProp['nely']+1)*(elx+1)+ely
        elemNodes[el,:] = np.array([n1+1, n2+1, n2, n1])
    bb = {}
    bb['xmin'],bb['xmax'],bb['ymin'],bb['ymax'] = \
        0., self.meshProp['nelx']*self.meshProp['elemSize'][0],\
        0., self.meshProp['nely']*self.meshProp['elemSize'][1]
        
    nodeXY = np.zeros((self.meshProp['numNodes'], 2))
    ctr = 0;
    for i in range(self.meshProp['nelx']+1):
      for j in range(self.meshProp['nely']+1):
        nodeXY[ctr,0] = self.meshProp['elemSize'][0]*i;
        nodeXY[ctr,1] = self.meshProp['elemSize'][1]*j;
        ctr += 1;
            
    # plt.ion();
    # plt.clf();
    # plt.plot(nodeXY[:,0],nodeXY[:,1],'o', markersize=2, color='k')
    # for i, (x, y) in enumerate(nodeXY):
    #   plt.text(x, y, str(i+1), fontsize=6, ha='center', va='center', color='blue')
    return edofMat, nodeIdx, elemNodes, nodeXY, bb
  #--------------------------#

  def generatePoints(self, res=1):
    # args: Mesh is dictionary containing nelx, nely, elemSize...
    # res is the number of points per elem
    # returns an array of size (numpts X 2)
    xy = np.zeros((res**2*self.meshProp['numElems'], self.meshProp['ndim']))
    ctr = 0
    for i in range(res*self.meshProp['nelx']):
      for j in range(res*self.meshProp['nely']):
        xy[ctr, 0] = self.meshProp['elemSize'][0]*(i + 0.5)/res
        xy[ctr, 1] = self.meshProp['elemSize'][1]*(j + 0.5)/res
        ctr += 1
    return xy
  #--------------------------#
  def plotFieldOnMesh(self, field, isNodal, res, titleStr, clrmap = 'jet', vmin = 0., vmax = 1.):
    plt.ion();
    plt.clf();
    k = 0
    if(isNodal):
      k = 1
    plt.imshow(np.flipud(field.reshape((res*(self.meshProp['nelx']+k),\
                                         res*(self.meshProp['nely']+k))).T), \
                                       cmap= clrmap, interpolation='none',\
                                         vmin = vmin, vmax = vmax)
    self.topFig.canvas.draw()
    
    plt.axis('Equal')
    plt.grid(False)
    plt.title(titleStr)
    plt.pause(0.01)
  #--------------------------#
  def computeFilter(self, rmin):
    nelx, nely = self.meshProp['nelx'], self.meshProp['nely']
    H = np.zeros((nelx*nely,nelx*nely));

    for i1 in range(nelx):
        for j1 in range(nely):
            e1 = (i1)*nely+j1;
            imin = max(i1-(np.ceil(rmin)-1),0.);
            imax = min(i1+(np.ceil(rmin)),nelx);
            for i2 in range(int(imin), int(imax)):
                jmin = max(j1-(np.ceil(rmin)-1),0.);
                jmax = min(j1+(np.ceil(rmin)),nely);
                for j2 in range(int(jmin), int(jmax)):
                    e2 = i2*nely+j2;
                    H[e1, e2] = max(0.,rmin-\
                                       np.sqrt((i1-i2)**2+(j1-j2)**2));

    Hs = np.sum(H,1);
    return H, Hs;
