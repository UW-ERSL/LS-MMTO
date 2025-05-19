import numpy as np

class Material:
  def __init__(self, matProp, physics):
    self.matProp = matProp
    self.physics = physics
    if(physics == 'structural'):
      nu = matProp['nu'];
      self.C = 1./(1-nu**2)*np.array([[1, nu, 0],[nu, 1, 0],[0, 0, (1-nu)/2]])
    elif(physics == 'thermal'):
      self.C = np.array([[1., 0.],[0., 1.]])
      
  #--------------------------#
  def computeSIMP_Interpolation(self, rho):
    E = 0.001*self.matProp['E'] + \
            (0.999*self.matProp['E'])*\
            (rho+0.01)**self.matProp['penal']
    return E
  #--------------------------#
  def computeRAMP_Interpolation(self, rho):
    E = 0.001*self.matProp['E']  +\
        (0.999*self.matProp['E'])*\
            (rho/(1.+self.matProp['penal']*(1.-rho)))
    return E
  #--------------------------#
  def getD0elemMatrix(self, meshProp):
    #TODO: do thermal
    if(self.physics == 'structural'):
      E = 1
      nu = self.matProp['nu'];
      k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,\
               -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
      D0 = E/(1-nu**2)*np.array\
        ([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
         [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
         [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
         [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
         [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
         [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
         [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
         [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ])
    elif(self.physics == 'thermal'):
      D0 = np.array([[ 2./3., -1./6., -1./3., -1./6.],\
              [-1./6.,  2./3., -1./6., -1./3.],\
              [-1./3., -1./6.,  2./3., -1./6.],\
              [-1./6., -1./3., -1./6.,  2./3.]])
        
      # all the elems have same base stiffness
    D0elem = meshProp['thickness']*meshProp['elemArea'][0]*\
            np.repeat(D0[np.newaxis, :, :], meshProp['numElems'], axis=0)
    return D0elem

 #--------------------------#
            
        
        
        
        

        
        
        
    
    