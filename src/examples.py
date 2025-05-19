import numpy as np

def getBC(example, meshProp):
  nelx, nely = meshProp['nelx'], meshProp['nely']
  if(example == 1): # tip cantilever
    exampleName = 'TipCantilever'
    physics = 'structural'
    numDOFPerNode = 2
    ndof = 2*(nelx+1)*(nely+1)
    force = np.zeros((ndof,1))
    dofs=np.arange(ndof);
    fixed = dofs[0:2*(nely+1):1]
    free = np.setdiff1d(np.arange(ndof), fixed)
    force[2*(nelx+1)*(nely+1)-2*nely+1, 0 ] = -1.

    symmetry = {'XAxis': {'isOn':False, 'midPt':0.5*nely*meshProp['elemSize'][1]},\
              'YAxis': {'isOn':False, 'midPt':0.5*nelx*meshProp['elemSize'][0]}}

    bc = {'exampleName':exampleName, 'physics':physics,\
          'numDOFPerNode': numDOFPerNode,'force':force,\
          'fixed':fixed, 'free':free, 'symmetry':symmetry}
      
  elif(example == 2): # mid cantilever
    exampleName = 'MidCantilever'
    physics = 'structural'
    numDOFPerNode = 2
    ndof = 2*(nelx+1)*(nely+1)
    force = np.zeros((ndof,1))
    dofs=np.arange(ndof);
    fixed = dofs[0:2*(nely+1):1]
    free = np.setdiff1d(np.arange(ndof), fixed)
    force[2*(nelx+1)*(nely+1)- (nely+1), 0 ] = -1.
    symmetry = {'XAxis': {'isOn':True, 'midPt':0.5*nely*meshProp['elemSize'][1]},\
              'YAxis': {'isOn':False, 'midPt':0.5*nelx*meshProp['elemSize'][0]}}
    bc = {'exampleName':exampleName, 'physics':physics, \
          'numDOFPerNode': numDOFPerNode,'force':force, \
          'fixed':fixed, 'free':free, 'symmetry':symmetry}
      
  elif(example == 3): # BodyHeatLoad
    exampleName = 'bodyHeatLoad'
    physics = 'thermal'
    numDOFPerNode = 1
    ndof = (nelx+1)*(nely+1)
    force = 0.*np.ones((ndof,1))
    netLoad = 1000.
    force[0::1] = netLoad/ndof
    fixed =  np.arange(int(nely/2 -2) , int(nely/2 + 2) );
    free = np.setdiff1d(np.arange(ndof), fixed)
    symmetry = {'XAxis': {'isOn':True, 'midPt':0.5*nely*meshProp['elemSize'][1]},\
                'YAxis': {'isOn':False, 'midPt':0.5*nelx*meshProp['elemSize'][0]}}
    bc = {'exampleName':exampleName, 'physics':physics, \
          'numDOFPerNode': numDOFPerNode,'force':force, \
          'fixed':fixed, 'free':free, 'symmetry':symmetry}
  elif(example == 4): # Michell
    exampleName = 'Michell'
    physics = 'structural'
    numDOFPerNode = 2
    ndof = 2*(nelx+1)*(nely+1);
    force = np.zeros((ndof,1))
    dofs=np.arange(ndof);
    fixed=np.array([ 0,1,2*(nelx+1)*(nely+1)-2*nely+1] );
    free = np.setdiff1d(np.arange(ndof), fixed)
    force[int(2*nelx*(nely+1)/4)+1 ,0]= -1e0*2;
    force[int(2*nelx*(nely+1)/2)+1 ,0]= -2e0*2;
    force[int(2*nelx*(nely+1)*3/4)+1 ,0]= -1e0*2;
    nonDesignRegion = None;
    symXAxis = False;
    symYAxis = True;
    symmetry = {'XAxis': {'isOn':symXAxis, 'midPt':0.5*nely*meshProp['elemSize'][1]},\
              'YAxis': {'isOn':symYAxis, 'midPt':0.5*nelx*meshProp['elemSize'][0]}}
    bc = {'exampleName':exampleName, 'physics':physics, \
          'numDOFPerNode': numDOFPerNode,'force':force, \
          'fixed':fixed, 'free':free, 'symmetry':symmetry}
 
 
 

  return bc

    
