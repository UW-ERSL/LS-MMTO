import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from mmapy import gcmmasub, kktcheck, asymp, concheck, raaupdate,mmasub
from scipy.optimize import minimize, Bounds
from scipy.sparse import diags
from scipy.linalg import solve

def set_seed(manualSeed):
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  torch.manual_seed(manualSeed)
  torch.cuda.manual_seed(manualSeed)
  torch.cuda.manual_seed_all(manualSeed)
  np.random.seed(manualSeed)
  random.seed(manualSeed)
  
def to_np(x):
  return x.detach().cpu().numpy()

def to_torch(x):
  return torch.tensor(x).float()

def plotConvergence(convg):
  plt.figure();
  for key in convg:
    y = torch.as_tensor(convg[key]).detach().cpu().numpy()
    plt.plot(y[1:], label = str(key))
    plt.xlabel('Iterations')
    plt.ylabel(str(key))
    plt.grid('True')
    plt.figure()

def timing(f):
  def wrap(*args, **kwargs):
    time1 = time.time()
    ret = f(*args, **kwargs)
    time2 = time.time()
    print('{:s} function took {:.3f} sec'.format(f.__name__, (time2-time1)))

    return ret
  return wrap


# def applySensitivityFilter(ft, x, dc, dv):
#     if (ft['type'] == 1):
#         dc[0:int(len(x)/3)] = np.matmul(ft['H'],\
#                          np.multiply(x[0:int(len(x)/3)], \
#                                      dc[0:int(len(x)/3)])/ft['Hs']/ \
#                                         np.maximum(1e-3,x[0:int(len(x)/3)]));
#         dc[int(len(x)/3):int(2*len(x)/3)] = np.matmul(ft['H'],\
#                          np.multiply(x[int(len(x)/3):int(2*len(x)/3)], \
#                                      dc[int(len(x)/3):int(2*len(x)/3)])/ft['Hs']/ \
#                                         np.maximum(1e-3,x[int(len(x)/3):int(2*len(x)/3)]));
#         dc[int(2*len(x)/3):] = np.matmul(ft['H'],\
#                                 np.multiply(x[int(2*len(x)/3):], \
#                                             dc[int(2*len(x)/3):])/ft['Hs']/ \
#                                                 np.maximum(1e-3,x[int(2*len(x)/3):]));




#     elif (ft['type'] == 2):
#         dc[0:int(len(x)/3)] = np.matmul(ft['H'], (dc[0:int(len(x)/3)]/ft['Hs']));
#         dv[0,0:int(len(x)/3)] = np.matmul(ft['H'], (dv[0,0:int(len(x)/3)].T/ft['Hs']));
#         dv[1,0:int(len(x)/3)] = np.matmul(ft['H'], (dv[1,0:int(len(x)/3)].T/ft['Hs']));
#     return dc, dv;

def applySensitivityFilter(ft, x, dc, dv, num_patches=None):
    if num_patches is None:
        num_patches = len(x)/3
    if (ft['type'] == 1):
        dc[0:int(len(x) - 2*num_patches)] = np.matmul(ft['H'],\
                         np.multiply(x[0:int(len(x) - 2*num_patches)], \
                                     dc[0:int(len(x) - 2*num_patches)])/ft['Hs']/ \
                                        np.maximum(1e-3,x[0:int(len(x) - 2*num_patches)]));

        # dc[0:int(len(x)/3)] = np.matmul(ft['H'],\
        #                  np.multiply(x[0:int(len(x)/3)], \
        #                              dc[0:int(len(x) /3)])/ft['Hs']/ \
        #                                 np.maximum(1e-3,x[0:int(len(x)/3)]));

        # dc[int(len(x)/3):int(2*len(x)/3)] = np.matmul(ft['H'],\
        #                     np.multiply(x[0:int(len(x)/3)], \
        #                                 dc[int(len(x)/3):int(2*len(x)/3)])/ft['Hs']/ \
        #                                     np.maximum(1e-3,x[int(len(x)/3):int(2*len(x)/3)]));
        # dc[int(2*len(x)/3):] = np.matmul(ft['H'],\
        #                         np.multiply(x[0:int(len(x)/3)], \
        #                                     dc[int(2*len(x)/3):])/ft['Hs']/ \
        #                                         np.maximum(1e-3,x[int(2*len(x)/3):]));




    # elif (ft['type'] == 2):
    #     dc[0:int(len(x)/3)] = np.matmul(ft['H'], (dc[0:int(len(x)/3)]/ft['Hs']));
    #     dv[0,0:int(len(x)/3)] = np.matmul(ft['H'], (dv[0,0:int(len(x)/3)].T/ft['Hs']));
    #     dv[1,0:int(len(x)/3)] = np.matmul(ft['H'], (dv[1,0:int(len(x)/3)].T/ft['Hs']));
    return dc, dv;
class MMA:
    # The code was modified from [MMA Svanberg 1987]. Please cite the paper if
    # you end up using this code.
    def __init__(self):
        self.epoch = 0;
    def resetMMACounter(self):
        self.epoch = 0;
    def registerMMAIter(self, xval, xold1, xold2):
        self.epoch += 1;
        self.xval = xval;
        self.xold1 = xold1;
        self.xold2 = xold2;
    def setNumConstraints(self, numConstraints):
        self.numConstraints = numConstraints;
    def setNumDesignVariables(self, numDesVar):
        self.numDesignVariables = numDesVar;
    def setMinandMaxBoundsForDesignVariables(self, xmin, xmax):
        self.xmin = xmin;
        self.xmax = xmax;
    def setObjectiveWithGradient(self, obj, objGrad):
        self.objective = obj;
        self.objectiveGradient = objGrad;
    def setConstraintWithGradient(self, cons, consGrad):
        self.constraint = cons;
        self.consGrad = consGrad;
    def setScalingParams(self, zconst, zscale, ylinscale, yquadscale):
        self.zconst = zconst;
        self.zscale = zscale;
        self.ylinscale = ylinscale;
        self.yquadscale = yquadscale;
    def setMoveLimit(self, movelim):
        self.moveLimit = movelim;
    def setLowerAndUpperAsymptotes(self, low, upp):
        self.lowAsymp = low;
        self.upAsymp = upp;

    def getOptimalValues(self):
        return self.xmma, self.ymma, self.zmma;
    def getLagrangeMultipliers(self):
        return self.lam, self.xsi, self.eta, self.mu, self.zet;
    def getSlackValue(self):
        return self.slack;
    def getAsymptoteValues(self):
        return self.lowAsymp, self.upAsymp;

    # Function for the MMA sub problem
    def mmasub(self, xval):
        m = self.numConstraints;
        n = self.numDesignVariables;
        iter = self.epoch;
        xmin, xmax = self.xmin, self.xmax;
        xold1, xold2 = self.xold1, self.xold2;
        f0val, df0dx = self.objective, self.objectiveGradient;
        fval, dfdx = self.constraint, self.consGrad;
        low, upp = self.lowAsymp, self.upAsymp;
        a0, a, c, d = self.zconst, self.zscale, self.ylinscale, self.yquadscale;
        move = self.moveLimit;

        epsimin = 0.0000001
        raa0 = 0.00001
        albefa = 0.1
        asyinit = 0.5
        asyincr = 1.2
        asydecr = 0.7
        eeen = np.ones((n, 1))
        eeem = np.ones((m, 1))
        zeron = np.zeros((n, 1))
        # Calculation of the asymptotes low and upp
        if iter <= 2:
            low = xval-asyinit*(xmax-xmin)
            upp = xval+asyinit*(xmax-xmin)
        else:
            zzz = (xval-xold1)*(xold1-xold2)
            factor = eeen.copy()
            factor[np.where(zzz>0)] = asyincr
            factor[np.where(zzz<0)] = asydecr
            low = xval-factor*(xold1-low)
            upp = xval+factor*(upp-xold1)
            lowmin = xval-10*(xmax-xmin)
            lowmax = xval-0.01*(xmax-xmin)
            uppmin = xval+0.01*(xmax-xmin)
            uppmax = xval+10*(xmax-xmin)
            low = np.maximum(low,lowmin)
            low = np.minimum(low,lowmax)
            upp = np.minimum(upp,uppmax)
            upp = np.maximum(upp,uppmin)
        # Calculation of the bounds alfa and beta
        zzz1 = low+albefa*(xval-low)
        zzz2 = xval-move*(xmax-xmin)
        zzz = np.maximum(zzz1,zzz2)
        alfa = np.maximum(zzz,xmin)
        zzz1 = upp-albefa*(upp-xval)
        zzz2 = xval+move*(xmax-xmin)
        zzz = np.minimum(zzz1,zzz2)
        beta = np.minimum(zzz,xmax)
        # Calculations of p0, q0, P, Q and b
        xmami = xmax-xmin
        xmamieps = 0.00001*eeen
        xmami = np.maximum(xmami,xmamieps)
        xmamiinv = eeen/xmami
        ux1 = upp-xval
        ux2 = ux1*ux1
        xl1 = xval-low
        xl2 = xl1*xl1
        uxinv = eeen/ux1
        xlinv = eeen/xl1
        p0 = zeron.copy()
        q0 = zeron.copy()
        p0 = np.maximum(df0dx,0)
        q0 = np.maximum(-df0dx,0)
        pq0 = 0.001*(p0+q0)+raa0*xmamiinv
        p0 = p0+pq0
        q0 = q0+pq0
        p0 = p0*ux2
        q0 = q0*xl2
        P = np.zeros((m,n)) ## @@ make sparse with scipy?
        Q = np.zeros((m,n)) ## @@ make sparse with scipy?
        P = np.maximum(dfdx,0)
        Q = np.maximum(-dfdx,0)
        PQ = 0.001*(P+Q)+raa0*np.dot(eeem,xmamiinv.T)
        P = P+PQ
        Q = Q+PQ
        P = (diags(ux2.flatten(),0).dot(P.T)).T
        Q = (diags(xl2.flatten(),0).dot(Q.T)).T
        b = (np.dot(P,uxinv)+np.dot(Q,xlinv)-fval)
        # Solving the subproblem by a primal-dual Newton method
        xmma,ymma,zmma,lam,xsi,eta,mu,zet,s = subsolv(m,n,epsimin,low,upp,alfa,\
                                                      beta,p0,q0,P,Q,a0,a,b,c,d)
        # Return values
        self.xmma, self.ymma, self.zmma = xmma, ymma, zmma;
        self.lam, self.xsi, self.eta, self.mu, self.zet = lam,xsi,eta,mu,zet;
        self.slack = s;
        self.lowAsymp, self.upAsymp = low, upp;


# Function for the GCMMA sub problem
def gcmmasub(m,n,iter,epsimin,xval,xmin,xmax,low,upp,raa0,raa,f0val,df0dx,\
             fval,dfdx,a0,a,c,d):
    eeen = np.ones((n,1))
    zeron = np.zeros((n,1))
    # Calculations of the bounds alfa and beta
    albefa = 0.1
    zzz = low+albefa*(xval-low)
    alfa = np.maximum(zzz,xmin)
    zzz = upp-albefa*(upp-xval)
    beta = np.minimum(zzz,xmax)
    # Calculations of p0, q0, r0, P, Q, r and b.
    xmami = xmax-xmin
    xmamieps = 0.00001*eeen
    xmami = np.maximum(xmami,xmamieps)
    xmamiinv = eeen/xmami
    ux1 = upp-xval
    ux2 = ux1*ux1
    xl1 = xval-low
    xl2 = xl1*xl1
    uxinv = eeen/ux1
    xlinv = eeen/xl1
    #
    p0 = zeron.copy()
    q0 = zeron.copy()
    p0 = np.maximum(df0dx,0)
    q0 = np.maximum(-df0dx,0)
    pq0 = p0+q0
    p0 = p0+0.001*pq0
    q0 = q0+0.001*pq0
    p0 = p0+raa0*xmamiinv
    q0 = q0+raa0*xmamiinv
    p0 = p0*ux2
    q0 = q0*xl2
    r0 = f0val-np.dot(p0.T,uxinv)-np.dot(q0.T,xlinv)
    #
    P = np.zeros((m,n)) ## @@ make sparse with scipy?
    Q = np.zeros((m,n)) ## @@ make sparse with scipy
    P = (diags(ux2.flatten(),0).dot(P.T)).T
    Q = (diags(xl2.flatten(),0).dot(Q.T)).T
    b = (np.dot(P,uxinv)+np.dot(Q,xlinv)-fval)
    P = np.maximum(dfdx,0)
    Q = np.maximum(-dfdx,0)
    PQ = P+Q
    P = P+0.001*PQ
    Q = Q+0.001*PQ
    P = P+np.dot(raa,xmamiinv.T)
    Q = Q+np.dot(raa,xmamiinv.T)
    P = (diags(ux2.flatten(),0).dot(P.T)).T
    Q = (diags(xl2.flatten(),0).dot(Q.T)).T
    r = fval-np.dot(P,uxinv)-np.dot(Q,xlinv)
    b = -r
    # Solving the subproblem by a primal-dual Newton method
    xmma,ymma,zmma,lam,xsi,eta,mu,zet,s = subsolv(m,n,epsimin,low,upp,alfa,\
                                                  beta,p0,q0,P,Q,a0,a,b,c,d)
    # Calculations of f0app and fapp.
    ux1 = upp-xmma
    xl1 = xmma-low
    uxinv = eeen/ux1
    xlinv = eeen/xl1
    f0app = r0+np.dot(p0.T,uxinv)+np.dot(q0.T,xlinv)
    fapp = r+np.dot(P,uxinv)+np.dot(Q,xlinv)
    # Return values
    return xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,f0app,fapp

def subsolv(m,n,epsimin,low,upp,alfa,beta,p0,q0,P,Q,a0,a,b,c,d):


    een = np.ones((n,1))
    eem = np.ones((m,1))
    epsi = 1
    epsvecn = epsi*een
    epsvecm = epsi*eem
    x = 0.5*(alfa+beta)
    y = eem.copy()
    z = np.array([[1.0]])
    lam = eem.copy()
    xsi = een/(x-alfa)
    xsi = np.maximum(xsi,een)
    eta = een/(beta-x)
    eta = np.maximum(eta,een)
    mu = np.maximum(eem,0.5*c)
    zet = np.array([[1.0]])
    s = eem.copy()
    itera = 0
    # Start while epsi>epsimin
    while epsi > epsimin:
        epsvecn = epsi*een
        epsvecm = epsi*eem
        ux1 = upp-x
        xl1 = x-low
        ux2 = ux1*ux1
        xl2 = xl1*xl1
        uxinv1 = een/ux1
        xlinv1 = een/xl1
        plam = p0+np.dot(P.T,lam)
        qlam = q0+np.dot(Q.T,lam)
        gvec = np.dot(P,uxinv1)+np.dot(Q,xlinv1)
        dpsidx = plam/ux2-qlam/xl2
        rex = dpsidx-xsi+eta
        rey = c+d*y-mu-lam
        rez = a0-zet-np.dot(a.T,lam)
        relam = gvec-a*z-y+s-b
        rexsi = xsi*(x-alfa)-epsvecn
        reeta = eta*(beta-x)-epsvecn
        remu = mu*y-epsvecm
        rezet = zet*z-epsi
        res = lam*s-epsvecm
        residu1 = np.concatenate((rex, rey, rez), axis = 0)
        residu2 = np.concatenate((relam, rexsi, reeta, remu, rezet, res), axis = 0)
        residu = np.concatenate((residu1, residu2), axis = 0)
        residunorm = np.sqrt((np.dot(residu.T,residu)).item())
        residumax = np.max(np.abs(residu))
        ittt = 0
        # Start while (residumax>0.9*epsi) and (ittt<200)
        while (residumax > 0.9*epsi) and (ittt < 200):
            ittt = ittt+1
            itera = itera+1
            ux1 = upp-x
            xl1 = x-low
            ux2 = ux1*ux1
            xl2 = xl1*xl1
            ux3 = ux1*ux2
            xl3 = xl1*xl2
            uxinv1 = een/ux1
            xlinv1 = een/xl1
            uxinv2 = een/ux2
            xlinv2 = een/xl2
            plam = p0+np.dot(P.T,lam)
            qlam = q0+np.dot(Q.T,lam)
            gvec = np.dot(P,uxinv1)+np.dot(Q,xlinv1)
            GG = (diags(uxinv2.flatten(),0).dot(P.T)).T-(diags\
                                     (xlinv2.flatten(),0).dot(Q.T)).T
            dpsidx = plam/ux2-qlam/xl2
            delx = dpsidx-epsvecn/(x-alfa)+epsvecn/(beta-x)
            dely = c+d*y-lam-epsvecm/y
            delz = a0-np.dot(a.T,lam)-epsi/z
            dellam = gvec-a*z-y-b+epsvecm/lam
            diagx = plam/ux3+qlam/xl3
            diagx = 2*diagx+xsi/(x-alfa)+eta/(beta-x)
            diagxinv = een/diagx
            diagy = d+mu/y
            diagyinv = eem/diagy
            diaglam = s/lam
            diaglamyi = diaglam+diagyinv
            # Start if m<n
            if m < n:
                blam = dellam+dely/diagy-np.dot(GG,(delx/diagx))
                bb = np.concatenate((blam,delz),axis = 0)
                Alam = np.asarray(diags(diaglamyi.flatten(),0) \
                    +(diags(diagxinv.flatten(),0).dot(GG.T).T).dot(GG.T))
                AAr1 = np.concatenate((Alam,a),axis = 1)
                AAr2 = np.concatenate((a,-zet/z),axis = 0).T
                AA = np.concatenate((AAr1,AAr2),axis = 0)
                solut = solve(AA,bb)
                dlam = solut[0:m]
                dz = solut[m:m+1]
                dx = -delx/diagx-np.dot(GG.T,dlam)/diagx
            else:
                diaglamyiinv = eem/diaglamyi
                dellamyi = dellam+dely/diagy
                Axx = np.asarray(diags(diagx.flatten(),0) \
                    +(diags(diaglamyiinv.flatten(),0).dot(GG).T).dot(GG))
                azz = zet/z+np.dot(a.T,(a/diaglamyi))
                axz = np.dot(-GG.T,(a/diaglamyi))
                bx = delx+np.dot(GG.T,(dellamyi/diaglamyi))
                bz = delz-np.dot(a.T,(dellamyi/diaglamyi))
                AAr1 = np.concatenate((Axx,axz),axis = 1)
                AAr2 = np.concatenate((axz.T,azz),axis = 1)
                AA = np.concatenate((AAr1,AAr2),axis = 0)
                bb = np.concatenate((-bx,-bz),axis = 0)
                solut = solve(AA,bb)
                dx = solut[0:n]
                dz = solut[n:n+1]
                dlam = np.dot(GG,dx)/diaglamyi-dz*(a/diaglamyi)\
                    +dellamyi/diaglamyi
                # End if m<n
            dy = -dely/diagy+dlam/diagy
            dxsi = -xsi+epsvecn/(x-alfa)-(xsi*dx)/(x-alfa)
            deta = -eta+epsvecn/(beta-x)+(eta*dx)/(beta-x)
            dmu = -mu+epsvecm/y-(mu*dy)/y
            dzet = -zet+epsi/z-zet*dz/z
            ds = -s+epsvecm/lam-(s*dlam)/lam
            xx = np.concatenate((y,z,lam,xsi,eta,mu,zet,s),axis = 0)
            dxx = np.concatenate((dy,dz,dlam,dxsi,deta,dmu,dzet,ds),axis = 0)
            #
            stepxx = -1.01*dxx/xx
            stmxx = np.max(stepxx)
            stepalfa = -1.01*dx/(x-alfa)
            stmalfa = np.max(stepalfa)
            stepbeta = 1.01*dx/(beta-x)
            stmbeta = np.max(stepbeta)
            stmalbe = max(stmalfa,stmbeta)
            stmalbexx = max(stmalbe,stmxx)
            stminv = max(stmalbexx,1.0)
            steg = 1.0/stminv
            #
            xold = x.copy()
            yold = y.copy()
            zold = z.copy()
            lamold = lam.copy()
            xsiold = xsi.copy()
            etaold = eta.copy()
            muold = mu.copy()
            zetold = zet.copy()
            sold = s.copy()
            #
            itto = 0
            resinew = 2*residunorm
            # Start: while (resinew>residunorm) and (itto<50)
            while (resinew > residunorm) and (itto < 50):
                itto = itto+1
                x = xold+steg*dx
                y = yold+steg*dy
                z = zold+steg*dz
                lam = lamold+steg*dlam
                xsi = xsiold+steg*dxsi
                eta = etaold+steg*deta
                mu = muold+steg*dmu
                zet = zetold+steg*dzet
                s = sold+steg*ds
                ux1 = upp-x
                xl1 = x-low
                ux2 = ux1*ux1
                xl2 = xl1*xl1
                uxinv1 = een/ux1
                xlinv1 = een/xl1
                plam = p0+np.dot(P.T,lam)
                qlam = q0+np.dot(Q.T,lam)
                gvec = np.dot(P,uxinv1)+np.dot(Q,xlinv1)
                dpsidx = plam/ux2-qlam/xl2
                rex = dpsidx-xsi+eta
                rey = c+d*y-mu-lam
                rez = a0-zet-np.dot(a.T,lam)
                relam = gvec-np.dot(a,z)-y+s-b
                rexsi = xsi*(x-alfa)-epsvecn
                reeta = eta*(beta-x)-epsvecn
                remu = mu*y-epsvecm
                rezet = np.dot(zet,z)-epsi
                res = lam*s-epsvecm
                residu1 = np.concatenate((rex,rey,rez),axis = 0)
                residu2 = np.concatenate((relam,rexsi,reeta,remu,rezet,res), \
                                         axis = 0)
                residu = np.concatenate((residu1,residu2),axis = 0)
                resinew = np.sqrt(np.dot(residu.T,residu))
                steg = steg/2
                # End: while (resinew>residunorm) and (itto<50)
            residunorm = resinew.copy()
            residumax = max(abs(residu))
            steg = 2*steg
            # End: while (residumax>0.9*epsi) and (ittt<200)
        epsi = 0.1*epsi
        # End: while epsi>epsimin
    xmma = x.copy()
    ymma = y.copy()
    zmma = z.copy()
    lamma = lam
    xsimma = xsi
    etamma = eta
    mumma = mu
    zetmma = zet
    smma = s

    return xmma,ymma,zmma,lamma,xsimma,etamma,mumma,zetmma,smma

def kktcheck(m,n,x,y,z,lam,xsi,eta,mu,zet,s,xmin,xmax,df0dx,fval,dfdx,a0,a,c,d):

    rex = df0dx+np.dot(dfdx.T,lam)-xsi+eta
    rey = c+d*y-mu-lam
    rez = a0-zet-np.dot(a.T,lam)
    relam = fval-a*z-y+s
    rexsi = xsi*(x-xmin)
    reeta = eta*(xmax-x)
    remu = mu*y
    rezet = zet*z
    res = lam*s
    residu1 = np.concatenate((rex,rey,rez),axis = 0)
    residu2 = np.concatenate((relam,rexsi,reeta,remu,rezet,res), axis = 0)
    residu = np.concatenate((residu1,residu2),axis = 0)
    residunorm = np.sqrt((np.dot(residu.T,residu)).item())
    residumax = np.max(np.abs(residu))
    return residu,residunorm,residumax

def raaupdate(xmma,xval,xmin,xmax,low,upp,f0valnew,\
              fvalnew,f0app,fapp,raa0,raa,raa0eps,raaeps,epsimin):

    raacofmin = 1e-12
    eeem = np.ones((raa.size,1))
    eeen = np.ones((xmma.size,1))
    xmami = xmax-xmin
    xmamieps = 0.00001*eeen
    xmami = np.maximum(xmami,xmamieps)
    xxux = (xmma-xval)/(upp-xmma)
    xxxl = (xmma-xval)/(xmma-low)
    xxul = xxux*xxxl
    ulxx = (upp-low)/xmami
    raacof = np.dot(xxul.T,ulxx)
    raacof = np.maximum(raacof,raacofmin)
    #
    f0appe = f0app+0.5*epsimin
    if np.all(f0valnew>f0appe) == True:
        deltaraa0 = (1.0/raacof)*(f0valnew-f0app)
        zz0 = 1.1*(raa0+deltaraa0)
        zz0 = np.minimum(zz0,10*raa0)
        raa0 = zz0
    #
    fappe = fapp+0.5*epsimin*eeem;
    fdelta = fvalnew-fappe
    deltaraa = (1/raacof)*(fvalnew-fapp)
    zzz = 1.1*(raa+deltaraa)
    zzz = np.minimum(zzz,10*raa)
    raa[np.where(fdelta>0)] = zzz[np.where(fdelta>0)]
    #
    return raa0,raa


def concheck(m,epsimin,f0app,f0valnew,fapp,fvalnew):

    eeem = np.ones((m,1))
    f0appe = f0app+epsimin
    fappe = fapp+epsimin*eeem
    arr1 = np.concatenate((f0appe.flatten(),fappe.flatten()))
    arr2 = np.concatenate((f0valnew.flatten(),fvalnew.flatten()))
    if np.all(arr1 >= arr2) == True:
        conserv = 1
    else:
        conserv = 0
    return conserv


def asymp(outeriter,n,xval,xold1,xold2,xmin,xmax,low,upp,raa0,raa,\
          raa0eps,raaeps,df0dx,dfdx):



    eeen = np.ones((n,1))
    asyinit = 0.5
    asyincr = 1.2
    asydecr = 0.7
    xmami = xmax-xmin
    xmamieps = 0.00001*eeen
    xmami = np.maximum(xmami,xmamieps)
    raa0 = np.dot(np.abs(df0dx).T,xmami)
    raa0 = np.maximum(raa0eps,(0.1/n)*raa0)
    raa = np.dot(np.abs(dfdx),xmami)
    raa = np.maximum(raaeps,(0.1/n)*raa)
    if outeriter <= 2:
        low = xval-asyinit*xmami
        upp = xval+asyinit*xmami
    else:
        xxx = (xval-xold1)*(xold1-xold2)
        factor = eeen.copy()
        factor[np.where(xxx>0)] = asyincr
        factor[np.where(xxx<0)] = asydecr
        low = xval-factor*(xold1-low)
        upp = xval+factor*(upp-xold1)
        lowmin = xval-10*xmami
        lowmax = xval-0.01*xmami
        uppmin = xval+0.01*xmami
        uppmax = xval+10*xmami
        low = np.maximum(low,lowmin)
        low = np.minimum(low,lowmax)
        upp = np.minimum(upp,uppmax)
        upp=np.maximum(upp,uppmin)
    return low,upp,raa0,raa