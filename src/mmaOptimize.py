import numpy as np
import time
from scipy.sparse import diags
import matplotlib.pyplot as plt
from utilFuncs import  MMA ,applySensitivityFilter


def optimize(mesh, optimizationParams, ft, \
             objectiveHandle, consHandle, numConstraints):
    rho = np.ones((mesh.meshProp['nelx']*mesh.meshProp['nely']*3)); # design variable for topology and material selection
    rho[0:mesh.meshProp['numElems']] = 0.5; # initial guess for topology
    rho[mesh.meshProp['numElems']:] = 0.0; # initial guess for material selection
    loop = 0;
    change = 1.;
    m = numConstraints; # num constraints
    n = mesh.meshProp['numElems']*3 ;
    mma = MMA();
    mma.setNumConstraints(numConstraints);
    mma.setNumDesignVariables(n);
    mma.setMinandMaxBoundsForDesignVariables\
        (np.zeros((n,1)),np.ones((n,1)));

    xval = rho[np.newaxis].T
    xold1, xold2 = xval.copy(), xval.copy();
    mma.registerMMAIter(xval, xold1, xold2);
    mma.setLowerAndUpperAsymptotes(np.ones((n,1)), np.ones((n,1)));
    mma.setScalingParams(1.0, np.zeros((m,1)), \
                         10000*np.ones((m,1)), np.zeros((m,1)))
    # mma.setMoveLimit(0.2);
    mma.setMoveLimit(0.2);
    mmaTime = 0;

    t0 = time.perf_counter();
    listJ = []
    listvf = []
    listmass = []

    while( (change > optimizationParams['relTol']) \
           and (loop < optimizationParams['maxIters'])\
           or (loop < optimizationParams['minIters'])):
        loop = loop + 1;
        rel_conv_tol = optimizationParams['relTol']
        J, dJ = objectiveHandle(rho);

        vc, dvc = consHandle(rho);

        dJ, dvc = applySensitivityFilter(ft, rho, dJ, dvc)

        J, dJ = J, dJ[np.newaxis].T
        tmr = time.perf_counter();
        mma.setObjectiveWithGradient(J, dJ);
        mma.setConstraintWithGradient(vc, dvc);


        # xval,_ = applySensitivityFilter(ft, rho, rho, dvc)
        # xval = xval[np.newaxis].T;
        xval = rho.copy()[np.newaxis].T;
        mma.mmasub(xval);

        xmma, _, _ = mma.getOptimalValues();
        xold2 = xold1.copy();
        xold1 = xval.copy();
        rho = xmma.copy().flatten()
        mma.registerMMAIter(rho, xval.copy(), xold2.copy())

        mmaTime += time.perf_counter() - tmr;

        status = 'Iter {:d}; J {:.2e}; vf {:.2F}'.\
                format(loop, J, np.mean(rho[0:mesh.meshProp['numElems']]));
        print(status)
        listJ.append(J)
        listvf.append(np.mean(rho[0:mesh.meshProp['numElems']]))
        listmass.append(vc)

        if loop > optimizationParams['minIters']:
            dJ = (listJ[-1] - listJ[-2]) / listJ[-2]

            if abs(dJ) < rel_conv_tol and abs(np.max(vc)) < rel_conv_tol:
                break

        if(loop%10 == 0):
            plt.imshow(-np.flipud(rho[0:mesh.meshProp['numElems']].reshape((mesh.meshProp['nelx'], \
                     mesh.meshProp['nely'])).T), cmap='gray');
            plt.title(status)
            plt.show()
    
    totTime = time.perf_counter() - t0;

    print('total time(s): ', totTime);
    print('mma time(s): ', mmaTime);
    print('FE time(s): ', totTime - mmaTime);
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Compliance', color='tab:blue')
    ax1.plot(listJ, color='tab:blue', label='Compliance')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # # # Plot volume fraction on right y-axis with dotted line
    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Volume Fraction', color='tab:orange')
    # ax2.plot(listvf, color='tab:orange', linestyle=':', label='Volume Fraction')
    # ax2.tick_params(axis='y', labelcolor='tab:orange')
    # ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    # plt.title('MMA: Compliance  and volume fraction vs. Iterations')


    # Plot mass on right y-axis with dotted line
    ax2 = ax1.twinx()
    ax2.set_ylabel('Mass constraint', color='tab:orange')
    ax2.plot(listmass, color='tab:orange', linestyle=':', label='Mass constraint')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))


    plt.title('MMA: Mass constraint and Compliance vs. Iterations')
    plt.show()
    return rho;