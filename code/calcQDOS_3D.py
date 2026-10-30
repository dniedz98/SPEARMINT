# Calculates a 3D histogram of the momentum resolved DOS at qz=0
#    Parameters are specified manually in this file

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time

t_start = time.time()

def get_LatticeMatrix(path2file):
    LatticeMatrix = []
    with open(path2file,'r') as file:
        # skip the first two lines
        #line = file.readline()
        line = file.readline()
        # get the contents as a list of lines
        contents = file.readlines()
        LM = [line.split()[:3] for line in contents]
        LatticeMatrix = np.array([ [float(i) for i in lm] for lm in LM ])
    return LatticeMatrix

def get_ionpos_ionsp(path2file):
    ionspecies = []
    ionpos = []
    with open(path2file,'r') as file:
        # Skip the comment line at the top of the file
        line = file.readline()
        # record ionspecies, and lattice coordinate
        while line:
            line = file.readline()
            line_text = line.split()
            if len(line_text) > 0:
                if line_text[0] == "ion":
                    ionspecies.append(line_text[1])
                    ionpos.append([float(x) for x in line_text[2:5]])
    return np.array(ionpos),np.array(ionspecies)  

# Coordinate system conversions and supercell
def lat2cart(M,ionpos):
    return np.dot(ionpos,np.transpose(M))
        
def cart2lat(M,ionpos):
    return np.dot(ionpos,np.linalg.inv(np.transpose(M)))

# G = (M^-1)^T
def lat2cart_RS(M,q):
    return (2*np.pi) * np.dot(q,np.linalg.inv(M))

# G = (M^T)^-1
def cart2lat_RS(M,q):
    return (1./(2*np.pi)) * np.dot(q,M)

def generateCellMap(Nx,Ny,Nz):
    retval = []
    for i in range(-Nx,Nx+1):
        for j in range(-Ny,Ny+1):
            for k in range(-Nz,Nz+1):
                retval.append([i,j,k])
    return np.array(retval)

# last dimension is 3D, but rotate allong z-axis
def rotate_zaxis(array,t):
    R = np.array([[np.cos(t),-np.sin(t), 0],
                  [np.sin(t), np.cos(t), 0],
                  [0        , 0        , 1]])
    return np.dot(array,R.T)

# graphene lattice constant (used to make square in momentum space to unfold bands onto)
ax, ay = 4.6638441, 4.6638441 # in bohr 

# Load in the wannier hamiltonian and the lattice/ionpos
H = np.load("../results/H.npy")
cellMap = np.load("../results/cellMap.npy")
ionpos, ionsp = get_ionpos_ionsp("../data/30DEG_TBG/30TBG.ionpos")
M             = get_LatticeMatrix("../data/30DEG_TBG/30TBG.lattice")
nBands = H.shape[-1]

histOutFileName = "../results/QDOS3D" # will be .npy

# Uniformly sample unique k-points in the supercell BZ
nBatches = 10 # 10
Nk = 10 # 10 # was 1000 for 224 wannier centers
Nktot = Nk*nBatches

histDims = (500,500,500) # This is a good size
#histDims = (100,100,100) # smaller for faster testing
hist = np.zeros(histDims)
for batch in range(nBatches):
    print("Starting batch", batch+1, "of", nBatches, "at", time.time()-t_start, "[s]")
    k_lat = np.random.uniform(0, 1, size=(Nk,3))

    # Generate more equivalent q-points on which to unfold the bandstructure
    Nx, Ny = int(M[0][0]/(ax*np.sqrt(3)/2)), int(M[1][1]/(ay*np.sqrt(3)/2)) 
    # sqrt(3)/2 factor to necessary to convert to graphene RLV correctly (ai dot bj = 2pi dij)
    
    Gs = generateCellMap(Nx,Ny,1)
    NG = len(Gs)
    q_lat = Gs[:,None,:] + k_lat[None,:,:]
    q_cart = lat2cart_RS(M,q_lat)
    r_i = lat2cart(M,ionpos)

    print("Nx, Ny =", Nx, Ny)
    print("Gs.shape =", Gs.shape)

    # Put everything on the gpu
    H       = cp.asarray(H)
    cellMap = cp.asarray(cellMap)
    r_i     = cp.asarray(r_i)
    k_lat   = cp.asarray(k_lat)
    q_cart  = cp.asarray(q_cart)
    print("Moved all data to gpu")

    # Evaluate the model at each kpoint
    Hk = np.tensordot(cp.exp((2j*cp.pi)*cp.dot(k_lat,cellMap.T)), H, axes=1)
    print("Formed Hk")
    evals,evects = cp.linalg.eigh(Hk) # evects [kpoint, orbital, band]
    print("Diagonalized all Hk")
    evects = cp.transpose(evects,(0,2,1)) # evects [kpoint, band, orbital]
    print("evects.shape", evects.shape)

    # Calculate the unfolding factors
    phase_factor = cp.exp(-1.0j*cp.einsum("gkx,ox-> gko", q_cart,r_i))
    print("phase_factor.shape", phase_factor.shape)
    weights = cp.abs( np.einsum("kbo, gko -> gkb", evects, phase_factor ) )**2 # [kpoint,RLV,band]
    print("weights.dtype", weights.dtype)
    print("weights.shape", weights.shape)
    print("max of raw weights:", cp.max(weights))

    # Bring the evals, qpoints and weights over to the cpu and re-shape them for histogramming
    evals   = evals.get()   #[kpoint,band]
    q_cart  = q_cart.get()  #[Gvec,kpoint,xyz]
    weights = weights.get() #[Gvec,kpoint,band]
    print("Moved all relevant data back to cpu")

    # Set the energy range based on the extent of the first batch of eigenvalues padded by 1% on each side
    min_eval, max_eval =  np.min(evals), np.max(evals)
    print("evals range =", min_eval, max_eval)
    if batch == 0 :
        Erange = max_eval - min_eval
        # sqrt3/2 factor necessary to convert to unitcell graphene RLV correctly 
        eqrange = (
                (min_eval-0.01*Erange, max_eval+0.01*Erange),
                (-2*np.pi/(ax*np.sqrt(3)/2),2*np.pi/(ax*np.sqrt(3)/2)),
                (-2*np.pi/(ay*np.sqrt(3)/2),2*np.pi/(ay*np.sqrt(3)/2))
                )

    # Flatten weights for histogramming
    weights = weights.flatten()
    # Broadcast evals for use in loop
    evals  = np.broadcast_to( evals[None,:,:,None], (NG,Nk,nBands,1) )

    # Average over 12fold symmetry to increase sampling density
    # For all 12 rotations of the q_cart about the Gamma point
    for i in range(12):
        q_cart_rot = rotate_zaxis(q_cart, 2*np.pi*(i/12) )

        # Broadcast and histogram the samples of the electronic structure
        q_cart_rot = np.broadcast_to( q_cart_rot[:,:,None,:2] , (NG,Nk,nBands,2) )
        samples = np.concatenate( (evals, q_cart_rot), axis=3 )
        samples = samples.reshape((-1,3))

        hist += np.histogramdd(samples, bins=histDims, range=eqrange, density=None, weights=weights)[0]

print("np.max(hist)", np.max(hist))
np.save(histOutFileName,hist)

# Save energy and momenta axes in atomic units
np.save("../results/Erange" , np.linspace(eqrange[0][0],eqrange[0][1],histDims[0])) # Hartree
np.save("../results/qxrange", np.linspace(eqrange[1][0],eqrange[1][1],histDims[0])) # bohr^-1
np.save("../results/qyrange", np.linspace(eqrange[2][0],eqrange[2][1],histDims[0])) # bohr^-1

print("Axes ranges:", eqrange)

print("DONE!")
