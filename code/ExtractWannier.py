import numpy as np
#import matplotlib.pyplot as plt
import re

# Define units
DEGREES = 2*np.pi/360

# Import common utility functions
from SPEARMINT_Utils import *


def loadHWannier(path,wanPat):
    #Read the MLWF cell map, weights and Hamiltonian:    
    cellMap = np.loadtxt(path+wanPat+".mlwfCellMap")[:,0:3].astype(int) #used to be np.int
    Wwannier = np.fromfile(path+wanPat+".mlwfCellWeights")
    nCells = cellMap.shape[0]
    nBands = int(np.sqrt(Wwannier.shape[0] / nCells))
    Wwannier = Wwannier.reshape((nCells,nBands,nBands)).swapaxes(1,2)
    #--- Get k-point folding from totalE.out:
    for line in open(path+'/totalE.out'):
        if line.startswith('kpoint-folding'):
            kfold = np.array([int(tok) for tok in line.split()[1:4]])
    kfoldProd = np.prod(kfold)
    kStride = np.array([kfold[1]*kfold[2], kfold[2], 1])
    #--- Read reduced Wannier Hamiltonian and expand it:
    Hreduced = np.fromfile(path+wanPat+".mlwfH").reshape((kfoldProd,nBands,nBands)).swapaxes(1,2)
    iReduced = np.dot(np.mod(cellMap, kfold[None,:]), kStride)
    Hwannier = Wwannier * Hreduced[iReduced]
    # optionally read the slab projected weights (not used in this demonstration)
    try:
        S1reduced = np.fromfile(path+wanPat+".mlwfW").reshape((kfoldProd,nBands,nBands)).swapaxes(1,2)
        S1wannier = S1reduced[iReduced]
        S1All = np.zeros((nBlocks,nK,nBands))
        slabProj = True
        print("Using Slab Projected Weights")
    except:
        slabProj = False

    return nCells,nBands,cellMap,Hwannier

def readWannierCenters(path,wanPat):
    # Read in the wannier function centers from wanPat.out
    grep_res = grep("spread: ",path+wanPat+".out")
    return np.array( [[float(x) for x in line.split()[1:4]] for line in grep_res] )

def getCellMapDict(cellMap):
    cellMapDict = {}
    # Iterate through the array and populate the dictionary
    for i, coord in enumerate(cellMap):
        # Convert the coordinate array to a tuple to make it hashable
        coord_tuple = tuple(coord)
        # Add the coordinate tuple as key and its index as value
        cellMapDict[coord_tuple] = i
    return cellMapDict

# Creates scatterplot visualization of orbital sites and optionally neighbor connections
def visualize(orbitals,ionpos,lattice,plotNeighborLines=False):
    plt.figure(figsize=(6,6))
    colors = ['b','r'] # for flake and substrate 
    for orb in orbitals:
        r = lat2cart(lattice,ionpos[orb.ionpos_indx])
        plt.scatter(r[0], r[1],c=colors[int(orb.isFlake)])
        if plotNeighborLines:
            for orb_n in orb.neighbors:
                r_n = lat2cart(lattice,ionpos[orb_n.ionpos_indx])
                plt.plot([r[0],r_n[0]],[r[1],r_n[1]],c='orange',alpha=0.1)
    plt.gca().set_aspect('equal')
    plt.show()
    
# Since hoppings to adjacent periodic images of a unitcell are stored at different parts of wannierH
#     in order to get the hopping between two orbitals, the cartesian displacement between them (dr)
#     must be specified in order to uniquely identify which unitcell orb2 is supposed to be in
#     (orb1 is always assumed to be in the central unitcell)
def getWanHop(orb1, orb2, dr, ionpos, lattice, Hwannier, cellMapDict):
        # The "wanneir centers index" field is used to index the hamiltonian 
        i, j = orb1.centers_indx, orb2.centers_indx
        orb2_imagePos     = ionpos[orb1.ionpos_indx]+ cart2lat(lattice,dr) # location of the image of orb2 (may not be in the central u.c.)
        orb2_centralUcPos = ionpos[orb2.ionpos_indx] # central unit cell location of orb2
        dR = orb2_imagePos - orb2_centralUcPos # This should be integer to machine precision
        dR = tuple( (np.rint(dR)).astype(int) ) # Round to nearest integer and cast to tuple
        c = cellMapDict[dR]
        return Hwannier[c][i][j]

# expectedShift is for sanity checking
def writeNNHoppings(path, flake_uc_pat, subs_uc_pat, wanPat, neighborhoodRadius, sourceRadius, targetRadius, expectedShift):
    outpath = path
    
    # Flake rotation is hardcoded for this demonstration,
    #   but could easily be read from files and used as argument of this function
    theta = 30*DEGREES
    Flake_Rotation = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                               [np.sin(theta),  np.cos(theta), 0.0],
                               [0.0          ,  0.0          , 1.0]])

    # Load in the wannier hamiltonian information
    nCells,nBands,cellMap,Hwannier = loadHWannier(path,wanPat)
    # Convert the cellMap into a dictionary to find cell index from Bravais Lattice Vectors
    cellMapDict = getCellMapDict(cellMap)

    # Load in the wannier function centers
    wannierCenters = readWannierCenters(path,wanPat)
    nOrbitals = len(wannierCenters)

    # Load in the structure: ionpos, lattice
    #    also load in the unitcells for the flake and substrate materials
    ionpos, ionsp = get_ionpos_ionsp(path+"/system.ionpos")
    M = get_LatticeMatrix(path+"/system.lattice")
    # Load the substrate unitcell
    subs_uc_ionpos, flake_uc_ionsp = get_ionpos_ionsp(subs_uc_pat+".ionpos")
    subs_uc_lattice = get_LatticeMatrix(subs_uc_pat+".lattice")
    # Load the flake unitcell
    flake_uc_ionpos, flake_uc_ionsp = get_ionpos_ionsp(flake_uc_pat+".ionpos")
    flake_uc_lattice = get_LatticeMatrix(flake_uc_pat+".lattice")
    # Rotate the flake unitcell
    flake_uc_lattice = np.dot(Flake_Rotation, flake_uc_lattice)

    # Load in ionpos labels differentiated by site symmetry information ionsp.aux
    symIonsps, ucIndxs, isFlakes = readIonspAux(path+"/ionsp.aux")
    unique_symIonsp = np.unique(symIonsps)
    #print(unique_symIonsp)

    # For each orbital i, create an Orbital object to organize information about this orbital
    #    set its centers_indx, ionpos_indx, and isFlake
    orbitals = []
    for i in range(nOrbitals):
        # Find the closest ionic center to this orbital
        # Since these are in lattice coordinates use the 3D indentity matrix as the latticeVector matrix
        ionpos_indx = argmin_distPBC( lat2cart(M,ionpos), lat2cart(M,wannierCenters[i]), np.eye(3))
        # Find the index of the ion in the flake/subs material unitcell that corresponds to this orbital
        #   using the information from ionsp.aux
        # Do the same for whether the orbital is in the flake or the substrate
        # Exclude orbitals with negative ionpos labels (indicating they are from terminating atoms)
        if ucIndxs[ionpos_indx] >= 0:
            orb = Orbital(i, ionpos_indx, ucIndxs[ionpos_indx], isFlakes[ionpos_indx])
            orb.neighborhoodR = neighborhoodRadius
            orbitals.append( orb )

    # Find the neighbors of each orbital
    for orb_i in orbitals:
        # get location in cartesian coords
        ri = lat2cart(M, ionpos[orb_i.ionpos_indx])
        # find all of the other orbitals who are within the defined cylinder arround this orbital (PBC accounting!)
        for orb_j in orbitals:
            rj = lat2cart(M, ionpos[orb_j.ionpos_indx])
            # add orb_j to orb_i's list of neighbors if within the cylinder
            if mindistPBC(rj,ri,M,onlyXY=True) <= orb_i.neighborhoodR:
                (orb_i.neighbors).append(orb_j)

    # Optionally visualize nearest neighbors
    doVisualize = False
    if doVisualize:
        visualize(orbitals,ionpos,M,plotNeighborLines=True)

    # Now calculate the interlayer registry for each orbital based on atomic center locations
    for orb,i in zip(orbitals, range(len(orbitals))):
        orb.calcRegistry(ionpos,M,flake_uc_ionpos,flake_uc_lattice,subs_uc_ionpos,subs_uc_lattice)
        #print("Registry =", orb.registry)
        #print(i+2, "ionpos, registry =", lat2cart(M,ionpos[orb.ionpos_indx]), orb.registry)

    # Calculate the center of the flake 
    flakePos = [] # lattice coordinates of the proxy system
    for orb in orbitals:
        if orb.isFlake:
            flakePos.append(ionpos[orb.ionpos_indx])    
    # WARNING: This logic is valid when using MINpuT to construct systems, 
    #   but may not be with custom constructions of initial coordinates.
    #   Also take care if user employed DFT code implicitly maps lattice coordinates to range [0,1) durring relaxation
    flakeCenter = np.mean(flakePos,axis=0)
    
    # Sanity check that flake shift files are in correct directory
    print("FlakeCenter =", cart2lat(subs_uc_lattice,lat2cart(M,flakeCenter)), "[latSubs]" )
    assertFlakeCenterCloseToExpected = True
    if assertFlakeCenterCloseToExpected:
        assert( np.linalg.norm(cart2lat(subs_uc_lattice,lat2cart(M,flakeCenter))[:2] - expectedShift) < 0.01 )

    # Find all orbitals within a radius of the center and record this as the centralCluster
    # do so for source and target clusters separately
    centralSourceCluster_indxs = []
    centralTargetCluster_indxs = []
    for orb,i in zip(orbitals, range(len(orbitals))):
        # get location in cartesian coords
        ri = lat2cart(M, ionpos[orb.ionpos_indx])
        # Find the PBC distance to the flakeCenter in the XY plane
        d = mindistPBC(ri,lat2cart(M,flakeCenter), M, onlyXY=True)
        # Add this index to a cluster if d < clusterRadius
        if d < sourceRadius:
            centralSourceCluster_indxs.append(i)
        if d < targetRadius:
            centralTargetCluster_indxs.append(i)
    #print("centralSourceCluster_indxs", centralSourceCluster_indxs)
    #print("centralTargetCluster_indxs", centralTargetCluster_indxs)
    
    ionpos_cart = lat2cart(M,ionpos) 
    
    # Optionally visualize the central cluster
    if doVisualize:
        # Visualize these central atoms:
        for orb in orbitals:
            plt.scatter(ionpos_cart[orb.ionpos_indx,0],ionpos_cart[orb.ionpos_indx,1],c='black')
        # Plot target cluster  
        for i in centralTargetCluster_indxs:
            plt.scatter(ionpos_cart[i,0], ionpos_cart[i,1], c='red')
        # Plot source cluster
        for i in centralSourceCluster_indxs:
            plt.scatter(ionpos_cart[i,0], ionpos_cart[i,1], c='cyan')
        #plt.legend()
        plt.show()
        
    # Finally, for each orbital part of the central cluster:
    # Write its neighborhood hopping information to an output file
    # (ion_id, orbital_id, registry), dr, (ion_id, orbital_id, registry), t
    with open(outpath+"/wannier.NNhoppings", 'w') as file:
        first_line = f"ion_id  orb1_id  registry[x]  registry[y]  dr_x[bohr]   dr_y[bohr]   dr_z[bohr]   ion_id  orb2_id  registry[x]  registry[y]  t(orb1,orb2)\n"
        #print(first_line)
        file.write(first_line)
        for i in centralSourceCluster_indxs:
            orb_i = orbitals[i]
            for j in centralTargetCluster_indxs:
                orb_j = orbitals[j]
                
                # Check in j is neighbor of i
                if orb_j in orb_i.neighbors:
                    #print("orb_i and orb_j are neighbors!")
                
                    # calculate the closest displacement vector between orb and all periodic images of orb_n
                    dr = mindiffPBC(lat2cart(M,ionpos[orb_j.ionpos_indx]), lat2cart(M,ionpos[orb_i.ionpos_indx]), M)
                    # Calculate the TB hopping from orbital to its neighbor based on the closest displacement
                    t = getWanHop(orb_i, orb_j, dr, ionpos, M, Hwannier, cellMapDict)
                    # Write all information as a line in "wannier.NNhoppings"
                    line = f"{symIonsps[i]:<7} orb1_id  {orb_i.registry[0]:10.8f} {orb_i.registry[1]:12.8f} {dr[0]:+13.8f} {dr[1]:+12.8f} {dr[2]:+12.8f}  {symIonsps[orb_j.ionpos_indx]:<7} orb2_id  {orb_j.registry[0]:10.8f} {orb_j.registry[1]:12.8f} {t:+13.8f}\n"
                    #print(line)
                    file.write(line)
                    
                
# Variables to be specified by the user
###########################################################################
# Specify file locations
global_path = "../data/30DEG_TBG/"
shift_paths = ["shift_00000_00000/","shift_00000_16666/","shift_00000_33333/","shift_00000_50000/","shift_00000_66666/","shift_00000_83333/",
               "shift_16666_00000/","shift_16666_16666/","shift_16666_33333/","shift_16666_50000/","shift_16666_66666/","shift_16666_83333/",
               "shift_33333_00000/","shift_33333_16666/","shift_33333_33333/","shift_33333_50000/","shift_33333_66666/","shift_33333_83333/",
               "shift_50000_00000/","shift_50000_16666/","shift_50000_33333/","shift_50000_50000/","shift_50000_66666/","shift_50000_83333/",
               "shift_66666_00000/","shift_66666_16666/","shift_66666_33333/","shift_66666_50000/","shift_66666_66666/","shift_66666_83333/",
               "shift_83333_00000/","shift_83333_16666/","shift_83333_33333/","shift_83333_50000/","shift_83333_66666/","shift_83333_83333/"]

flakeSize_paths = ["C13H09", "C22H12", "C37H15"]

flake_uc_pat = global_path+"graphene" 
subs_uc_pat  = global_path+"graphene"
wanPat = "/wannier"
neighborhoodRadius = 7.5 # bohr  (4NN)
sourceRadius       = 3.0 # bohr  (1NN)
targetRadius       = neighborhoodRadius
    
# Expected shifts for sanity checking:
expectedShifts = np.array([[0.00000, 0.00000],[0.00000, 0.16666],[0.00000, 0.33333],[0.00000, 0.50000],[0.00000, 0.66666],[0.00000, 0.83333],
                           [0.16666, 0.00000],[0.16666, 0.16666],[0.16666, 0.33333],[0.16666, 0.50000],[0.16666, 0.66666],[0.16666, 0.83333],
                           [0.33333, 0.00000],[0.33333, 0.16666],[0.33333, 0.33333],[0.33333, 0.50000],[0.33333, 0.66666],[0.33333, 0.83333],
                           [0.50000, 0.00000],[0.50000, 0.16666],[0.50000, 0.33333],[0.50000, 0.50000],[0.50000, 0.66666],[0.50000, 0.83333],
                           [0.66666, 0.00000],[0.66666, 0.16666],[0.66666, 0.33333],[0.66666, 0.50000],[0.66666, 0.66666],[0.66666, 0.83333],
                           [0.83333, 0.00000],[0.83333, 0.16666],[0.83333, 0.33333],[0.83333, 0.50000],[0.83333, 0.66666],[0.83333, 0.83333] ])
    
###########################################################################
    
    
# MAIN CODE:
    
# Calculate and write wannier.NNhoppings for all specified paths
for shift_path, expectedShift in zip(shift_paths,expectedShifts):
    ts_shift = []
    for flakeSize_path in flakeSize_paths:
        path = global_path+shift_path+flakeSize_path
        print()
        print(path)
        writeNNHoppings(path, flake_uc_pat, subs_uc_pat, wanPat, neighborhoodRadius, sourceRadius, targetRadius, expectedShift)
            
print("DONE!")
