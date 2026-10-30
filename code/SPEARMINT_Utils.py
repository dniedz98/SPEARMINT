# Contents:
# def grep()
# def get_LatticeMatrix()
# def get_ionpos_ionsp()
# def readIonspAux()
# def cart2lat()
# def lat2cart()
# def generateCellMap()
# def argmin_distPBC()
# def mindistPBC()
# def mindiffPBC()
# def compactify()
# def findLocalOrigin()
# class Orbital:

import numpy as np
import re

# return a list of all lines in filename with pattern
def grep(pattern, filename):
    retval = [] 
    foundPattern = False
    file = open(filename, "r")
    for line in file:
        if re.search(pattern, line):
            retval.append(line)
            foundPattern = True
    if not foundPattern:
        print("grep failed to find:", pattern, "in", filename)
        print("returning an empty list")
    return retval

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

# path+"/"+infilename
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

def readIonspAux(path2file):
    symIonsps = []
    ucIndxs   = []
    isFlakes  = []
    with open(path2file, 'r') as file:
        for line in file:
            if "#" not in line:
                parts = line.split()  # Split the line by whitespace
                if len(parts) == 3:  # Ensure there are exactly two columns
                    symIonsps.append(parts[0])
                    ucIndxs.append(int(parts[1]))  # Convert column to integer
                    isFlakes.append(int(parts[2]))  # Convert column to integer
    return symIonsps, ucIndxs, isFlakes

# Coordinate system conversions and supercell
def lat2cart(M,ionpos):
    return np.dot(ionpos,np.transpose(M))
        
def cart2lat(M,ionpos):
    return np.dot(ionpos,np.linalg.inv(np.transpose(M)))

def generateCellMap(Nx,Ny,Nz):
    retval = []
    for i in range(-Nx,Nx+1):
        for j in range(-Ny,Ny+1):
            for k in range(-Nz,Nz+1):
                retval.append([i,j,k])
    return np.array(retval)

# Assumes all values are in 3D cartesian coordinates
# returns the index of A whose position is closest to b when considering periodic boundary conditions
def argmin_distPBC(A,b,M, onlyXY=False):
    cM = lat2cart(M,generateCellMap(1,1,1)) # [-1,-1,0] to [1,1,0]
    min_indx = -1
    min_diff = np.inf
    for i in range(len(A)):
        diffs = (A[i] + cM) - b
        if onlyXY:
            diffs[:,:,2] = 0 # project displacement vectors onto the x-y plane
        least_diff = np.min(np.linalg.norm( diffs, axis=-1))
        if least_diff < min_diff:
            min_indx = i
            min_diff = least_diff
    return min_indx

# Assumes all values are in 3D cartesian coordinates
# onlyXY=True gives the closest XY projected distance between a and b
def mindistPBC(a,b,M, onlyXY=False):
    cM = lat2cart(M,generateCellMap(1,1,1)) # [-1,-1,0] to [1,1,0]
    diffs = (a + cM) - b
    if onlyXY:
        diffs[:,2] = 0 # project displacement vectors onto the x-y plane
    least_dist = np.min(np.linalg.norm( diffs, axis=-1))
    return least_dist

# Assumes all values are in 3D cartesian coordinates
# onlyXY=True gives the closest XY projected distance between a and b
# From b to a
def mindiffPBC(a,b,M, onlyXY=False):
    cM = lat2cart(M,generateCellMap(1,1,1)) # [-1,-1,0] to [1,1,0]
    diffs = (a + cM) - b
    if onlyXY:
        diffs[:,2] = 0 # project displacement vectors onto the x-y plane
    least_dist_indx = np.argmin(np.linalg.norm( diffs, axis=-1))
    return diffs[least_dist_indx]

# This function assumes the input is in lattice coordinates
# It translates a set of vectors on the periodic unit box
#   s.t. they are in their most compact form
#   either -0.5 < ax < 0.5 or 0 < ax < 1
def compactify(ionpos, axis=2):
    # 1) 0 < z_i < 1
    ionpos_01 = np.copy(ionpos)
    ionpos_01[:,axis] = np.mod(ionpos_01[:,axis]+2.0, 1.0)
    ax_avg_01 = np.mean(ionpos_01[:,axis])
    ax_stdev_01 = np.std(ionpos_01[:,axis])
    #ionpos_01[:,axis] -= ax_avg_01 # This line was in MINpuT, but shouldn't be here
    # 2) -0.5 < z < 0.5
    ionpos_mp = np.copy(ionpos)
    ionpos_mp[:,axis] = np.mod(ionpos_mp[:,axis]+0.5, 1.0) - 0.5
    ax_avg_mp = np.mean(ionpos_mp[:,axis])
    ax_stdev_mp = np.std(ionpos_mp[:,axis])
    #ionpos_mp[:,axis] -= ax_avg_mp # This line was in MINpuT, but shouldn't be here 
    if ax_stdev_01 < ax_stdev_mp:
        return ionpos_01
    else:
        return ionpos_mp
        

# return in fractional coords of the MINT-cell
def findLocalOrigin(orb, ionpos, lattice, uc_ionpos, uc_lattice):
    # vector location of the orbital in the MINT-cell in bohr
    r  = lat2cart(lattice, ionpos[orb.ionpos_indx])
    # vector displacement from this orbital's unitcell location to the origin of that unitcell 
    dr = -lat2cart(uc_lattice, uc_ionpos[orb.uc_indx])
    return cart2lat(lattice, r + dr)

# A class to hold information about an orbital
class Orbital:  
    def __init__(self, centers_indx, ionpos_indx, unitcell_indx, isFlake):
        self.centers_indx = centers_indx # Used to index wannierCenters and wannierHamiltonian
        self.ionpos_indx  = ionpos_indx  # Used to index the ion centers, ionsp, and symmetrically unique ionsp labels (ionpos.aux)
        self.isFlake      = isFlake      # Boolean: True-> is part of the flake, False -> is part of the substrate
        self.uc_indx      = unitcell_indx # Used to index the unitcell of material this orbital belongs to
        # A 2D vector on the open square [0,1) x [0,1)
        # if part of the flake, then "registry" is a vector within the substrate unitcell
        # if part of the substrate, then "registry" is a vector within the flake unitcell
        self.registry     = np.array([0.0,0.0]) # Initialize registry to zero
        self.neighbors    = [] # A list of other orbitals considered to be this one's neighbors
        # The neighborhood is defined as a cylinder with radius R ( in future maybe even from [-Z/2 , Z/2] )
        self.neighborhoodR = 0.0 #
        #self.neighborhoodZ = 0.0 #
        
    # An interlayer registry is only unique w.r.t. a chosen lattice+basis
    #    therefore if one wishes to compare them accross multiple different calculations,
    #    then the reference lattice+basis MUST be consistent
    # NOTE: Assumes the c-axis is orthogonal to the a-b plane
    def calcRegistry(self,ionpos,lattice,flakeIonpos,flakeLattice,subsIonpos,subsLattice):
        # Find this orbital's location in lattice coordinates of the MINT-cell
        r = ionpos[self.ionpos_indx]

        # Find what the other layer's lattice is
        if self.isFlake:
            otherlayerLattice = subsLattice
            otherlayerIonpos  = subsIonpos
        else:
            otherlayerLattice = flakeLattice
            otherlayerIonpos  = flakeIonpos
        
        # For each of this orbital's neighbors in the other layer calculate a predicted registry
        predicted_registries = []
        dists = []
        
        for orb_n in self.neighbors:
            # If it's in the other layer
            if orb_n.isFlake != self.isFlake:
                
                # Find the origin of its local unitcell in fractional coords of the MINT-cell
                local_origin = findLocalOrigin(orb_n, ionpos, lattice, otherlayerIonpos, otherlayerLattice) 
                
                # The registry is this orbital's location minus the local origin % unitcell
                registry = r - local_origin # project onto 2D later
                # Change units to fractional coordinates of the otherlayerLattice
                registry = cart2lat( otherlayerLattice , lat2cart(lattice, registry) )
                # Take the modulus on the unitbox
                registry = np.mod(registry, [1,1,1])
                
                # Add registry to the list
                predicted_registries.append(registry)
                
                # Use distance between this orbital and its neighbor as a weight
                dr = r-ionpos[orb_n.ionpos_indx] # TODO: account from PBC
                dists.append( np.linalg.norm(lat2cart(lattice,  dr )) )
                
        predicted_registries = np.array(predicted_registries)
        dists = np.array(dists)
        
        # If no neighbors in the other layer were found (e.g. uncovered substrate)
        #    set the registry to NaN, else perform the weighted sum
        if len(dists) == 0:
            #print("setting to nan")
            registry = [np.nan,np.nan]
        else:
            # Perform a weighted average of these predicted registries (accounting for PBC)
            #    use the distance to the NN's as weights (closer atoms matter more!)
            weights = np.exp(-0.25*dists)

            # first compactify in x and y; this accounts for PBC wrapping
            predicted_registries = compactify( compactify(predicted_registries, axis=0), axis=1)

            # then perform the weighted average 
            #registry = np.sum(weights*predicted_registries, axis=0) / np.sum(weights)
            registry = np.dot(weights,predicted_registries) / np.sum(weights)

            # finally modulo the registry and project onto the first two lattice vectors
            registry = np.mod(registry, [1,1,1]) [0:2]
        # update this orbital's registry field
        self.registry = registry