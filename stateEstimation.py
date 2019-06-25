# -------------------------------------------------
# UNIVERSITY OF SAO PAULO
# SÃO CARLOS SCHOOL OF ENGINEERING (EESC)
# DEPARTMENT OF ELECTRICAL AND COMPUTER ENGINEERING (SEL)
# AUTHOR: Álvaro Augusto Volpato
# DATE: 04/07/2018
# VERSION: 5.0
# DESCRIPTION: this program is a state estimator for an electric power system.
# The program first tests for observability, restores it from available pseudo-measures if needed, and pro-
# ceeds to find critical measurements. Next the program estimates the network's states through a Newton-
# -Raphson method proposed in MONTICELLI(1995), and tests for normalized residue.
# -------------------------------------------------

# -------------------------------------------------
# (1) IMPORTING LIBRARIES  {{{1
# -------------------------------------------------
# (1.1) Numpy libraries for standard operations  {{{2
import numpy as np
norm = np.linalg.norm
abs = np.absolute
cos = np.cos
sin = np.sin
array = np.array
real = np.real
imag = np.imag
conc = np.concatenate
transpose = np.transpose
inv = np.linalg.inv
sqrt = np.sqrt
pi = np.pi

# (1.2) SCIPY FOR LU DECOMPOSITION 
import scipy
import scipy.linalg
LU = scipy.linalg.lu

# (1.3) COPY library for temporary variable deep copy 
from copy import deepcopy
copy = deepcopy

# (1.4) OS and SYS libraries for defining clear terminal command 
import os
import sys

# (1.5) Time library: allows to count execution time 
import time

# (1.6) Argument Parser to pass arguments to script  2
import argparse
parser = argparse.ArgumentParser(
	prog = 'Power flow calculator',
	description = 'Power flow calculator that takes a system data in the IEEE CDF and a measurements file in the LACO CDF.',
	)

# -------------------------------------------------
# (2) CUSTOM FUNCTIONS AND ARGUMENTS{{{1
# -------------------------------------------------
# (2.1) Set printout options: this was added to allow program printout to span the whole terminal 
np.set_printoptions(suppress=False,linewidth=999,threshold=999)

# (2.2) "diag" function: takes a n-sized vector and output an n by n diagonal matrix which diagonal is composed of the input vector's elements 
def diag(dW):
	W = np.zeros((len(dW),len(dW)))
	for i in range(len(dW)): W[i,i] = dW[i]
	return W

# (2.3) Clear  function: clears the terminal. Similar to MATLAB's cls 
def clear(): os.system('cls' if os.name == 'nt' else 'clear')
#clear()
#clear()

# (2.3) "pause" function: halts program execution and asks for enter 
def pause(string):
    programPause = input(string)

# (2.4) Arguments and options 
# Creates arguments:
# --net [NETWORK FILE] is the target system network file. This string is stored in a 'networkFile' variable
parser.add_argument(	"net",
			type = str,
			metavar = 'NETFILE',
			help='Target network file')

# --meas [MEASUREMENTS FILE] is the target system measurements file. This string is stored in a 'measurementsFile' variable
parser.add_argument(	"meas",
			type = str,
			metavar = 'MEASFILE',
			help='Target measurements file.')

# Creates options:

# VERBOSE option
parser.add_argument(	'--verbose', '-v',
			type = int,
			choices = [0,1,2],
			default = 1,
			nargs = 1,
			help = 'Verbose level. 0 for no output, 1 for critical output, 2 for detailed output. Default is 1.')

# OBS option
parser.add_argument(	'--obs',
			default = False,
			action = 'store_true',
			help = 'Performs observability check. If the system is deemed not observable, the program tries to add the supplied pseudo-measures from the measurements file. Default value is false.')

# CRIT option
parser.add_argument(	'--crit',
			default = False,
			action = 'store_true',
			help = 'Performs critical measurements check. Default is false.')

# CLEAR option
parser.add_argument(	'--cls',
			default = False,
			action = 'store_true',
			help = 'Clears the terminal before executing the program.')

# Gathering arguments and options
args = parser.parse_args()

verbose = args.verbose[0]
measurementsFile = args.meas
networkFile = args.net
obs = args.obs
crit = args.crit
cls = args.cls

if clear:
	clear()
	clear()

# -------------------------------------------------
# (3) DESCRIBRING SYSTEM: READING FILE {{{1
# -------------------------------------------------

# (3.1) READING NETWORK FILE AND BUILDING TARGET SYSTEM DATA {{{2
with open(networkFile,'r') as fileData:

	breakFlag = True

	# (3.1.2) Acquiring bus data parameters
	print('\n --> Beginning bus data reading... ',end='' )

	# Searching for bus data start card
	while breakFlag==True:
		line = fileData.readline()
		if line[0:16] == 'BUS DATA FOLLOWS':
			
			nbars = int(line[18:20])	# Extracting number of buses in system
			breakFlag = False

	vDesired = np.empty(14)
	angleDesired = np.empty(14)
	bsh = np.zeros((nbars,nbars))
	gsh = np.zeros((nbars,nbars))

	breakFlag = True
	i = 0
	while breakFlag == True:
		line = fileData.readline()
		if line[0:2] == '-9':
			breakFlag  = False

		if breakFlag == True:
			vDesired[i] = float(line[27:32])
			angleDesired[i] = pi/180*float(line[33:40])
			bsh[i,i] = float(line[114:122])
			gsh[i,i] = float(line[106:114])
			i += 1

	# (3.1.3) Searching for branch data start card
	print(' Done.\n --> Beginning branch data reading... ',end='' )

	while breakFlag==True:
		line = fileData.readline()
		if line[0:18] == 'BRANCH DATA FOLLOWS':
			breakFlag = False

	# (3.1.4) Acquiring branch data parameters
	fileData.readline()
	
	# Initializing resistance and reactance matrixes
	r = np.zeros((nbars,nbars))
	x = np.zeros((nbars,nbars))

	# Initializing tap levels matrix
	a = np.ones((nbars,nbars))
	
	# Initializing connection matrix
	K = [ [] for i in range(nbars)]

	breakFlag = True
	i = 0
	while breakFlag == True:
		line = fileData.readline()
		if line[0:2] == '-9':
			breakFlag  = False

		if breakFlag == True:
			fromBus = int(float(line[0:4]))
			toBus = int(float(line[5:9]))

			K[fromBus-1] += [toBus - 1]
			K[toBus-1] += [fromBus - 1]

			r[fromBus-1,toBus-1] = float(line[19:29])
			r[toBus-1,fromBus-1] = float(line[19:29])

			x[fromBus-1,toBus-1] = float(line[29:40])
			x[toBus-1,fromBus-1] = float(line[29:40])

			bsh[fromBus-1,toBus-1] = float(line[40:50])	
			bsh[toBus-1,fromBus-1] = float(line[40:50])

			if float(line[76:82]) != 0:
				a[fromBus-1,toBus-1] = 1/float(line[76:82])

			i += 1

	# (3.1.5) Building Y matrix
	print(' Done.\n --> Building Y matrix... ',end='')
	
	z = array([[r[m,n] + 1j*x[m,n] for m in range(nbars)] for n in range(nbars)])
	y = array([[1/z[m,n] if z[m,n] != 0 else 0 for m in range(nbars)] for n in range(nbars)])

	g = real(copy(y))
	b = imag(copy(y))

	bsh = array([[ bsh[m,n]/2 if m!=n else bsh[m,n] for n in range(nbars)] for m in range(nbars)])

	Y  = -a*transpose(a)*y

	for k in range(nbars): Y[k,k] = 1j*bsh[k,k] + sum([a[k,m]**2*y[k,m] + 1j*bsh[k,m] for m in K[k]])

	G = real(copy(Y))
	B = imag(copy(Y))
	print(' Done.' )


# END OF FILE MANIPULATION: CLOSING FILE --------

# (3.2) ACQUIRING POWER MEASUREMENTS {{{2
with open(measurementsFile,'r') as fileData:

	breakFlag = True

	# (3.2.1) Searching for start card

	while breakFlag==True:
		line = fileData.readline()
		for k in range(len(line)-31):
			if line[k:k+31] == 'BEGGINING ACTIVE POWER MEASURES': breakFlag = False

	print(' --> Beginning active power measurements data reading... ',end='' )

	fileData.readline()
	isP = np.zeros((nbars,nbars),dtype=int)
	wP = np.zeros((nbars,nbars))
	P = np.zeros((nbars,nbars))

	# (3.2.2) Acquiring active power measures
	breakFlag = True
	while breakFlag == True:
		line = fileData.readline()
		if line[0:2] == '-9':
			breakFlag  = False

		if breakFlag == True:
			fromBus = int(float(line[0:3]))
			toBus = int(float(line[4:7]))
			isP[fromBus-1,toBus-1] = 1
			P[fromBus-1,toBus-1] = float(line[8:19])
			wP[fromBus-1,toBus-1] = float(line[20:31])	

	# (3.2.3) Acquiring reactive power measures
	print(' Done.\n --> Beginning reactive power measurements data reading... ',end='' )

	for i in range(3): fileData.readline()
	wQ = np.zeros((nbars,nbars))
	Q = np.zeros((nbars,nbars))

	breakFlag = True
	while breakFlag == True:
		line = fileData.readline()
		if line[0:2] == '-9':
			breakFlag  = False

		if breakFlag == True:
			fromBus = int(float(line[0:3]))
			toBus = int(float(line[4:7]))
			Q[fromBus-1,toBus-1] = float(line[8:19])
			wQ[fromBus-1,toBus-1] = float(line[20:31])	

	# (3.2.4) Acquiring voltage measures
	print(' Done.\n --> Beginning voltage measurements data reading... ',end='' )

	for i in range(3): fileData.readline()
	isV = np.zeros(nbars,dtype=int)
	wV = np.zeros(nbars)
	Vm = np.zeros(nbars)
	breakFlag = True
	while breakFlag == True:
		line = fileData.readline()
		for k in range(len(line)-1):
			if line[k:k+2] == '-9':
				breakFlag  = False
				break

		if breakFlag == True:
			fromBus = int(float(line[0:3]))
			isV[fromBus-1] = 1
			Vm[fromBus-1] = float(line[8:19])
			wV[fromBus-1] = float(line[20:31])	

	# (3.2.5) Acquiring available pseudo-measures
	if obs:
		print(' Done.\n --> Beginning available pseudo-measurements data reading... ',end='' )
		for i in range(4): fileData.readline()

		# Preallocating isPseudo with one row which will be deleted after
		isPseudo = np.zeros((1,2))
		pseudoP = np.zeros((nbars,nbars))
		pseudoWP = np.zeros((nbars,nbars))
		pseudoQ = np.zeros((nbars,nbars))
		pseudoWQ = np.zeros((nbars,nbars))

		breakFlag = True
		while breakFlag == True:
			line = fileData.readline()
			if line[0:2] == '-9':
				breakFlag  = False

			if breakFlag == True:
				fromBus = int(float(line[0:3]))
				toBus = int(float(line[4:7]))
				isPseudo = conc((isPseudo,array([[fromBus-1,toBus-1]])),axis=0)
				pseudoP[fromBus-1,toBus-1] = float(line[8:19])
				pseudoWP[fromBus-1,toBus-1] = float(line[20:30])	
				pseudoQ[fromBus-1,toBus-1] = float(line[30:40])
				pseudoWQ[fromBus-1,toBus-1] = float(line[40:50])

		# Deleting first row from preallocating
		isPseudo = np.delete(isPseudo,0,axis=0)

	print(' Done.\n')

# END OF FILE MANIPULATION: CLOSING FILE --------

#(3.2.6) Counting P and V readings
numP = isP.sum() # Number of P readings
numV = isV.sum()

# -------------------------------------------------
# (4) TESTING OBSERVABILITY {{{1
# -------------------------------------------------

if obs:
	# (3.3.1) Building Hdelta and gqain matrix
	if verbose > 0: print('\n' + '-'*50 + '\n TESTING OBSERVABILITY \n' + '-'*50)
	else: print('\n --> Beggining observability test.',end='')

	obs = np.zeros((numP,nbars),dtype=float)

	i = 0
	for m in range(nbars):
		for n in range(nbars):
			if isP[m,n]:
				if m == n:
					for k in range(nbars):
						if k != m and k in K[m]: obs[i,k] = -1
						elif k == m: obs[i,k] = len(K[m])
				else:
					for k in range(nbars):
						if k==m: obs[i,k] = 1
						if k==n: obs[i,k] = -1
				i += 1

	gain = transpose(obs)@obs

	# (3.3.2) Printing results
	if verbose > 2: print('\n --> H matrix: H = \n{0}'.format(obs))

	Pt,Lt,Ut = LU(obs)
	dU = array([Ut[k,k] for k in range(Ut.shape[0])])

	if verbose > 2: print('\n --> Upper-triangularized linear-model jacobian H: HTri = \n{0}'.format(Ut))
	#print('\n --> Main diagonal of U: diag(U) = \n{0}'.format(dU))

	if verbose > 2: print('\n --> Gain matrix: GTri = \n{0}'.format(gain))

	Pt,Lt,Ut = LU(gain)
	dU = array([Ut[k,k] for k in range(Ut.shape[0])])

	if verbose > 1: print('\n --> Upper-triangular factorization of G: U = \n{0}'.format(Ut))
	#print('\n --> Main diagonal of U: diag(U) = \n{0}'.format(dU))

	# (3.3.3) Checking for null pivots and restoring observability from avalable pseudos
	zeroPivots = 0	# Number of null pivots: should be one in order for the network to be observable
	for k in range(dU.shape[0]):
		if abs(dU[k]) < 1e-8:
			zeroPivots += 1

	if zeroPivots == 1:
		print('\n --> The system seems to be observable: single null pivot detected.')
	else:
		# Restoring observability from available pseudo-measures
		pause('\n --> The system appears not to be observable: there were {0} null pivots. Attempting to restore observability from available pseudo-measures.'.format(zeroPivots))

		# Copy isP
		tempIsP = copy(isP)
		while zeroPivots != 1:
			for pseudoTestCounter in range(isPseudo.shape[0]):
				fromBus = int(isPseudo[pseudoTestCounter,0])
				toBus = int(isPseudo[pseudoTestCounter,1])
				
				# Adding pseudo-measure to the measures roster
				tempIsP[fromBus,toBus] = 1

				try:
					addedPseudoNumber = addedPseudos.shape[1]
				except NameError:
					addedPseudoNumber = 0

				# Building new observability and gain matrices
				obs = np.zeros((numP + addedPseudoNumber + 1,nbars))

				i = 0
				for m in range(nbars):
					for n in range(nbars):
						if tempIsP[m,n]:
							if m == n:
								for k in range(nbars):
									if k != m and k in K[m]: obs[i,k] = -1
									elif k == m: obs[i,k] = len(K[m])
							else:
								for k in range(nbars):
									if k==m: obs[i,k] = 1
									if k==n: obs[i,k] = -1
							i += 1

				gain = transpose(obs)@obs

				Pt,Lt,Ut = LU(gain)
				dU = array([Ut[k,k] for k in range(Ut.shape[0])])

				# Checking for null pivots
				tempZeroPivots = 0
				for k in range(dU.shape[0]):
					if abs(dU[k]) < 1e-8:
						tempZeroPivots += 1

				# Now, if the new pseudo-measure added a new information, add this measure to the set
				if tempZeroPivots < zeroPivots:
					zeroPivots = tempZeroPivots
					if verbose > 0: print(' --> Measure ({0:3},{1:3}) added a new non-null pivot. This measure is added to the measures roster.'.format(fromBus+1,toBus+1))
					try:
						addedPseudos = conc((addedPseudos,array([[fromBus,toBus]])),axis=0)
					except NameError:
						addedPseudos = array([[fromBus,toBus]])
				# But if it did not add anu information, remove this measure
				else: 
					tempIsP[fromBus,toBus] = 0
					if verbose > 0: print(' --> Measure ({0:3},{1:3}) did not add new non-null pivot. Discarding this measure.'.format(fromBus+1,toBus+1))
					if pseudoTestCounter == isPseudo.shape[0]:
						sys.exit('--> Fatal error: the available pseudo-measures were not able to restore observability!')

		# (3.3.4) Updating system measurement parameters and printing added pseudos
		# Updating P, Q and V measurements with pseudos
		isP = copy(tempIsP)

		P += pseudoP
		wP += pseudoWP
		Q += pseudoQ
		wQ += pseudoWQ
		numP = isP.sum() # Number of P readings
		numV = isV.sum()

		print('\n --> Summary: added pseudo-measures: \n{0}'.format(addedPseudos + np.ones((addedPseudos.shape[0],addedPseudos.shape[1]))))

# -------------------------------------------------
# (5) TESTING FOR CRITICAL MEASUREMENTS {{{1
# -------------------------------------------------

if crit:
	if verbose > 0: print('\n' + '-'*50 + '\n TESTING FOR CRITICAL MEASURES \n' + '-'*50)
	else: print('\n --> Testing for critical measures.', end = '')

	tempIsP = copy(isP)

	# Sweeping through available measurements
	for p in range(nbars):
		for q in range(nbars):
			if isP[p,q]:
				if verbose > 0: print(' --> Removing measure ({0:3},{1:3})...'.format(p+1,q+1), end='')
				# Temporarily removing measure (p,q)
				tempIsP[p,q] = 0

				# Building observability matrix again
				obs = np.zeros((numP-1,nbars),dtype=float)

				i = 0
				for m in range(nbars):
					for n in range(nbars):
						if tempIsP[m,n]:
							if m == n:
								for k in range(nbars):
									if k != m and k in K[m]: obs[i,k] = -1
									elif k == m: obs[i,k] = len(K[m])
							else:
								for k in range(nbars):
									if k==m: obs[i,k] = 1
									if k==n: obs[i,k] = -1
							i += 1

				# Calculating gain matrix and triangularizing
				gain = transpose(obs)@obs
				Pt,Lt,Ut = LU(gain)
				dU = array([Ut[k,k] for k in range(Ut.shape[0])])

				# Testing for null pivots
				zeroPivots = 0
				for k in range(dU.shape[0]):
					if abs(dU[k]) < 1e-8:
						zeroPivots += 1

				if zeroPivots > 1:
					print('\n  !-> The system has {2:3} null pivots. Measure ({0:3},{1:3}) is critical.\n'.format(p+1,q+1,zeroPivots))

					# Try to add this measure to the critical measure roster
					try:
						criticalMeas = conc((criticalMeas,array([[p,q]])),axis=0)
					# In case the array criticalMeas does not exist, catch exception and create it
					except NameError:
						criticalMeas = array([[p,q]])

				elif zeroPivots == 1:
					if verbose > 0: 
						print(' Gain matrix has exactly one null pivot.')
				elif zeroPivots == 0: print('\n\n\n ERROR: zero null pivots detected!')

				tempIsP[p,q] = 1

	# Try to print critical measurement list
	try:
		print('\n --> Summary: list of critical measurements: \n{0}'.format(criticalMeas + np.ones((criticalMeas.shape[0],criticalMeas.shape[1]))))
	# In case no critical measurement was found, then this will throw an error
	except NameError:
		print('\n --> No critical measurements were found.')

# -------------------------------------------------
# (3.5) CALCULATING FLAT START
# -------------------------------------------------
			
# (3.5.1) CONSTRUCTING W MATRIX
W = np.empty((2*numP + numV,1))			# Preallocating
i = 0						# Starting counter
for j in range(nbars):	
	for n in range(nbars):
		if isP[j,n]:
			W[i] = 1/wP[j,n]**2
			i+=1
for j in range(nbars):	
	for n in range(nbars):
		if isP[j,n]:
			W[i] = 1/wQ[j,n]**2
			i += 1

for j in range(nbars):	
	if isV[j]:
		W[i] = 1/wV[j]**2
		i += 1
W = diag(W)

# (3.5.2) DESCRIBING MEASUREMENT ARRAY Z = [P,Q,V].
if verbose > 1: pause('-'*50 + '\n DESCRIBING MEASURING ARRAY\n' + '-'*50)
i=0
Z = np.empty((2*numP + numV,1))
for j in range(nbars):	
	for n in range(nbars):
		if isP[j,n]:
			Z[i] = P[j,n]
			if verbose > 1: print('Z[{0:2.0f}] = P[{1:3},{2:3}] = {3:>10f}'.format(i+1,j+1,n+1,P[j,n]))
			i+=1
for j in range(nbars):	
	for n in range(nbars):
		if isP[j,n]:
			Z[i] = Q[j,n]
			if verbose > 1: print('Z[{0:2.0f}] = Q[{1:3},{2:3}] = {3:>10f}'.format(i+1,j+1,n+1,Q[j,n]))
			i += 1

for j in range(nbars):	
	if isV[j]:
		Z[i] = Vm[j]
		if verbose > 1: print('Z[{0:2.0f}] = V[{1:3}]     = {2:>10f}'.format(i+1,j+1,Vm[j]))
		i += 1

# (3.5.3) PRINTING RESULTS
if verbose > 1:
	print('\n' + '-'*50 + '\n DESCRIBING SYSTEM \n' + '-'*50)
	print(' -->  g = \n{2}\n\n --> b = \n{3}\n\n --> G = \n{0}\n\n --> B = \n{1}\n\n--> bsh = \n{4}\n\n --> stdP = \n{5}\n\n --> stdQ = \n{6}\n\n --> a = \n{7}\n'.format(G,B,g,b,bsh,wP,wQ,a))

# -------------------------------------------------
# (6) DEFINING MATRIX FUNCTIONS {{{1
# -------------------------------------------------

def dPdT(V,theta):
	H = np.empty((numP, nbars))	# Preallocating
	i = 0	# Starting counter
	for j in range(nbars):	
		for n in range(nbars):
			if isP[j,n]:
				for k in range(nbars):
					if j==n: # If measuring power injection
						if k==j: # If calculating on angle j
							H[i,k] = V[j]*sum([V[m]*(-G[j,m]*sin(theta[j] - theta[m]) + B[j,m]*cos(theta[j] - theta[m])) for m in K[j]])
						elif (k in K[j]): # If calculating on angle k, n and j are connected
							H[i,k] = V[j]*V[k]*(G[j,k]*sin(theta[j] - theta[k]) - B[j,k]*cos(theta[j] - theta[k]))
						else: # If n and j are not connected
							H[i,k] = 0
					else:	# If measuring power flow
						if k!=j and k!=n: # Since power transfer is function of only angles j and n
							H[i,k] = 0
						else:
							if k==j: H[i,k] = V[j]*V[n]*a[j,n]*a[n,j]*(  g[j,n]*sin(theta[j] - theta[n]) - b[j,n]*cos(theta[j] - theta[n]))
							else: H[i,k] =    V[j]*V[n]*a[j,n]*a[n,j]*( -g[j,n]*sin(theta[j] - theta[n]) + b[j,n]*cos(theta[j] - theta[n]))
							
				i+=1
	# Deleting the reference bar
	H = np.delete(H,0,axis=1)
	return H

def dPdV(V,theta):
	N = np.empty((numP, nbars))
	i = 0
	for j in range(nbars):	
		for n in range(nbars):
			if isP[j,n]:
				for k in range(nbars):
					if j==n:
						if k==j: 
							N[i,k] = 2*V[j]*G[j,j] + sum([V[m]*(G[j,m]*cos(theta[j] - theta[m]) + B[j,m]*sin(theta[j] - theta[m])) for m in K[j]])
						elif k in K[j]: 
							N[i,k] = V[j]*(G[j,k]*cos(theta[j] - theta[k]) + B[j,k]*sin(theta[j] - theta[k]))
						else: N[i,k] = 0
					else:
						if k!=j and k!=n:
							N[i,k] = 0
						else:
							if k == j: N[i,k] = 2*a[j,n]*V[j]*g[j,n] - a[j,n]*a[n,j]*V[n]*(g[j,n]*cos(theta[j] - theta[n]) + b[j,n]*sin(theta[j] - theta[n]))
							else: N[i,k] = -a[j,n]*a[n,j]*V[j]*(g[j,n]*cos(theta[j] - theta[n]) + b[j,n]*sin(theta[j] - theta[n]))
				i+=1
	return N

def dQdT(V,theta):
	M = np.empty((numP, nbars))
	i = 0
	for j in range(nbars):	
		for n in range(nbars):
			if isP[j,n]:
				for k in range(nbars):
					if j==n: # If measuring power injection
						if j==k: # If calculating on angle j
							M[i,k] = V[j]*sum([V[m]*(G[j,m]*cos(theta[j] - theta[m]) + B[j,m]*sin(theta[j] - theta[m])) for m in K[j]])
						elif (k in K[j] and k!=j):
							M[i,k] = -V[j]*V[k]*(G[j,k]*cos(theta[j] - theta[k]) + B[j,k]*sin(theta[j] - theta[k]))
						else: M[i,k] = 0
					else:
						if k!=j and k!=n: #Since power transfer is function of only angles j and n
							M[i,k] = 0
						else:
							if k == j: M[i,k] = -a[j,n]*a[n,j]*V[j]*V[n]*(g[j,n]*cos(theta[j] - theta[n]) + b[j,n]*sin(theta[j] - theta[n]))
							else: M[i,k] = a[j,n]*a[n,j]*V[j]*V[n]*(g[j,n]*cos(theta[j] - theta[n]) + b[j,n]*sin(theta[j] - theta[n]))
							
				i+=1
	M = np.delete(M,0,axis=1)
	return M

def dQdV(V,theta):
	L = np.ones((numP, nbars))
	i = 0
	for j in range(nbars):	
		for n in range(nbars):
			if isP[j,n]:
				for k in range(nbars):
					if j==n: # If measuring power injection
						if j==k: L[i,k] = -2*V[j]*B[j,j] + sum([V[m]*(G[j,m]*sin(theta[j] - theta[m]) - B[j,m]*cos(theta[j] - theta[m])) for m in K[j]])
						elif (k in K[j]): L[i,k] = V[j]*(G[j,k]*sin(theta[j] - theta[k]) - B[j,k]*cos(theta[j] - theta[k]))
						else: L[i,k] = 0
					else:
						if k!=j and k!=n: #Since power transfer is function of only voltages j and n
							L[i,k] = 0
						else:
							if k == j: L[i,k] = -2*a[j,n]**2*V[j]*(b[j,n] + bsh[j,n]) + a[j,n]*a[n,j]*V[n]*(b[j,n]*cos(theta[j] - theta[n]) - g[j,n]*sin(theta[j] - theta[n]))
							else: L[i,k] = a[j,n]*a[n,j]*V[j]*(b[j,n]*cos(theta[j] - theta[n]) - g[j,n]*sin(theta[j] - theta[n]))
				i += 1
	return L

def h(V,theta):
	h = np.zeros((2*numP + numV,1))
	i = 0
	for j in range(nbars):	
		for n in range(nbars):
			if isP[j,n]:
				if j==n: # If measuring power injection on bar j
					h[i] = V[j]**2*G[j,j] + V[j]*sum([ V[m]*(G[j,m]*cos(theta[j] - theta[m]) + B[j,m]*sin(theta[j] - theta[m])) for m in K[j]])
				else:	# If measuring power flux from bar j to n
					h[i] = V[j]**2*a[j,n]**2*g[j,n] - a[j,n]*a[n,j]*V[j]*V[n]*(g[j,n]*cos(theta[j] - theta[n]) + b[j,n]*sin(theta[j] - theta[n]))
					
				i +=1
				
	for j in range(nbars):	
		for n in range(nbars):
			if isP[j,n]:
				if j==n:
					h[i] = -V[j]**2*bsh[j,j] + sum([ -V[j]**2*a[j,m]**2*(b[j,m] + bsh[j,m]) + a[m,j]*a[j,m]*V[j]*V[m]*( -g[j,m]*sin(theta[j] - theta[m]) + b[j,m]*cos(theta[j] - theta[m]) ) for m in K[j]])
				else:
					h[i] =  -V[j]**2*a[j,n]**2*(b[j,n] + bsh[j,n]) + a[n,j]*a[j,n]*V[j]*V[n]*( -g[j,n]*sin(theta[j] - theta[n]) + b[j,n]*cos(theta[j] - theta[n]) )
				i +=1

	for j in range(nbars):	
		if isV[j]:
			h[i] = V[j]
			i += 1
	return h

def J(V,theta):	
	O = np.identity(nbars)
	deleteList = array([i for i in range(nbars) if isV[i]==0])
	O = np.delete(O,deleteList,axis=0) # Deleting non-V measures
	O = conc((np.zeros((O.shape[0],O.shape[1]-1)),O),axis=1)

	dP = conc(( dPdT(V,theta), dPdV(V,theta)),axis=1)
	dQ = conc(( dQdT(V,theta), dQdV(V,theta)),axis=1)
	dPdQ = conc((dP,dQ),axis=0)
	
	return conc((dPdQ,O),axis=0)

# -------------------------------------------------
# (7) SETTING UP NUMERICAL METHOD {{{1
# -------------------------------------------------
# (5.1) Initial guess: flat start
V = np.ones(nbars)
theta = np.zeros(nbars)

if verbose > 1:
	print('\n' + '-'*50 + '\n FLAT START \n' + '-'*50)
else:
	print(' --> Beggining flat start calculations... ', end='')

if verbose > 1:
	print('\n --> J = \n{0}\n\n --> h = \n{1}'.format(J(V,theta),h(V,theta)))

# (5.2) Defining simulation parameters
absTol = 1e-12	# Absolute tolerance
deltaMax = 10	# Maximum state step to consider as diverting
maxIter = 1000	# maximum iterations

# (5.3) Initiating iteration counter
itCount = 0	

# (5.4) Calculating jacobian and its gradient at flat start
H = J(V,theta)
grad = -transpose(H) @ W @ (Z - h(V,theta))

if verbose <= 1: print('Done.')

# (5.5) Printing method parameters

if verbose > 1:
	print('\n --> Norm of the jacobian gradient at flat start: |grad(J)| = {0}'.format(norm(grad)))

if verbose > 0:
	pause('\n --> Next step is the numerical method. Press <ENTER> to continue.')
	print('\n' + '-'*50 + '\n BEGGINING NUMERICAL METHOD \n' + '-'*50)
	print('\n --> Newton-Raphson method parameters: tolerance = {0}, maximum step = {1}, maximum interations = {2}'.format(absTol,deltaMax,maxIter))
else:
	print(' --> Beggining numerical method... ', end='')

# (5.6) Start time counte6
tstart = time.time()

# -------------------------------------------------
# (8) STARTING NUMERICAL METHOD {{{1
# -------------------------------------------------

while(True):
# (6) STARTING ITERATIONS

	# (6.1) Increasing iteration counter
	itCount += 1
	
	# (6.2) Printing iteration report
	if verbose > 0: print('\n ==> Iteration #{0:3.0f} '.format(itCount) + '-'*50)
	
	# (6.3) Calculating jacobian
	H = J(V,theta)
	
	# (6.4) Calculating gain matrix
	U = transpose(H) @ W @ H

	# (6.5) Calculating state update
	dX = inv(U) @ transpose(H) @ W @ (Z - h(V,theta))

	# (6.6) Updating V and theta
	theta[1:] += dX[0:nbars-1,0]
	V += dX[nbars-1:,0]

	# (6.7) Calculating jacobian gradient
	grad = -transpose(H) @ W @ (Z - h(V,theta))
	
	# (6.8) Printing iteration results
	if verbose > 0:
		print(' --> |dX| = {0}\n --> dV = {1}\n --> dTheta = {2}\n --> |grad(J)| = {3}'.format(norm(dX),transpose(dX[nbars-1:,0]),dX[0:nbars-1,0],transpose(norm(grad))))
	if verbose > 1:
		print('\n --> J = \n{0},\n\n --> r = Z - h = \n{2}*\n{1}'.format(J(V,theta),(Z-h(V,theta))/norm(Z-h(V,theta)),norm(Z-h(V,theta))))

	# (6.9) Testing for iteration sucess or divergence
	if norm(dX) < absTol: # (6.8.1) If success
		print('\n --> Sucess!')
		print('\n' + '-'*50 + '\n CONVERGENCE RESULTS \n' + '-'*50)
		print('\n --> Numerical method stopped at iteration {1} with |deltaX| = {0}.\n --> |grad(J)| = {2}'.format(norm(dX),itCount,norm(grad)))
		break	

	if norm(dX) > deltaMax: #(6.8.2) If diverted
		print(' --> Error: the solution appears to have diverged on iteration number {0}: |deltaX| = {1}.'.format(itCount,norm(dX)))
		break
	
	if itCount > maxIter - 1: # (6.8.3) If reached maximum iteration
		print(' --> Error: the solution appears to have taken too long. Final iteration: {0}: |deltaX| = {1}.'.format(itCount,norm(dX)))
		break

	# (6.10) Pausing for each iteration
	if verbose>1:
		pause('\n --> Program paused for next iteration. ')


# (6.11) Calculating elapsed time
elapsed = time.time() - tstart

# (6.12) Calculating normalized residual
r = Z - h(V,theta)
cov = inv(W) - H @ inv(U) @ transpose(H)
rn = array([ r[i]/sqrt(cov[i,i]) for i in range(len(r))])

# (6.13) Printing convergence results
print(' --> Final result:\n\n	theta = {0} \n\n	V     = {1}'.format(theta,V))

if verbose > 1: print('\n --> Residual = \n{2}\n\n --> Normalized residual = \n{1}\n\n --> Elapsed time: {0} s'.format(elapsed,rn,r))
