import numpy as np
import matplotlib.pyplot as plt
import astropy
import astropy.units as u
import astropy.constants as const

# This functions implement analytic formulae for stellar evolution presented in Hurley, Pols, and Tout 2000 (MNRAS 315, 543-569)


def get_Z(xi):
	return 0.02 * 10.**xi


# ================ Coefficients from Appendix A ================================== #


def a_coeff(xi,n):

	# ---------- tMS + LTMS: ---------------

	if n == 1:
		params = [1.593890e3, 2.053038e3, 1.231226e3, 2.327785e2, 0.]

	if n == 2:
		params = [2.706708e3, 1.483131e3, 5.772723e2, 7.411230e1, 0.]

	if n == 3:
		params = [1.466143e2, -1.048442e2, -6.795374e1, -1.391127e1, 0.]

	if n == 4:
		params = [4.141960e-2, 4.564888e-2, 2.958542e-2, 5.571483e-3, 0.]

	if n == 5:
		params = [3.426349e-1, 0., 0., 0., 0.]

	if n == 6:
		params = [1.949814e1, 1.758178e0, -6.008212e0, -4.470533e0, 0.]

	if n == 7:
		params = [4.903830e0, 0., 0., 0., 0.]

	if n == 8:
		params = [5.212154e-2, 3.166411e-2, -2.750074e-3, -2.271549e-3, 0.]

	if n == 9:
		params = [1.312179e0, -3.294936e-1, 9.231860e-2, 2.610989e-2, 0.]

	if n == 10:
		params = [8.073972e-1, 0., 0., 0., 0.]

	if n == 11:
		params = [1.031538e0, -2.434480e-1, 7.732821e0, 6.460705e0, 1.374484e0]

	if n == 12:
		params = [1.043715e0, -1.577474e0, -5.168234e0, -5.596506e0, -1.299394e0]

	if n == 13:
		params = [7.859573e2, -8.542048e0, -2.642511e1, -9.585707e0, 0.]

	if n == 14:
		params = [3.858911e3, 2.459681e3, -7.630093e1, -3.486057e2, -4.861703e1]

	if n == 15:
		params = [2.888720e2, 2.952979e2, 1.850341e2, 3.797254e1, 0.]

	if n == 16:
		params = [7.196580e0, 5.613746e-1, 3.805871e-1, 8.398728e-2, 0.]

	# --------- RTMS ----------------

	if n == 17:
		Z = get_Z(xi)
		sigma = np.log10(Z)
		loga = max(0.097 - 0.1072*(sigma+3), max(0.097, min(0.1461, 0.1461 + 0.1237*(sigma+2))))
		return 10.**loga

	if n == 18:
		params = [2.187715e-1, -2.154437e0, -3.768678e0, -1.975518e0, -3.021475e-1]

	if n == 19:
		params = [1.466440e0, 1.839725e0, 6.442199e0, 4.023635e0, 6.957529e-1]

	if n == 20:
		params = [2.652091e1, 8.178458e1, 1.156058e2, 7.633811e1, 1.950698e1]

	if n == 21:
		params = [1.472103e0, -2.947609e0, -3.312828e0, -9.945065e-1, 0.]

	if n == 22:
		params = [3.071048e0, -5.679941e0, -9.745523e0, -3.594543e0, 0.]

	if n == 23:
		params = [2.617890e0, 1.019135e0, -3.292551e-2, -7.445123e-2, 0.]

	if n == 24:
		params = [1.075567e-2, 1.773287e-2, 9.610479e-3, 1.732469e-3, 0.]

	if n == 25:
		params = [1.476246e0, 1.899331e0, 1.195010e0, 3.035051e-1, 0.]

	if n == 26:
		params = [5.502535e0, -6.601663e-2, 9.968707e-2, 3.599801e-2, 0.]

	# ----------- LBGB --------------------

	if n == 27:
		params = [9.511033e1, 6.819618e1, -1.045625e1, -1.474939e1, 0.]

	if n == 28:
		params = [3.113458e1, 1.012033e1, -4.650511e0, -2.463185e0, 0.]

	if n == 29:
		params = [1.413057e0, 4.578814e-1, -6.850581e-2, -5.588658e-2, 0.]

	if n == 30:
		params = [3.910862e1, 5.196646e1, 2.264970e1, 2.873680e0, 0.]

	if n == 31:
		params = [4.597479e0, -2.855179e-1, 2.709724e-1, 0., 0.]

	if n == 32:
		params = [6.682518e0, 2.827718e-1, -7.294429e-2, 0., 0.]

	# --------- Delta L ----------------

	if n == 33:
		a = min(1.4, 1.5135 + 0.3769*xi)
		return max(0.6355 - 0.4192*xi, max(1.25, a))

	if n == 34:
		params = [1.910302e-1, 1.158624e-1, 3.348990e-2, 2.599706e-3, 0.]

	if n == 35:
		params = [3.931056e-1, 7.277637e-2, -1.366593e-1, -4.508946e-2, 0.]

	if n == 36:
		params = [3.267776e-1, 1.204424e-1, 9.988332e-2, 2.455361e-2, 0.]

	if n == 37:
		params = [5.990212e-1, 5.570264e-2, 6.207626e-2, 1.777283e-2, 0.]

	# ----------- Delta R --------------

	if n == 38:
		params = [7.330122e-1, 5.192827e-1, 2.316416e-1, 8.346941e-3, 0.]

	if n == 39:
		params = [1.172768e0, -1.209262e-1, -1.193023e-1, -2.859837e-2, 0.]

	if n == 40:
		params = [3.982622e-1, -2.296279e-1, -2.262539e-1, -5.219837e-2, 0.]

	if n == 41:
		params = [3.571038e0, -2.223625e-2, -2.611794e-2, -6.359648e-3, 0.]

	if n == 42:
		params = [1.9848e0, 1.1386e0, 3.5640e-1, 0., 0.]

	if n == 43:
		params = [6.300e-2, 4.810e-2, 9.840e-3, 0., 0.]

	if n == 44:
		params = [1.200e0, 2.450e0, 0., 0., 0.]

	# ----------- alpha L --------------

	if n == 45:
		params = [2.321400e-1, 1.828075e-3, -2.232007e-2, -3.378734e-3, 0.]

	if n == 46:
		params = [1.163659e-2, 3.427682e-3, 1.421393e-3, -3.710666e-3, 0.]

	if n == 47:
		params = [1.048020e-2, -1.231921e-2, -1.686860e-2, -4.234354e-3, 0.]

	if n == 48:
		params = [1.555590e0, -3.223927e-1, -5.197429e-1, -1.066441e-1, 0.]

	if n == 49:
		params = [9.7700e-2, -2.3100e-1, -7.5300e-2, 0., 0.]

	if n == 50:
		params = [2.4000e-1, 1.8000e-1, 5.9500e-1, 0., 0.]

	if n == 51:
		params = [3.3000e-1, 1.3200e-1, 2.1800e-1, 0., 0.]

	if n == 52:
		params = [1.1064e0, 4.1500e-1, 1.8000e-1, 0., 0.]

	if n == 53:
		params = [1.1900e0, 3.7700e-1, 1.7600e-1, 0., 0.]

	# ----------- beta L --------------

	if n == 54:
		params = [3.855707e-1, -6.104166e-1, 5.676742e0, 1.060894e1, 5.284014e0]

	if n == 55:
		params = [3.579064e-1, -6.442936e-1, 5.494644e0, 1.054952e1, 5.280991e0]

	if n == 56:
		params = [9.587587e-1, 8.777464e-1, 2.017321e-1, 0., 0.]

	if n == 57:
		a = min(1.4, 1.5135 + 0.3769*xi)
		return max(0.6355 - 0.4192*xi, max(1.25, a))

	# ---------- alpha R ---------------

	if n == 58:
		params = [4.907546e-1, -1.683928e-1, -3.108742e-1, -7.202918e-2, 0.]

	if n == 59:
		params = [4.537070e0, -4.465455e0, -1.612690e0, -1.623246e0, 0.]

	if n == 60:
		params = [1.796220e0, 2.814020e-1, 1.423325e0, 3.421036e-1, 0.]

	if n == 61:
		params = [2.256216e0, 3.773400e-1, 1.537867e0, 4.396373e-1, 0.]

	if n == 62:
		params = [8.4300e-2, -4.7500e-2, -3.5200e-2, 0., 0.]

	if n == 63:
		params = [7.3600e-2, 7.4900e-2, 4.4260e-2, 0., 0.]

	if n == 64:
		params = [1.3600e-1, 3.5200e-2, 0., 0., 0.]

	if n == 65:
		params = [1.564231e-3, 1.653042e-3, -4.439786e-3, -4.951011e-3, -1.216530e-3]

	if n == 66:
		params = [1.4770e0, 2.9600e-1, 0., 0., 0.]

	if n == 67:
		params = [5.210157e0, -4.143695e0, -2.120870e0, 0., 0.]

	if n == 68:
		params = [1.1160e0, 1.6600e-1, 0., 0., 0.]

	# ---------- beta R ---------------

	if n == 69:
		params = [1.071489e0, -1.164852e-1, -8.623831e-2, -1.582349e-2, 0.]

	if n == 70:
		params = [7.108492e-1, 7.935927e-1, 3.926983e-1, 3.622146e-2, 0.]

	if n == 71:
		params = [3.478514e0, -2.585474e-2, -1.512955e-2, -2.833691e-3, 0.]

	if n == 72:
		params = [9.132108e-1, -1.653695e-1, 0., 3.636784e-2, 0.]

	if n == 73:
		params = [3.969331e-3, 4.539076e-3, 1.720906e-3, 1.897857e-4, 0.]

	if n == 74:
		params = [1.600e0, 7.640e-1, 3.322e-1, 0., 0.]

	# ----------- gamma --------------

	if n == 75:
		params = [8.109e-1, -6.282e-1, 0., 0., 0.]

	if n == 76:
		params = [1.192334e-2, 1.083057e-2, 1.230969e0, 1.551656e0, 0.]

	if n == 77:
		params = [-1.668868e-1, 5.818123e-1, -1.105027e1, -1.668070e1, 0.]

	if n == 78:
		params = [7.615495e-1, 1.068243e-1, -2.011333e-1, -9.371415e-2, 0.]

	if n == 79:
		params = [9.409838e0, 1.522928e0, 0., 0., 0.]

	if n == 80:
		params = [-2.7110e-1, -5.7560e-1, -8.3800e-2, 0., 0.]

	if n == 81:
		params = [2.4930e0, 1.1475e0, 0., 0., 0.]

	# -------------------------

	alpha, beta, gamma, eta, mu = params

	a = alpha + beta*xi + gamma*xi**2. + eta*xi**3. + mu*xi**4.

	if n == 11:
		return a * a_coeff(xi, 14)
	elif n == 12:
		return a * a_coeff(xi, 14)

	elif n == 18:
		return a * a_coeff(xi,20)
	elif n == 19:
		return a * a_coeff(xi,20)

	elif n == 29:
		return a**a_coeff(xi,32)

	elif n == 42:
		return min(1.25, max(1.1, a))
	elif n == 44:
		return min(1.3, max(0.45, a))

	elif n == 49:
		return max(a, 0.145)
	elif n == 50:
		return min(a, 0.306 + 0.053*xi)
	elif n == 51:
		return min(a, 0.3625 + 0.062*xi)
	elif n == 52:
		Z = get_Z(xi)
		a = max(a, 0.9)
		if Z > 0.01:
			return min(a, 1.0)
		else:
			return a
	elif n == 53:
		Z = get_Z(xi)
		a = max(a, 1.0)
		if Z > 0.01:
			return min(a, 1.1)
		else:
			return a

	elif n == 62:
		return max(0.065, a)
	elif n == 63:
		Z = get_Z(xi)
		if Z < 0.004:
			return min(0.055, a)
		else:
			return a
	elif n == 64:
		a = max( 0.091, min(0.121, a) )
		if a_coeff(xi, 68) > (a_coeff(xi, 66)):
			a = alpha_R(a_coeff(xi, 66), xi)
		return a
	elif n == 66:
		a = max(a, min(1.6, -0.308 - 1.046*xi))
		return max(0.8, min(0.8 - 2.0*xi, a))
	elif n == 68:
		return max(0.9, min(a, 1.0))

	elif n == 72:
		Z = get_Z(xi)
		if Z > 0.01:
			return max(a, 0.95)
		else:
			return a
	elif n == 74:
		return max(1.4, min(a, 1.6))

	elif n == 75:
		a = max(1.0, min(a, 1.27))
		return max(a, 0.6355 - 0.4192*xi)
	elif n == 76:
		return max(a, -0.1015564 - 0.2161264*xi - 0.05182516*xi**2.)
	elif n == 77:
		return max(-0.3868776 - 0.5457078*xi -0.1463472*xi**2., min(0.0, a))
	elif n == 78:
		return max(0.0, min(a, 7.454 + 9.046*xi))
	elif n == 79:
		return min(a, max(2.0, -13.3 - 18.6*xi))
	elif n == 80:
		return max(0.0585542, a)
	elif n == 81:
		return min(1.5, max(0.4, a))

	else:
		return a


# ====================== Main Sequence Lifetime ===================== #


def x_param(xi):
	# Eq. 6:
	return max( 0.95, min(0.95 - 0.03*(xi + 0.30103), 0.99) )

def Mu_param(M,xi):
	# Eq. 7:
	return max( 0.5, 1.0 - 0.01*max( a_coeff(xi,6)/(M**a_coeff(xi,7)), a_coeff(xi,8) + a_coeff(xi,9)/(M**a_coeff(xi,10)) ) )

def t_BGB(M,xi):
	# Base of the Giant Branch (BGB) lifetime, Eq. 4:
	return (a_coeff(xi,1) + a_coeff(xi,2)*M**4. + a_coeff(xi,3)*M**5.5 + M**7.) / (a_coeff(xi,4)*M**2. + a_coeff(xi,5)*M**7.)

def t_hook(M, xi):
	return Mu_param(M, xi) * t_BGB(M, xi)

def t_MS(M,xi):
	# Main Sequence (MS) lifetime, Eq. 5:
	return max( t_hook(M, xi), x_param(xi)*t_BGB(M,xi) )

def M_hook(xi):
	# Initial mass above which a hook appears in the MS, Eq. 1:
	return 1.0185 + 0.16015 * xi + 0.0892 * xi**2.

def M_HeF(xi):
	# Maximum initial mass for which He ignites degenerately in a helium flash, Eq. 2:
	return 1.995 + 0.25*xi + 0.087*xi**2.

def M_FGB(Z):
	# Maximum initial mass for which He ignites on the first giant branch, Eq. 3:
	return (13.048 * (Z/0.02)**0.06) / (1. + 0.0012 * (0.02/Z)**1.27 )


# ================= Zero-age Main-sequence (ZAMS) radii and luminosities, from Tout, Pols, Eggleton and Han 1996 (MNRAS 281, 257-262) ==================== #


def lum_coeff_matrix():

	# Coefficients for ZAMS luminosity, Table 1:
	row1 = [ 0.39704170,  -0.32913574,  0.34776688,  0.37470851, 0.09011915 ]
	row2 = [ 8.52762600, -24.41225973, 56.43597107, 37.06152575, 5.45624060 ]
	row3 = [ 0.00025546,  -0.00123461, -0.00023246,  0.00045519, 0.00016176 ]
	row4 = [ 5.43288900,  -8.62157806, 13.44202049, 14.51584135, 3.39793084 ]
	row5 = [ 5.56357900, -10.32345224, 19.44322980, 18.97361347, 4.16903097 ]
	row6 = [ 0.78866060,  -2.90870942,  6.54713531,  4.05606657, 0.53287322 ]
	row7 = [ 0.00586685,  -0.01704237,  0.03872348,  0.02570041, 0.00383376 ]

	return np.matrix( [row1, row2, row3, row4, row5, row6, row7] )

def rad_coeff_matrix():

	# Coefficients for ZAMS Radius, Table 2:
	row1 = [  1.71535900,  0.62246212,  -0.92557761,  -1.16996966, -0.30631491 ]
	row2 = [  6.59778800, -0.42450044, -12.13339427, -10.73509484, -2.51487077 ]
	row3 = [ 10.08855000, -7.11727086, -31.67119479, -24.24848322, -5.33608972 ]
	row4 = [  1.01249500,  0.32699690,  -0.00923418,  -0.03876858, -0.00412750 ]
	row5 = [  0.07490166,  0.02410413,   0.07233664,   0.03040467,  0.00197741 ]
	row6 = [  0.01077422,  0.        ,   0.        ,   0.        ,  0.         ]
	row7 = [  3.08223400,  0.94472050,  -2.15200882,  -2.49219496, -0.63848738 ]
	row8 = [ 17.84778000, -7.45345690, -48.96066856, -40.05386135, -9.09331816 ]
	row9 = [  0.00022582, -0.00186899,   0.00388783,   0.00142402, -0.00007671 ]

	return np.matrix( [row1, row2, row3, row4, row5, row6, row7, row8, row9] )

def coeff(Z, params):

	assert np.size(params) == 5

	a, b, c, d, e = np.squeeze(np.asarray(params))

	# Coefficients, Eq. 3:

	Zsun = 0.02
	x = np.log10(Z/Zsun)

	return a + b*x + c*x**2. + d*x**3. + e*x**4.

def L_ZAMS(M, Z):

	# ZAMS Luminosity, Eq. 1:

	mx = lum_coeff_matrix()

	ms = np.sqrt(M)

	num = coeff(Z, mx[0,:])*M**5.*ms + coeff(Z, mx[1,:])*M**11.
	den = coeff(Z, mx[2,:]) + M**3. + coeff(Z, mx[3,:])*M**5. + coeff(Z, mx[4,:])*M**7. + coeff(Z, mx[5,:])*M**8. + coeff(Z, mx[6,:])*M**9.*ms

	return num / den

def R_ZAMS(M, Z):

	# ZAMS Radius, Eq. 2:

	mx = rad_coeff_matrix()
	ms = np.sqrt(M)

	num = (coeff(Z, mx[0,:])*M**2. + coeff(Z, mx[1,:])*M**6.)*ms + coeff(Z, mx[2,:])*M**11. + (coeff(Z, mx[3,:]) + coeff(Z, mx[4,:])*ms)*M**19.
	den = coeff(Z, mx[5,:]) + coeff(Z, mx[6,:])*M**2. + (coeff(Z, mx[7,:])*M**8. + M**18. + coeff(Z, mx[8,:])*M**19.)*ms

	return num / den

'''

Figures 2, 3

M = np.linspace(0.1, 100, 1000)

L = L_ZAMS(M,0.02)
L1 = L_ZAMS(M, 0.0003)

plt.plot( np.log10(M), np.log10(L) )
plt.plot( np.log10(M), np.log10(L1) , '--')
plt.xlim([-1,2])
plt.ylim([-3, 7])
plt.grid()
plt.show()

R = R_ZAMS(M,0.02)
R1 = R_ZAMS(M, 0.001)

plt.plot( np.log10(M), np.log10(R) )
plt.plot( np.log10(M), np.log10(R1) , '--')
plt.xlim([-1,2])
plt.ylim([-1.1, 1.3])
plt.grid()
plt.show()
'''

# ============== Luminosity as a function of time in the MS ====================================


def Luminosity(M, xi, t): # Luminosity as a function of time, Eq. 12


	Z = get_Z(xi)

	# Fractional time scale on the MS, Eq. 11
	tau = t / t_MS(M, xi)

	tau1, tau2 = tau_12(M, xi, t)

	LZAMS = L_ZAMS(M, Z)

	LTMS = L_TMS(M, xi)

	eta = eta_exp(M, Z)

	DeltaL = Delta_L(M, xi)

	alphaL = alpha_L(M, xi)
	betaL  =  beta_L(M, xi)

	logratio = alphaL*tau + betaL*tau**eta + ( np.log10(LTMS/LZAMS) - alphaL - betaL )*tau**2. - DeltaL*( tau1**2. - tau2**2. )

	return LZAMS * 10.**logratio

#Luminosity = np.vectorize(Luminosity)


def tau_12(M, xi, t):

	thook = t_hook(M, xi)

	eps = 0.01

	# Eq. 14:
	tau1 = min(1.0, t / thook)

	# Eq. 15:
	tau2 = max( 0.0, min( 1.0, (t - (1.0 - eps)*thook)/(eps*thook) ) )

	return tau1, tau2


def alpha_L(M, xi):

	# Luminosity alpha Coefficient

	# Eq. 19a:
	if M >= 2.0:
		return (a_coeff(xi,45) + a_coeff(xi,46)*M**a_coeff(xi,48)) / (M**0.4 + a_coeff(xi,47)*M**1.9)

	# Eq. 19b:
	if M < 0.5:
		return a_coeff(xi,49)

	if (M >= 0.5) & (M < 0.7):
		return a_coeff(xi,49) + 5.0 * (0.3 - a_coeff(xi,49)) * (M - 0.5)

	if (M >= 0.7) & (M < a_coeff(xi,52)):
		return 0.3 + (a_coeff(xi,50) - 0.3)*(M - 0.7)/(a_coeff(xi,52) - 0.7)

	if (M >= a_coeff(xi,52)) & (M < a_coeff(xi,53)):
		return a_coeff(xi,50) + (a_coeff(xi,51) - a_coeff(xi,50))*(M - a_coeff(xi,52))/(a_coeff(xi,53) - a_coeff(xi,52))

	if (M >= a_coeff(xi,53)) & (M < 2.0):

		B = alpha_L(2.0, xi)

		return a_coeff(xi,51) + (B - a_coeff(xi,51))*(M - a_coeff(xi,53))/(2.0 - a_coeff(xi,53))

def beta_L(M, xi):

	# Luminosity beta Coefficient, Eq. 20:

	beta = max( 0.0, a_coeff(xi,54) - a_coeff(xi,55)*M**a_coeff(xi,56) )

	if (M > a_coeff(xi,57)) & (beta > 0.0):

		B = beta_L(a_coeff(xi, 57), xi)

		beta = max(0.0, B - 10.0*(M - a_coeff(xi,57))*B)

	return beta

def eta_exp(M, Z):

	# Exponent eta, Eq. 18:

	if Z <= 0.0009:

		if M <= 1.0:
			eta = 10
		elif M >= 1.1:
			eta = 20
		else:
			eta = np.interp( M, [1.0, 1.1], [10., 20.])

	else:
		eta = 10

	return eta


def Delta_L(M, xi):

	# Luminosity perturbation, Eq. 16:

	Mhook = M_hook(xi)

	if M <= Mhook:
		return 0.

	elif (M > Mhook) & (M < a_coeff(xi,33)):

		B = Delta_L( a_coeff(xi,33), xi)

		return B * ( (M - Mhook)/(a_coeff(xi,33) - Mhook) )**0.4

	elif M >= a_coeff(xi, 33):

		return min( a_coeff(xi,34) / (M**a_coeff(xi,35)), a_coeff(xi,36) / (M**a_coeff(xi,37)) )


def L_TMS(M, xi):

	# Luminosity at the end of the MS, Eq. 8:

	return (a_coeff(xi,11)*M**3. + a_coeff(xi,12)*M**4. + a_coeff(xi,13)*M**(a_coeff(xi,16)+1.8) ) / (a_coeff(xi,14) + a_coeff(xi,15)*M**5. + M**a_coeff(xi,16))


def L_BGB(M, xi):

	c2 = 9.301992
	c3 = 4.637345

	# Luminosity at the base of the GB, Eq. 10:

	return (a_coeff(xi,27)*M**a_coeff(xi,31) + a_coeff(xi,28)*M**c2) / (a_coeff(xi,29) + a_coeff(xi,30)*M**c3 + M**a_coeff(xi,32))

'''
Z = 0.02
xi = np.log10(Z/0.02)

M = 1.25

t = np.linspace(0., t_MS(M, xi), 5000)
Lum = Luminosity(M, xi, t)

plt.plot( t/t_MS(M, xi), np.log10(Lum) )
plt.xlim([0,1.05])
plt.xlabel('tau', fontsize=18)
plt.ylabel('log L/LSun', fontsize=18)
plt.show()
'''

# ============== Radius as a function of time in the MS ====================================


def Radius(M, xi, t): # Radius as a function of time, Eq. 12

	Z = get_Z(xi)

	# Fractional time scale on the MS, Eq. 11
	tau = t / t_MS(M, xi)

	tau1, tau2 = tau_12(M, xi, t)

	RZAMS = R_ZAMS(M, Z)

	RTMS = R_TMS(M, xi)

	alphaR = alpha_R(M, xi)
	betaR  =  beta_R(M, xi)
	gammaR  = gamma_R(M, xi)

	DeltaR = Delta_R(M, xi)

	logratio = alphaR*tau + betaR*tau**10. + gammaR*tau**40. + (np.log10(RTMS/RZAMS) - alphaR - betaR - gammaR)*tau**3. - DeltaR*( tau1**3. - tau2**3. )

	R = RZAMS * 10.**logratio

	# Low-mass MS stars can be degenerate, Eq.24:

	if M < M_HeF(xi):

		X = 0.76 - 3.0*Z # Initial idrogen abundance

		R = max(R, 0.0258 * (1.0 + X)**(5./3.) * M**(-1./3.))

	return R

#Radius = np.vectorize(Radius)

def alpha_R(M, xi):

	# Radius alpha Coefficient

	a68 = min(a_coeff(xi,68), a_coeff(xi,66))

	# Eq. 21a:
	if (M >= a_coeff(xi,66)) & (M <= a_coeff(xi,67)):
		return (a_coeff(xi,58) * M**a_coeff(xi,60)) / (a_coeff(xi,59) + M**a_coeff(xi,61))

	# Eq. 21b:
	if M < 0.5:
		return a_coeff(xi,62)

	if (M >= 0.5) & (M < 0.65):
		return a_coeff(xi,62) + (a_coeff(xi,63) - a_coeff(xi,62)) * (M - 0.5) / 0.15

	if (M >= 0.65) & (M < a68):

		return a_coeff(xi,63) + (a_coeff(xi,64) - a_coeff(xi,63)) * (M - 0.65) / (a68 - 0.65)

	if (M >= a68) & (M < a_coeff(xi,66)):

		B = alpha_R( a_coeff(xi, 66), xi)

		return a_coeff(xi,64) + (B - a_coeff(xi,64)) * (M - a68) / (a_coeff(xi,66) - a68)

	if M > a_coeff(xi,67):

		C = alpha_R( a_coeff(xi,67), xi)

		return C + a_coeff(xi,65) * (M - a_coeff(xi,67))


def beta_R(M, xi):

	# Radius beta coefficient

	# Eq. 22a:
	if (M >= 2.0) & (M <= 16.0):
		beta1 = (a_coeff(xi,69) * M**3.5) / (a_coeff(xi,70) + M**a_coeff(xi,71))

	# Eq. 22b:
	if M <= 1.0:

		beta1 = 1.06

	if (M > 1.0) & (M < a_coeff(xi,74)):

		beta1 = 1.06 + (a_coeff(xi,72) - 1.06) * (M - 1.0) / (a_coeff(xi,74) - 1.06)

	if (M >= a_coeff(xi,74)) & (M < 2.0):

		B = beta_R(2.0, xi) + 1

		beta1 = a_coeff(xi,72) + (B - a_coeff(xi,72)) * (M - a_coeff(xi,74)) / (2.0 - a_coeff(xi,74))

	if M > 16.0:

		C = beta_R(16.0, xi) + 1

		beta1 = C + a_coeff(xi,73) * (M - 16.0)

	return beta1 - 1


def gamma_R(M, xi):

	# Radius Gamma coefficient, Eq. 23:

	if M > a_coeff(xi,75) + 0.1:
		gamma = 0.0

	elif M <= 1.0:
		gamma = a_coeff(xi,76) + a_coeff(xi,77) * (M - a_coeff(xi,78))**a_coeff(xi,79)

	elif (M > 1.0) & (M <= a_coeff(xi,75)):

		B = gamma_R(1.0, xi)

		gamma = B + (a_coeff(xi,80) - B) * ( (M - 1.0)/(a_coeff(xi,75) - 1.0) )**a_coeff(xi,81)

	elif (M > a_coeff(xi,75)) & (M < a_coeff(xi,75) + 0.1):

		if a_coeff(xi,75) == 1.0:

			C = gamma_R(1.0, xi)

		else:
			C = a_coeff(xi,80)

		gamma = C - 10.0 * (M - a_coeff(xi,75)) * C

	#if gamma < 0:
	#	print 'ERROR: gamma < 0'
	#	gamma = 0.0
		#pass

	return gamma


def Delta_R(M, xi):

	# Radius perturbation, Eq. 17:

	Mhook = M_hook(xi)

	if M <= Mhook:

		return 0.

	elif (M > Mhook) & (M <= a_coeff(xi,42)):

		return a_coeff(xi,43) * ( (M - Mhook) / (a_coeff(xi,42) - Mhook) )**0.5

	elif (M > a_coeff(xi,42)) & (M < 2.0):

		B = Delta_R(2.0, xi)

		return a_coeff(xi,43) + (B - a_coeff(xi,43)) * ( (M - a_coeff(xi,42)) / (2.0 - a_coeff(xi,42) ))**a_coeff(xi,44)

	elif M >= 2.0:

		return (a_coeff(xi,38) + a_coeff(xi,39)*M**3.5) / (a_coeff(xi,40)*M**3. + M**a_coeff(xi,41)) - 1.0


def R_TMS(M, xi):

	# Radius at the end of the MS

	Mstar = a_coeff(xi,17) + 0.1
	c1 = -8.672073e-2

	# Eq. 9a:
	if M <= a_coeff(xi,17):

		RTMS =  (a_coeff(xi,18) + a_coeff(xi,19)*M**a_coeff(xi,21)) / (a_coeff(xi,20) + M**a_coeff(xi,22))

	elif M >= Mstar:

		RTMS = (c1*M**3. + a_coeff(xi,23)*M**a_coeff(xi,26) + a_coeff(xi,24)*M**(a_coeff(xi,26) + 1.5)) / (a_coeff(xi,25) + M**5.)

	else:

		M1 = a_coeff(xi,17)
		R1 = R_TMS(M1, xi)

		M2 = Mstar
		R2 = R_TMS(Mstar, xi)

		RTMS = np.interp( M, [M1, M2], [R1, R2]) # straight-line interpolation

		if M < 0.5:

			Z = get_Z(xi)

			RTMS = max(RTMS, 1.5 * R_ZAMS(M, Z))

	return RTMS


'''
Rad = Radius(M, xi, t)

plt.plot( t/t_MS(M, xi), np.log10(Rad) )
plt.xlim([0,1.05])
plt.xlabel('tau', fontsize=18)
plt.ylabel('log R/RSun', fontsize=18)
plt.show()
'''

# ================= Mass Loss ( Section 7 ) =================== #


def Mdot_NJ(M, L, R, Z):

	# Mass-Loss prescription given by Nieuwenhuijzen & De Jager (1990), Paragraph 7

	return 9.6e-15 * (Z/0.02)**0.5 * R**0.81 * L**1.24 * M**0.16 # [Msun / yr]


def Mdot_LBV(L, R):

	# LBV-like mass-loss for stars beyond the Humphreys-Davidson limit, Paragraph 7

	return 0.1 * (1.e-5 * R * L**0.5  - 1.0)**3. * ( L / 6.e5 - 1.0) # [Msun / yr]


def Mdot_WR(mu, L):

	# Wolf-Rayet-like mass loss, for small hydrogen-evelope mass (mu < 1.0), Paragraph 7

	if mu < 1.0:
		return 1.e-13 * L**1.5 * (1.0 - mu) # [Msun / yr]
	else:
		return 0.

def Mass_Loss(M, L, R, xi, t):

	Z = get_Z(xi)

	Mdot = Mdot_WR(1., L)

	if L > 4000.:
		Mdot = max(Mdot, Mdot_NJ(M, L, R, Z) )

	if (L > 6.e5) & (1.e-5 * R * L**0.5 > 1.0):
		Mdot = Mdot + Mdot_LBV(L, R)

	Mloss = Mdot * t*1.e6 # [MSun]

	return M - Mloss

#Mass_Loss = np.vectorize(Mass_Loss)

# ========================


def T_eff(R, L, t):

	R = R * u.solRad
	L = L * u.L_sun

	return ((L / (4.*np.pi*R**2.*const.sigma_sb))**(1./4.)).to(u.K).value # Effective Temperature, [K]
'''
Teff = T_eff(Rad, Lum, t)

plt.plot( t/t_MS(M, xi), Teff )
plt.xlim([0,1.05])
plt.xlabel('tau', fontsize=18)
plt.ylabel('Teff [K]', fontsize=18)
plt.show()
'''

#  ==================== General Function to Get Stellar Parameters =========================


def get_TempRad(M, xi, t):

	L = Luminosity(M, xi, t)
	R = Radius(M, xi, t)

	T = T_eff(R, L, t)

	#Mt = Mass_Loss(M0, L, R, xi, t)

	#t = t * t_MS(Mt, xi)/t_MS(M0,xi)

	#L = Luminosity(Mt, xi, t)
	#R = Radius(Mt, xi, t)

	return T, R

'''
path = '/net/vuntus/data2/marchetti/HVS_Gaia_Prediction/'
mass, age, radius = np.loadtxt(path + 'M_tAge_R_grid_Mmin=0.1_allt.txt', unpack = True)
mass, age, temperature = np.loadtxt(path + 'M_tAge_T_grid_Mmin=0.1_allt.txt', unpack = True)
mass, age, mass_t = np.loadtxt(path + 'M_tAge_M_grid_Mmin=0.1_allt.txt', unpack = True)

mytemp = np.empty(len(mass))
myrad  = np.empty(len(mass))
mymass = np.empty(len(mass))

for i in range(0, len(mass)):
	mytemp[i], myrad[i], mymass[i] = get_TRM(mass[i], 0., age[i])
'''
