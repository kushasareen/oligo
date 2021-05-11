'''
To Do:
	- find a good way to visualize this
	- try sampling branches from sholl analysis
	- test with graph connectivity metrics
	- overhaul plotting framework with plotly
	- check out PyDSTool for n oisy models

'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import igraph as g
from scipy.integrate import ode
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import axes3d
from skimage import measure
plt.style.use('ggplot')
import plotly.graph_objects as go
import PyDSTool as dst
from scipy.signal import find_peaks

class oligo:

	"""Creates a 'oligodendrocyte' object where a branching structure can be initialized and a PDE system is simulated on branches
	'system' can be overwritten to solve many different PDEs
	
	Attributes:
	    dx (float): Spatial compartment size while discretizing
	    G (2D array of float): Second derivative matrix for graph
	    num_compart (int): Total number of discretized compartments
	    params (dict): Dictionary of relavant parameters for 'self.system'
	    sol (2D array of float): Solution in space and time of system
	    tspan (tuple of float): Time span to solve model on
	    tree (iGraph object): Graph to solve model on
	"""
	
	def __init__(self):
		"""Initialize class variables to default values
		"""

		self.tree = g.Graph(directed = True)
		self.dx = 5e-1
		self.tree.add_vertices(1)

		self.params = {'c0': [], 'h0': [], 'er0': [],
		'c1': 0.185, 'v1': 6, 'v2': 0.11, 'v3': 0.9, 'k3': 0.1, 'd1': 0.13, 
		'd2': 1.049, 'd3': 0.9434, 'd5': 0.08234, 'a2': 0.2, 'kout': 2e-3, 'vryr': 0.2, 
		'kr': 0.5, 'vncx': 2000, 'cao': 1.8, 'nao': 140, 'nai': 12, 'kna': 87.5,
		'kca': 1.38, 'ksat': 0.1, 'eta': 0.35, 'V': -67, 'F': 96.5, 'R': 8.314,
		'T': 310, 'V_cell': 0.04, 'ip3': 0.41, 'vsocc': 0.25, 'ksocc': 8,
		'cdiff': 1, 'xi': 0.1, 'gamma': 15, 'pin': 3e-4, 'hc3': 3, 'inmax': 1e-3, 'intau': 13, 
		'jint0': [], 'baseline': 0.15}

		self.num_compart = 0
		self.sol = None
		self.tspan = [0,900]
		self.G = None

	def add_branch(self, start, end, leng):
		"""Adds branch to 'self.tree' to solve on
		
		Args:
		    start (int): Start vertex ID
		    end (int): Start vertex ID
		    leng (float): Description, must be multiple of 'self.dx'
		"""

		if start >= self.tree.vcount() or end >= self.tree.vcount():
			self.tree.add_vertices(1)

		self.tree.add_edges([(start, end)])
		self.tree.es[len(self.tree.es) - 1]["leng"] = leng
		self.num_compart += int(leng/self.dx)
	
	def view_tree(self):
		""" Plots an image of the current graph structure as initialized
		Doesn't work well in .ipynb files
		"""

		layout = self.tree.layout("kk")
		g.plot(self.tree, layout = layout, vertex_label = [v.index for v in self.tree.vs])

	def ddx2(self):
		"""Computes second derivative matrix of the compartment graph of 'self.tree'
		
		Returns:
		    2D array of floats: Second derivative matrix
		"""

		if self.num_compart <=1:
			return [0]
  
		G = np.zeros((self.num_compart, self.num_compart))
		for i in range(self.num_compart):
			if i > 0:
				G[i][i-1] = 1
			G[i][i] = -2
			if i < self.num_compart - 1:
				G[i][i+1] = 1

		cs = [0, *(map(int, np.cumsum(self.tree.es["leng"])/self.dx))]

		def side(e, vid):
			"""Given a vertex e, assigns a compartment to split at according to convention
			
			Args:
			    e (iPython vertex objext): Vertex where split is happening
			    vid (int): Vertex ID
			
			Returns:
			    int: Compartment to split at
			"""
			if vid == e.tuple[0]:
				return cs[e.index]
			elif vid == e.tuple[1]:
				return cs[e.index] + int(self.tree.es["leng"][e.index]/self.dx) - 1

		def connect(a, b):
			"""Creates a connection between two compartments
			
			Args:
			    a (int): First compartment
			    b (int): Second compartment
			"""
			G[a][b] = 1
			G[b][a] = 1

		def remove(i, mode):
			"""Removes all connections into or out of a given compartment
			
			Args:
			    i (int): Compartment index
			    mode (str): Specify whether inward or outward connections are removed
			
			"""
			if i < 0 or i > len(G) - 1:
				return

			if mode == "out":
				if i + 1 < len(G):
					G[i][i+1] = 0
			elif mode == "in":
				if i - 1 >= 0:
					G[i][i-1] = 0

		for x in cs:
			remove(x - 1, "out")
			remove(x, "in")

		for v in self.tree.vs:
			el = v.all_edges()

			first = side(el[0], v.index)
			G[first][first] = -v.degree()

			if len(el) > 1:
				for e in el[1:]:
					connect(first, side(e, v.index))

		self.G = G

	def solve(self):
		"""Solves the system for the defined oligo object by solving coupled IVPs in each compartment
		"""

		if self.G is None:
			self.ddx2()


		if len(self.params['c0']) == 0:
			self.params['c0'] = np.full(self.num_compart, 0.15)

		if len(self.params['h0']) == 0:	
			self.params['h0'] = np.full(self.num_compart, 0.74)

		if len(self.params['er0']) == 0:
			self.params['er0'] = np.full(self.num_compart, 9)

		if len(self.params['jint0']) == 0:
			self.params['jint0'] = np.full(self.num_compart, 0)

		sys = self.system()
		self.sol = solve_ivp(fun = sys.solve, t_span = (0, self.tspan[1]), y0 = [*self.params['c0'], *self.params['h0'], *self.params['er0'], *self.params['jint0']], args=([self.G]), dense_output=True)


	def system(self):
		"""
		Closure describing all model equations
		Using a closure provides a computational speed-up, makes model equations easier to read and allows for flux access outside the system
		Defines coupled PDE as a system of equations as return d[EQUATIONS]
		Must contain method 'solve' that returns d[EQUATIONS]
		
		Returns:
		    function with equations as arguments: Function describing system with equations as arguments
		"""
		c0, h0, er0, c1, v1, v2, v3, k3, d1, d2, d3, d5, a2, kout, vryr, kr,vncx, cao, nao, nai, kna, kca, ksat, eta, V, F, R, T,V_cell, ip3, vsocc, ksocc, cdiff,xi,gamma,pin,hc3,inmax,intau,jint0, baseline = list(self.params.values())
		dx2 = self.dx**2

		def solve(t, z, g):
			"""returns d[EQUATIONS]
			
			Args:
			    t (float): Time
			    z (2D list of float): State vector [EQUATIONS] at time t, shape = (number of time-dep variables, number of compartments)
			    g (2D array of float): Second derivative matrix
			
			Returns:
			    2D list of float: Solution vector d[EQUATIONS] at time t

			"""

			c, h, er, jint = np.array_split(z,4)
			jer = jip3(c, er ,h) - jserca(c) + jerleak(c,er) + jryr(c, er)

			dC = solve_c(c, er, g, jer, jint)
			dH = solve_h(c, h)
			dER = solve_er(er, jer ,g)
			dJint = jintdep(c, jint)

			return [*dC, *dH, *dER, *dJint]

		def solve_c(c, er, g, jer, jint):
			"""solve for dC according to PDE
			
			Args:
			    c (list of float): Cytosolic calcium in uM
			    er (list of float): ER calcium in uM
			    g (2d array of float): Second derivative matrix
			    jer (list of float): Flux through ER membrane
			
			Returns:
			    list of float: dC in x
			"""
			J = jer + pin - kout*c + jint
			dC = cdiff*(g @ c)/dx2 + J		
			return dC

		def solve_h(c, h):
			"""solve for dh according to PDE
			
			Args:
			    c (list of float): Cytosolic calcium in uM
			    h (list of float): open-fraaction variable h in Li-Rinzel model
			
			Returns:
			    TYPE: dh in x
			"""
			ah = a2*d2*(ip3+d1)/(ip3+d3)
			bh = a2*c
			
			return ah*(1 - h) - bh*h + xi*np.random.randn(len(c))
			
		def solve_er(er, jer, g):
			"""solve for dER according to PDE
			
			Args:
			    jer (list of float): Flux through ER membrane
			
			Returns:
			    list of float: dER in x
			"""
			return -gamma*jer + cdiff*(g @ er)/dx2

		def solve_mp(mp, d):
			"""Unused, flux through stretch channels
			"""

			mpinf = 1/(1+np.exp(-(d-3.5)/0.25))
			taup = 1
			return(mpinf - mp)/taup

		def jip3(c, er, h):
			"""Flux through IP3 receptors
			
			Args:
			    c (list of float): Cytosolic calcium in uM
			    er (list of float): ER calcium in uM
			    h (list of float): open-fraaction variable h in Li-Rinzel model
			
			Returns:
			    list of float: flux through IP3 receptors
			"""
			m = ip3/(ip3 + d1)
			n = c/(c+d5)

			return m**3*n**3*h**3*c1*v1*(er - c)

		def jncx(c):
			"""Flux through NCX channels
			
			Args:
			    c (list of float): Cytosolic calcium in uM
			
			Returns:
			    list of float: flux through NCX receptors
			"""
			incx = vncx*(np.exp(eta*V*F/(R*T))*nai**3*cao - np.exp((1-eta)*V*F/(R*T))*nao**3*c)/((kna**3+nao**3)*(kca+cao)*(1+ksat*np.exp((1-eta)*V*F/(R*T))))
			return -incx/(2000*F*V_cell)

		def jsocc(er):
			"""Flux though SOCC channels
			
			Args:
			    er (list of float): ER calcium in uM
			
			Returns:
			    list of float: flux through SOCC receptors
			"""
			return vsocc*ksocc**4/(ksocc**4 + er**4)

		def jryr(c, er):
			"""Flux though ryanodine receptors
			
			Args:
			    c (list of float): Cytosolic calcium in uM
			    er (list of float): ER calcium in uM
			
			Returns:
			    list of float: flux though ryanodine receptors
			"""
			return (vryr*c**3/(c**3 + kr**3))*(er - c)

		def jin(c, er):
			"""Flux though cell membrane
			
			Args:
			    c (list of float): Cytosolic calcium in uM
			    er (list of float): ER calcium in uM
			
			Returns:
			    list of float: flux though cell membrane
			"""
			return jsocc(er) - jncx(c) - jout(c)

		def jserca(c):
			"""Flux through SERCA pump
			
			Args:
			    c (list of float): Cytosolic calcium in uM
			
			Returns:
			    list of float: flux though SERCA pump
			"""
			return v3*c**hc3/(k3**hc3 + c**hc3)

		def jerleak(c, er):
			"""Leak flux through ER membrane
			
			Args:
			    c (list of float): Cytosolic calcium in uM
			    er (list of float): ER calcium in uM
			
			Returns:
			    list of float: leak flux through ER membrane
			"""
			return c1*v2*(er - c)    	

		def jout(c):
			"""Exocytosis flux through cell membrane
			
			Args:
			    c (list of float): Cytosolic calcium in uM
			
			Returns:
			    list of float: exocytosis flux through cell membrane
			"""
			return kout*c

		def jintdep(c, jint):
			"""Time dependent inward flux lagging cytosolic calcium
			
			Args:
			    c (list of float): Cytosolic calcium in uM
			    jint (list of float): Fraction of t-dep channels open

			
			Returns:
			    list of float: time dependent inward flux lagging cytosolic calcium
			"""
			vinf = inmax*(c-baseline)
			return (vinf - jint)/intau

		def jpiezo(c, mp):
			"""Unused, flux through stretch channels
			"""
			return gp*mp*(V - Ep)

		out = lambda : None
		out.solve = solve
		out.solve_c = solve_c
		out.solve_er = solve_er
		out.solve_h = solve_h
		out.solve_mp = solve_mp
		out.jip3 = jip3
		out.jncx = jncx
		out.jsocc = jsocc
		out.jryr = jryr
		out.jin = jin
		out.jserca = jserca
		out.jerleak = jerleak
		out.jout = jout
		out.jpiezo = jpiezo

		return out

	def plot(self, i):
		"""Plots C,, jin h and ER trace over time in ith compartment
		
		Args:
		    i (int): Compartment to plot time series
		"""
		t = np.linspace(self.tspan[0], self.tspan[1], (self.tspan[1]-self.tspan[0])*5)
		z = self.sol.sol(t)

		fig, axs = plt.subplots(4)
		fig.suptitle("x = " + str(i))

		axs[0].plot(t, z[i])
		axs[0].set(ylabel = r'Cytosol Calcium ($uM$)')

		axs[1].plot(t, z[i + 3*self.num_compart]*1e6, 'tab:purple')
		axs[1].set(ylabel = r'Jin ($\frac{M}{um^2s}$)')

		axs[2].plot(t, z[i + self.num_compart], 'tab:orange')
		axs[2].set(ylabel = r'h-open fraction')

		axs[3].plot(t, z[i + 2*self.num_compart], 'tab:green')
		axs[3].set(ylabel = r'ER Calcium ($uM$)')
		axs[3].set(xlabel = r'Time ($s$)')

		plt.show()

	def save_plot(self, i):
		"""Does a simulation and saves plot of C to a folder 'parametersweep/', useful for parameter sweeps
		
		Args:
		    i (int): Compartment to plot time series
		"""

		t = np.linspace(self.tspan[0], self.tspan[1], (self.tspan[1]-self.tspan[0])*5)
		z = self.sol.sol(t)

		plt.plot(t, z[i])
		plt.ylabel('Cytosol Calcium (uM)')
		plt.xlabel('Time (s)')

		plt.savefig('parametersweep/' + str(o.params['d5']) + " " +  str(o.params['v1']) + " " +str(o.params['vryr']) + " " +str(o.params['kr']) + " " +str(o.params['a2']) + " " +str(o.params['v3']) + " " +str(o.params['k3']) + " " +str(o.params['kout']) +" " + str(o.params['ip3']) + " " +str(o.params['xi']) + '.png')
		plt.close()

	def plot_window(self, i, leng = 1):
		"""Plots the average calcium along a window of some given length, for comparison to experimental data (ROI)
		
		Args:
		    i (int): Starting compartment
		    leng (int, optional): Length of window in um
		"""

		n = int(leng/self.dx)
		t = np.linspace(0, self.tend, self.tend*5)
		z = self.sol.sol(t)

		fig, axs = plt.subplots(3)
		fig.suptitle("window x = " + str(i)+ " to " + str(i+n))

		window = [np.mean([z[i+j] for j in range(n)], axis = 0),
		np.mean([z[i+j + self.num_compart] for j in range(n)], axis = 0),
		np.mean([z[i+j + 2*self.num_compart] for j in range(n)], axis = 0),
		np.mean([z[i+j + 3*self.num_compart] for j in range(n)], axis = 0)]

		axs[0].plot(t, window[0])
		axs[0].set(ylabel = r'Cytosol Calcium ($uM$)')

		axs[1].plot(t, window[3], 'tab:purple')
		axs[1].set(ylabel = r'Jin ($\frac{M}{um^2s}$)')

		axs[2].plot(t, window[1], 'tab:orange')
		axs[2].set(ylabel = r'h-open fraction')

		axs[3].plot(t, window[2], 'tab:green')
		axs[3].set(ylabel = r'ER Calcium ($uM$)')
		axs[3].set(xlabel = r'Time ($s$)')
		plt.show()

	def view(self, vmax = -1):
		"""Plots meshgrid of calcium in x and t to show travelling waves along branches
		Breaks if only 1 branch since axs is an array
		
		Args:
		    vmax (float, optional): Maximal 
		"""

		fig, axs = plt.subplots(len(self.tree.es))
		fig.tight_layout()
		t = np.linspace(self.tspan[0], self.tspan[1], (self.tspan[1]-self.tspan[0])*5)
		z = self.sol.sol(t)
	

		if vmax == -1:
			vmax = np.max(z[:int(len(z)/3)])

		c = 0
		for i in range(len(self.tree.es)):
			start = c
			c += int(self.tree.es[i]["leng"]/self.dx)
			end = c
			axs[i].set_title("Branch " + str(self.tree.es[i].tuple[0]) + " to " + str(self.tree.es[i].tuple[1]))

			x = np.arange(start, end, 1)*self.dx
			xx, tt = np.meshgrid(x, t)
			im = axs[i].pcolormesh(xx, tt, z[start:end].T, vmin = np.min(z[:int(len(z)/3)]), vmax = vmax, cmap = plt.get_cmap('gist_earth'), norm=matplotlib.colors.PowerNorm(gamma= 1))
			axs[i].set(ylabel = 't (s)')
			
		axs[-1].set(xlabel = 'x (um)')

		fig.subplots_adjust(right=0.8)
		cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
		fig.colorbar(im, cax=cbar_ax)
		plt.show()

	def phase_plane(self):
		"""Plot 3D phase plane of reduced system
		"""

		def plot_implicit(fns, xbbox, ybbox, zbbox):
			"""Summary
			
			Args:
			    fns (list of functions): Functions for surfaces
			    xbbox (tuple of float): x bounding box
			    ybbox (tuple of float): y bounding box
			    zbbox (tuple of float): z bounding box
			"""

			xmin, xmax = xbbox
			ymin, ymax = ybbox
			zmin, zmax = zbbox
			fig = plt.figure()
			ax = plt.axes(projection='3d')
			c = ['tab:blue', 'tab:green', 'tab:orange']
			xl = np.linspace(xmin, xmax, 10)
			yl = np.linspace(ymin, ymax, 10)
			zl = np.linspace(zmin, zmax, 10)
			X, Y, Z = np.meshgrid(xl, yl, zl)

			for i in range(len(fns)):
				F = fns[i](X, Y, Z)

				verts, faces, normals, values = measure.marching_cubes(F, 0, spacing=[np.diff(xl)[0], np.diff(yl)[0], np.diff(zl)[0]])
				ax.plot_trisurf(verts[:, 0]*(np.diff(yl)[0]/np.diff(xl)[0]) + ymin, verts[:, 1]*(np.diff(xl)[0]/np.diff(yl)[0])  + xmin, faces, verts[:, 2] + zmin, lw=0)

			ax.set_xlabel('er (uM)')
			ax.set_ylabel('c (uM)')
			ax.set_zlabel('h')

			plt.show()

		sys = self.system()

		def c1d(c, er, h):
			"""Reduced C system
			
			Returns:
			    List of float: dC in 1 time step dt
			"""

			jin = sys.jin(c, er)
			jer = sys.jip3(c, er ,h) - sys.jserca(c) + sys.jerleak(c,er) + sys.jryr(c, er)
			dC = jer + jin		
			return dC

		def h1d(c, er, h):
			"""Reduced h system

			Returns:
			    List of float: dh in 1 time step dt
			"""

			ah = self.params['a2']*self.params['d2']*(self.params['ip3']+self.params['d1'])/(self.params['ip3']+self.params['d3'])
			bh = self.params['a2']*c
		
			return ah*(1 - h) - bh*h

		def er1d(c, er, h):
			"""Reduced ER system
			
			Returns:
			    List of float: dER in 1 time step dt
			"""

			return -self.params['gamma']*(sys.jip3(c, er ,h) - sys.jserca(c) + sys.jerleak(c,er)+ sys.jryr(c, er))

		plot_implicit(fns = [c1d, er1d, h1d], xbbox = (0,1), ybbox = (6,10), zbbox = (0.3,1))

	def plot_ddt(self, i):
		"""Plot dC/dt and dER/dt vs. t to show which variable is fast and which is slow
		
		Args:
		    i (TYPE): Compartment (in spatial direction) to plot time series
		"""

		t = np.linspace(self.tspan[0], self.tspan[1], (self.tspan[1]-self.tspan[0])*5)
		z = self.sol.sol(t)

		plt.title("d/dt at x = " + str(i))

		dc = np.gradient(z[i], t)
		plt.plot(t, dc, label = 'Cytosol Calcium (uM)')

		der = np.gradient(z[i + 2*self.num_compart], t)
		plt.plot(t, der, 'tab:green', label = 'ER Calcium (uM)')

		plt.xlabel('Time (s)')

		plt.legend()
		plt.show()


	def isi_std(self, n):
		"""Runs n simulations, computes the ISI-STD plot and saves to file
		
		Args:
		    n (int): Number of iterations
		"""

		T = []
		stdev = []
		h = []

		for i in range(n):
			print(i)
			self.tspan = [0, 900]
			self.solve()
			t = np.linspace(self.tspan[0], self.tspan[1], (self.tspan[1]-self.tspan[0])*5)
			z = self.sol.sol(t)
			data = z[30]

			pks, _ = find_peaks(data/(max(data) - min(data)), height = 0.1, threshold = 0.04)

			ISI = 2*np.diff(pks)
			if np.std(ISI, ddof = 1) != 0:
				T.append(np.mean(ISI))
				stdev.append(np.std(ISI, ddof = 1))
				h.append(data[pks])

		plt.scatter(T, stdev)
		# plt.plot(T,T)
		plt.xlabel('ISI Mean (s)')
		plt.ylabel('ISI Standard Deviation (s)')
		plt.show()

		np.save("ISI-STD/t1", T)
		np.save("ISI-STD/std1", stdev)
		np.save("ISI-STD/h1", h)

	def bifurcation(self, mode = 'ode'):
		"""Code to generate bifuraction diagrams using PYDSTool, not working framework to be added to
		
		Args:
		    mode (str, optional): ode for the ode model and pde for the travelling wave pde model
		"""

		if mode == 'ode':
			DSargs = dst.args(name = 'oligodendrocyte calcium model')
			DSargs.pars = self.params
			DSargs.fnspecs = {}
			DSargs.varspecs = {}
			DSargs.ics = {}

			DSargs.tdomain = [0, self.tend]
			ode = dst.Generator.Vode_ODEsystem(DSargs)

			ode.set(pars = {})
			ode.set(ics = {})

			PC = dst.args(name = 'EQ1', type = 'EP-C')
			PCargs.freepars = []
			PCargs.MaNumPoints = 450
			PCargs.MaxStepSize  = 2
			PCargs.MinStepSize  = 1e-5
			PCargs.StepSize     = 2e-2
			PCargs.LocBifPoints = 'LP'                     # detect limit points / saddle-node bifurcations
			PCargs.SaveEigen    = True                     # to tell unstable from stable branches

			PC.newCurve(PCargs)
			PC['EQ1'].forward()
			PC.display([], stability = True, figure = 3)
			
			PC['EQ1'].info()
			print(PC['EQ1'].getSpecialPoint(''))

		if mode == 'pde':
			pass

	def fluxes(self):
		"""Plot some relavant non-time dependent fluxes (JRYR, JIP3, JIN) in 3D, against C, ER axes
		"""

		fig = plt.figure()
		ax = plt.gca(projection='3d')

		er = np.linspace(5,10)
		h = 0.7
		c = np.linspace(0,2)

		sys = self.system()
		C, ER = np.meshgrid(c,er)
		r = ax.plot_surface(C, ER, sys.jryr(C, ER), color = 'red')
		ip = ax.plot_surface(C, ER, sys.jip3(C, ER, h), color = 'blue')
		i = ax.plot_surface(C, ER, sys.jin(C, ER), color = 'purple')

		ax.set_xlabel('c')
		ax.set_ylabel('er')
		ax.set_zlabel('j')
		plt.show()