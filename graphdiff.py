import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import igraph as g
from scipy.integrate import ode
from scipy.integrate import solve_ivp
import sys

class graphDiff:
	def __init__(self):
		self.tree = g.Graph(directed = True)
		self.dx = 1e-1
		self.tree.add_vertices(1)
		self.params = {}
		self.num_compart = 0
		self.sol = None
		self.tend = 20

	def add_branch(self, start, end, leng):
		if start >= self.tree.vcount() or end >= self.tree.vcount():
			self.tree.add_vertices(1)

		self.tree.add_edges([(start, end)])
		self.tree.es[len(self.tree.es) - 1]["leng"] = leng
		self.num_compart += int(leng/self.dx)
	
	def view_tree(self):
		layout = self.tree.layout("kk")
		g.plot(self.tree, layout = layout, vertex_label = [v.index for v in self.tree.vs])


	def discretize(self):
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
			if vid == e.tuple[0]:
				return cs[e.index]
			elif vid == e.tuple[1]:
				return cs[e.index] + int(self.tree.es["leng"][e.index]/self.dx) - 1

		def connect(a, b):
			G[a][b] = 1
			G[b][a] = 1

		def remove(i, mode):
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
		
		return G

	def solve(self):
		G = self.discretize()
		C0 = np.full(self.num_compart, self.params['c0'])
		H0 = np.full(self.num_compart, self.params['h0'])
		ER0 = np.full(self.num_compart, self.params['er0'])
		
		C0[45] = 3
		self.sol = solve_ivp(fun = self.system, t_span = (0, self.tend), y0 = [*C0, *H0, *ER0], args=([G]), dense_output=True)


	def system(self, t, z, g):
		c, h, er = np.array_split(z,3)
		jer = self.jip3(c, er ,h) + self.jryr(c, er) - self.jserca(c) + self.jerleak(c,er)

		dC = self.solve_c(c, er, g, jer)
		dH = self.solve_h(c, h)
		dER = self.solve_er(jer)

		# print(dC)

		return [*dC, *dH, *dER]

	def solve_c(self, c, er, g, jer):
		dC = 0.05*(g @ c)/self.dx**2
		return dC

	def solve_h(self, c, h):
		ah = self.params['a2']*self.params['d2']*(self.params['ip3']+self.params['d1'])/(self.params['ip3']+self.params['d3'])
		bh = self.params['a2']*c
		# return ah*(1 - h) - bh*h

		return np.zeros(self.num_compart)
		
	def solve_er(self, jer):
		# return -jer
		return np.zeros(self.num_compart)

	def iplot(self, i):
		t = np.linspace(0, self.tend, self.tend*5)
		z = self.sol.sol(t)

		fig, axs = plt.subplots(3)
		fig.suptitle("x = " + str(i))

		axs[0].plot(t, z[i])
		axs[0].set(ylabel = 'Cytosol Calcium (uM)')

		axs[1].plot(t, z[i + self.num_compart], 'tab:orange')
		axs[1].set(ylabel = 'h-open fraction')

		axs[2].plot(t, z[i + 2*self.num_compart], 'tab:green')
		axs[2].set(ylabel = 'ER Calcium (uM)')
		axs[2].set(xlabel = 'Time (s)')

		for ax in axs:
			ax.grid()

		plt.show()

	def view(self):
		fig, axs = plt.subplots(len(self.tree.es))
		fig.tight_layout()
		t = np.linspace(0, self.tend, self.tend*5)
		z = self.sol.sol(t)
		t = np.linspace(0, self.tend, self.tend*5)

		print(z.shape)
		print(z[0:30].shape)
		
		c = 0
		for i in range(len(self.tree.es)):
			start = c
			c += int(self.tree.es[i]["leng"]/self.dx)
			end = c
			axs[i].set_title("Branch " + str(self.tree.es[i].tuple[0]) + " to " + str(self.tree.es[i].tuple[1]))

			x = np.arange(start, end, 1)*self.dx
			xx, tt = np.meshgrid(x, t)
			im = axs[i].pcolormesh(xx, tt, z[start:end].T, vmin = 0.15, vmax = 1, cmap = plt.get_cmap('tab20c'), norm=matplotlib.colors.PowerNorm(gamma=0.3))
			axs[i].set(ylabel = 't (s)')
			
		axs[-1].set(xlabel = 'x (um)')

		fig.subplots_adjust(right=0.8)
		cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
		fig.colorbar(im, cax=cbar_ax)
		plt.show()

o = oligo()
o.add_branch(0,1,3)
o.add_branch(1,2,3)
o.add_branch(0,3,3)
o.add_branch(3,1,3)


o.view_tree()
o.solve()
o.view()