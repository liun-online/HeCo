import numpy as np
import scipy.sparse as sp

####################################################
# This tool is to generate meta-path based adjacency
# matrix given original links.
####################################################

pa = np.genfromtxt("./dblp/pa.txt")
pc = np.genfromtxt("./dblp/pc.txt")
pt = np.genfromtxt("./dblp/pt.txt")

A = 4057
P = 14328
C = 20
T = 7723

pa_ = sp.coo_matrix((np.ones(pa.shape[0]),(pa[:,0], pa[:, 1])),shape=(P,A)).toarray()
pc_ = sp.coo_matrix((np.ones(pc.shape[0]),(pc[:,0], pc[:, 1])),shape=(P,C)).toarray()
pt_ = sp.coo_matrix((np.ones(pt.shape[0]),(pt[:,0], pt[:, 1])),shape=(P,T)).toarray()

apa = np.matmul(pa_.T, pa_) > 0
apa = sp.coo_matrix(apa)
sp.save_npz("./dblp/apa.npz", apa)

apc = np.matmul(pa_.T, pc_) > 0
apcpa = np.matmul(apc, apc.T) > 0
apcpa = sp.coo_matrix(apcpa)
sp.save_npz("./dblp/apcpa.npz", apcpa)

apt = np.matmul(pa_.T, pt_) > 0
aptpa = np.matmul(apt, apt.T) > 0
aptpa = sp.coo_matrix(aptpa)
sp.save_npz("./dblp/aptpa.npz", aptpa)
