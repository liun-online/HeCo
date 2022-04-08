import numpy as np
import scipy.sparse as sp

####################################################
# This tool is to collect neighbors, and reform them
# as numpy.array style for futher usage.
####################################################

# This is for DBLP
pa = np.genfromtxt("./dblp/pa.txt")
a_n = {}
for i in pa:
  if i[1] not in a_n:
    a_n[int(i[1])]=[]
    a_n[int(i[1])].append(int(i[0]))
  else:
    a_n[int(i[1])].append(int(i[0]))
    
keys =  sorted(a_n.keys())
a_n = [a_n[i] for i in keys]
a_n = np.array([np.array(i) for i in a_n])
np.save("nei_p.npy", a_n)

# give some basic statistics about neighbors
l = [len(i) for i in a_n]
print(max(l),min(l),np.mean(l))




# This is for ACM, Freebase, AMiner
pa = np.genfromtxt("./aminer/pa.txt")
p_n = {}
for i in pa:
  if i[0] not in p_n:
    p_n[int(i[0])]=[]
    p_n[int(i[0])].append(int(i[1]))
  else:
    p_n[int(i[0])].append(int(i[1]))
    
keys =  sorted(p_n.keys())
p_n = [p_n[i] for i in keys]
p_n = np.array([np.array(i) for i in p_n])
np.save("nei_a.npy", p_n)

# give some basic statistics about neighbors
l = [len(i) for i in p_n]
print(max(l),min(l),np.mean(l))
