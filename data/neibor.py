import numpy as np
import scipy.sparse as sp

####################################################
# This tool is to collect neighbors, and reform them
# as numpy.array style for futher usage.
####################################################

pa = np.genfromtxt("./dblp/pa.txt")
a_n = {}
for i in pa:
  if i[1] not in a_n:
    a_n[int(i[1])]=[]
    a_n[int(i[1])].append(int(i[0]))
  else:
    a_n[int(i[1])].append(int(i[0]))
a_n = list(a_n.values())
a_n = np.array([np.array(i) for i in a_n])
np.save("nei_p.npy", a_n)

# give some basic statistics about neighbors
l = [len(i) for i in a_n]
print(max(l),min(l),np.mean(l))
