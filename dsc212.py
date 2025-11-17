import networkx as nx
import matplotlib.pyplot as mp
import numpy as np

colour_map = ['red','blue','green','orange','purple','darkgreen','lightcoral'] # Arbitrarily sized colour template from which we shall colour our communities
q = 0                                                                          # Global terative variable defined for ease of graph rendering
G = nx.karate_club_graph()                                                     # Karate club graph initialized
count = 0                                                                      # Global iterative variable defined for ease of graph splitting
colour = [0 for i in range(34)]                                                # Empty list which shall be filled with the convention such that the colour at the i-th index represents the colour of the i-th node, which by extension represents the colour of its community
F = G                                                                          # Copy of G to allow visualisation after each split
Q = []                                                                         # List containing modularity of the graph after each iteration
com = dict.fromkeys(list(nx.nodes(G)))                                         # List defined to represent the community the i-th node belongs to through the use of integers at the i-th index in the list
it_data = dict.fromkeys(list(nx.nodes(G)))                                     # Dictionary containing data about which node belonged to which community for each iteration
for i in nx.nodes(G):                                                          # Initialization of dictionary and list
    com[i] = -1
    it_data[i] = [(0,"null")]

def modularity(g):                                                                 # Function to compute modularity
    k = np.array(g.degree)
    A = nx.adjacency_matrix(g)
    k = k[:,1]
    k = k[:, np.newaxis]
    k_t = k.T
    K = k @ k_t
    a = np.sum(k)
    K = K/a
    B = (A - K)
    l, u = np.linalg.eig(B)
    s = u[:,list(l).index(np.max(l))]
    s = s[:, np.newaxis]
    s_T = s.T
    Q.append((1/(2*a))*float((s_T @ B @ s)[0,0].real))

def identify(g):                                                               # Function to partition a given graph into sets of two points based on the community they belong to through spectral decomposition
    global count
    count = 0
    k = np.array(g.degree)
    A = nx.adjacency_matrix(g)
    k = k[:,1]
    k = k[:, np.newaxis]
    k_t = k.T
    K = k @ k_t
    a = np.sum(k)
    K = K/a
    B = (A - K)
    l, u = np.linalg.eig(B)
    ind = 0
    s = u[:,list(l).index(np.max(l))]
    for i in range(len(s)):
        if s[i].imag != 0:
            s[i] = float(0)
        else:
            s[i] = float(s[i].real)
    for i in list(g.nodes()):
        if s[ind]>0:
            com[i]=count
        else:
            com[i]=count+1
        ind = ind + 1
    count = count + 2

def partition(f):                                                              # Function the split given graph into two subgraphs where each is a community
    l1 = []
    l2 = []
    for i in nx.nodes(f):
        if com[i]%2 == 0:
            l1.append(i)
        else:
            l2.append(i)
    f_1 = nx.subgraph(f, l1)
    f_2 = nx.subgraph(f, l2)
    return f_1,f_2

def update(g, color):                                                          # Function to update the colour a set of nodes are supposed to be
    for i in nx.nodes(g):
        it_data[i].append((it_data[i][-1][0]+1,color))

def terminal(g):                                                               # Function to determine if anymore communities can be formed in the given graph
    if(nx.is_empty(g)==True):
        return 0
    t = []
    for i in nx.nodes(g):
        t.append(com[i])
    p = set(t)
    return len(p)

def edge_cut(a, b):                                                            # Function to isolate communities in the graph F
    for i in nx.nodes(a):
        for j in nx.nodes(b):
            if (i,j) in nx.edges(F):
                F.remove_edge(i, j)

mp.figure(figsize=(20, 16))                                                    # Predefining the plotting figure size

def REC(g):                                                                    # Recursive function to identify communities
    if nx.is_empty(g)==True:
        return 0
    else:
        global q
        global colour
        identify(g)
        a, b = partition(g)
        identify(a)
        identify(b)
        update(a,"blue")
        update(b,"red")
        for i in nx.nodes(G):
            if i in nx.nodes(a):
                colour[i] = colour_map[q]
            if i in nx.nodes(b):
                colour[i] = colour_map[q+1]

        subax1 = mp.subplot(2, 2, q+1)
        nx.draw_spring(F, with_labels = True, node_color = colour)

        edge_cut(a, b)
        modularity(F)
        
        subax2 = mp.subplot(2,2,q+2)
        nx.draw_spring(F, with_labels = True, node_color = colour)

        q += 2
        
        if nx.is_empty(a)==False and terminal(a)!=1:
            REC(a)
        if nx.is_empty(b)==False and terminal(b)!=1:
            REC(b)

modularity(G)                                                              # Computing modularity of G first so that the 0-th index of Q corresponds to the original graph
REC(G)

for i in it_data.keys():
    it_data[i].pop(0)

depth = 0

for i in it_data.keys():
    x = len(it_data[i])
    if x > depth:
        depth = x

colour_dict = {key: [] for key in range(1,depth+1)}
deg_c = dict.fromkeys(range(1,depth+1))
bet_c = dict.fromkeys(range(1,depth+1))
clo_c = dict.fromkeys(range(1,depth+1))
clus_co = dict.fromkeys(range(1,depth+1))

def construct(D, d):                                                    
    temp = 0
    for i in range(1,d+1):
        red = []
        blue = []
        for j in D.keys():
            for k in D[j]:
                if k[0] == i:
                    if k[1] == "red":
                        red.append(j)
                        colour_dict[i].append(colour_map[temp])
                    else:
                        blue.append(j)
                        colour_dict[i].append(colour_map[temp+1])
        a = nx.subgraph(G, red)
        b = nx.subgraph(G, blue)
        y = nx.degree_centrality(a) | nx.degree_centrality(b)
        deg_c[i] = dict(sorted(y.items()))
        y = nx.betweenness_centrality(a) | nx.betweenness_centrality(b)
        bet_c[i] = dict(sorted(y.items()))
        y = nx.closeness_centrality(a) | nx.closeness_centrality(b)
        clo_c[i] = dict(sorted(y.items()))
        y = nx.clustering(a) | nx.clustering(b)
        clus_co[i] = dict(sorted(y.items()))
        temp += 2

construct(it_data, depth)

for i in nx.nodes(G):                                                           # Logic to allow the metrics for nodes in terminal communities to be carried on in further iterations. For example if node 7 was split in iteration 1 and then labeled part of a "terminal community" then in further iterations it should possess the same value for the various metrics and thus it's terminal value is carried forth into further iterations
    if i not in deg_c[2]:
        deg_c[2][i] = deg_c[1][i]
    if i not in bet_c[2]:
        bet_c[2][i] = bet_c[1][i]
    if i not in clo_c[2]:
        clo_c[2][i] = clo_c[1][i]
    if i not in clus_co[2]:
        clus_co[2][i] = clus_co[1][i]

mp.figure(figsize=(16,10))

subax1 = mp.subplot(1,2,1)
nx.draw_kamada_kawai(G, with_labels = True, node_color = colour)

for i in nx.edges(G):
    if com[i[0]]!=com[i[1]]:
        G.remove_edge(i[0],i[1])

subax2 = mp.subplot(1,2,2)
nx.draw_spring(G, with_labels=True, node_color = colour)

mp.figure(figsize=(16,10))

x = [1, 2]
y_1 = [deg_c[1][k] for k in deg_c[1].keys()]
y_2 = [deg_c[2][k] for k in deg_c[2].keys()]

for i in nx.nodes(G):
    y = [y_1[i], y_2[i]]
    mp.plot(x, y, label = f"Node {i}")

mp.xlabel('Iterations')
mp.ylabel('Degree Centrality')
mp.legend(loc = 'right')
mp.show()

mp.figure(figsize=(16,10))

x = [1, 2]
y_1 = [bet_c[1][k] for k in bet_c[1].keys()]
y_2 = [bet_c[2][k] for k in bet_c[2].keys()]

for i in nx.nodes(G):
    y = [y_1[i], y_2[i]]
    mp.plot(x, y, label = f"Node {i}")

mp.xlabel('Iterations')
mp.ylabel('Betweenness Centrality')
mp.legend(loc = 'right')
mp.show()

mp.figure(figsize=(16,10))

x = [1, 2]
y_1 = [clo_c[1][k] for k in clo_c[1].keys()]
y_2 = [clo_c[2][k] for k in clo_c[2].keys()]

for i in nx.nodes(G):
    y = [y_1[i], y_2[i]]
    mp.plot(x, y, label = f"Node {i}")

mp.xlabel('Iterations')
mp.ylabel('Closeness Centrality')
mp.legend(loc = 'right')
mp.show()

mp.figure(figsize=(16,10))

x = [1, 2]
y_1 = [clus_co[1][k] for k in clus_co[1].keys()]
y_2 = [clus_co[2][k] for k in clus_co[2].keys()]

for i in nx.nodes(G):
    y = [y_1[i], y_2[i]]
    mp.plot(x, y, label = f"Node {i}")

mp.xlabel('Iterations')
mp.ylabel('Clustering Coefficient')
mp.legend(loc = 'right')
mp.show()


mp.figure(figsize=(16,10))

x = [0, 1, 2]
mp.plot(x, Q)
mp.xlabel('Iterations')
mp.ylabel('Modularity')
mp.show()
