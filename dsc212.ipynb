# Importing the libraries we need
import networkx as nx
import matplotlib.pyplot as mp
import numpy as np

# 1. Initial Setup

# A list of colors for our plots.
plot_colors = ['red', 'blue', 'green', 'orange', 'purple', 'darkgreen', 'lightcoral', 'yellow', 'cyan']

# This will help us position our plots in a grid (e.g., subplot 1, 2, 3...)
plot_position = 0

# Loading the graph
G = nx.karate_club_graph()

# We'll make a copy of the graph to draw on. We'll cut edges from this copy to show the communities separating.
drawing_graph = G.copy()

# A list to save the modularity score at each step
modularity_scores = []

# This dictionary will hold which community each node belongs to.
node_community = {node: -1 for node in G.nodes()}

# This will track which community a node was in at each iteration.
# It's a dictionary where each node gets a list of tuples: (iteration_number, color_label)
node_history = {node: [(0, "null")] for node in G.nodes()}

# A list to hold the color name for each node, e.g., node_colors[0] = 'red'
# This is just for plotting.
node_colors = [0] * G.number_of_nodes()

# A counter to give new communities a unique ID (0, 1, 2, 3...)
community_counter = 0

# 2. Helper functions

def get_modularity_score(g):
    """
    This calculates the modularity of a given graph (or subgraph).
    It follows the math from the handout:
    1. Build the modularity matrix B = A - (k*k^T / 2m)
    2. Find the leading eigenvector 's' of B
    3. Calculate the score: Q = (1 / 2m) * s^T * B * s
    """
    # Get the node degrees, e.g., [(0, 16), (1, 9), ...]
    degrees_list = np.array(g.degree)
    
    # Get the adjacency matrix (as a 0/1 numpy array)
    A = nx.adjacency_matrix(g).toarray()
    
    # Get just the degree numbers (k)
    k = degrees_list[:, 1]
    k_col = k[:, np.newaxis]  # Make it a column vector (n x 1)
    
    # Calculate the k*k^T matrix (outer product)
    k_outer = k_col @ k_col.T
    
    # Get 2m (sum of all degrees)
    two_m = np.sum(k)
    
    # Calculate the "expected edges" matrix P = k*k^T / 2m
    P = k_outer / two_m
    
    # The Modularity Matrix! B = A - P
    B = A - P
    
    # Get eigenvalues and eigenvectors. .eigh is good for symmetric matrices.
    eigenvals, eigenvecs = np.linalg.eigh(B)
    
    # Get the leading eigenvector (the one for the biggest eigenvalue)
    s = eigenvecs[:, np.argmax(eigenvals)]
    s_col = s[:, np.newaxis] # Make it a column vector
    
    # Calculate the final modularity score
    # We use (1 / two_m) which is the same as (1 / (2*a)) in your code
    s_T = s_col.T
    score = (1 / two_m) * float((s_T @ B @ s_col)[0, 0].real)
    
    # Save this score to our global list
    modularity_scores.append(score)

def find_split(g):
    """
    This function finds the best way to split a subgraph 'g' into two.
    It calculates the modularity matrix B and its leading eigenvector 's'.
    It then updates the global 'node_community' map with new community IDs.
    
    This MODIFIES the global 'node_community' and 'community_counter'.
    """
    global community_counter
    
    # Get the list of nodes in this subgraph
    nodes_in_g = list(g.nodes())
    if not nodes_in_g: # If the subgraph is empty, just stop.
        return

    # Get degrees and adjacency matrix for *this subgraph*
    degrees_list = np.array(g.degree)
    A = nx.adjacency_matrix(g).toarray()
    
    k = degrees_list[:, 1]
    k_col = k[:, np.newaxis]
    
    k_outer = k_col @ k_col.T
    two_m = np.sum(k)
    
    # Avoid division by zero if the subgraph has no edges
    if two_m == 0:
        return
        
    P = k_outer / two_m
    B = A - P
    
    # Get eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(B)
    
    # Get the leading eigenvector 's'
    s = eigenvecs[:, np.argmax(eigenvals)]

    # Now, split the nodes based on the sign of their entry in 's'
    i = 0 # index for the eigenvector 's'
    for node in nodes_in_g:
        # If the number is positive, put it in the first new community
        if s[i].real > 0:
            node_community[node] = community_counter
        # Otherwise, put it in the second new community
        else:
            node_community[node] = community_counter + 1
        i += 1
        
    # We used two new community IDs, so increment the counter by 2
    community_counter += 2

def split_into_subgraphs(g):
    """
    This reads the global 'node_community' map and splits graph 'g'
    into two new subgraphs based on the IDs we just assigned in 'find_split'.
    """
    group_a_nodes = []
    group_b_nodes = []
    
    # Check if the node's community ID is even or odd
    for node in g.nodes():
        if node_community[node] % 2 == 0:
            group_a_nodes.append(node)
        else:
            group_b_nodes.append(node)
            
    # Create the new subgraphs (using the original graph 'G' to preserve all node properties)
    subgraph_a = nx.subgraph(G, group_a_nodes)
    subgraph_b = nx.subgraph(G, group_b_nodes)
    
    return subgraph_a, subgraph_b

def log_history(g, color_name):
    """
    Updates the 'node_history' log for every node in subgraph 'g'.
    This tells us which nodes were part of which split at which iteration.
    """
    for node in g.nodes():
        # Get the last iteration number for this node and add 1
        last_iter = node_history[node][-1][0]
        node_history[node].append((last_iter + 1, color_name))

def is_indivisible(g):
    """
    Checks if a subgraph should be split further.
    In your original code, this just checks if all nodes in 'g'
    have the same community ID.
    
    NOTE: A better check (from the handout) would be to see if the
    largest eigenvalue of B is <= 0. But we'll keep your logic!
    """
    if nx.is_empty(g):
        return True # Can't split an empty graph
        
    communities = []
    for node in g.nodes():
        communities.append(node_community[node])
        
    # If there's only 1 unique community ID, it's indivisible.
    is_terminal = (len(set(communities)) == 1)
    return is_terminal

def cut_edges_between(group_a, group_b):
    """
    Removes edges from our global 'drawing_graph' that
    connect a node in group_a to a node in group_b.
    """
    for node_a in group_a.nodes():
        for node_b in group_b.nodes():
            if drawing_graph.has_edge(node_a, node_b):
                drawing_graph.remove_edge(node_a, node_b)

# 3. The Main Recursive Splitter 

# Set up the figure for our iteration plots
mp.figure(figsize=(20, 16))

def run_recursive_splits(g):
    """
    This is the main recursive function. It takes a graph 'g' and:
    1. Splits it into two new subgraphs, 'group_a' and 'group_b'.
    2. Logs this split for our metrics.
    3. Updates the colors for plotting.
    4. Draws the 'before' and 'after' plots for this split.
    5. Calculates the new modularity score.
    6. Calls itself on 'group_a' and 'group_b' to split them further.
    """
    if nx.is_empty(g):
        return  # Stop recursing if the graph is empty
    else:
        global plot_position
        global node_colors
        
        # 1. Find the split and update the global 'node_community' map
        find_split(g)
        
        # 2. Get the two new subgraphs based on the split
        group_a, group_b = split_into_subgraphs(g)
        
        # 3. This part (from your original code) re-calculates community IDs 
        #    for the *new* subgraphs. This prepares them for the *next* recursive call.
        find_split(group_a)
        find_split(group_b)
        
        # 4. Log this iteration for our metric plots
        log_history(group_a, "blue") # 'blue' is just a placeholder
        log_history(group_b, "red")  # 'red' is just a placeholder

        # 5. Update the global 'node_colors' list for plotting this step
        for node in G.nodes():
            if node in group_a.nodes():
                node_colors[node] = plot_colors[plot_position]
            if node in group_b.nodes():
                node_colors[node] = plot_colors[plot_position + 1]

        # 6. Draw the 'BEFORE' split plot
        mp.subplot(2, 2, plot_position + 1)
        nx.draw_spring(drawing_graph, with_labels=True, node_color=node_colors)
        mp.title(f"Iteration {plot_position // 2}: Before Split")

        # 7. Cut the edges on our 'drawing_graph' for the next plot
        cut_edges_between(group_a, group_b)
        
        # 8. Calculate the modularity of this new state
        get_modularity_score(drawing_graph)
        
        # 9. Draw the 'AFTER' split plot
        mp.subplot(2, 2, plot_position + 2)
        nx.draw_spring(drawing_graph, with_labels=True, node_color=node_colors)
        mp.title(f"Iteration {plot_position // 2}: After Split")

        # 10. Move to the next plot positions
        plot_position += 2
        
        # 11. Recurse!
        #    We only continue if the new group is not empty
        #    and is not "indivisible" (i.e., it's not a final community)
        if not nx.is_empty(group_a) and not is_indivisible(group_a):
            run_recursive_splits(group_a)
        if not nx.is_empty(group_b) and not is_indivisible(group_b):
            run_recursive_splits(group_b)

# 4. Running the Algorithm 

# First, let's get the modularity of the *original* graph
get_modularity_score(G)

# Start the recursive splitting!
run_recursive_splits(G)

# Clean up the history log (remove the "null" entry we started with)
for node in node_history.keys():
    node_history[node].pop(0)

# Find out how many iterations we actually ran
max_iterations = 0
for node in node_history.keys():
    num_splits = len(node_history[node])
    if num_splits > max_iterations:
        max_iterations = num_splits

# 5. Metrics for each iteration 

# These will store the metric dictionaries for each iteration.
# e.g., degree_history[1] = {0: 0.5, 1: 0.3, ...}
#       degree_history[2] = {0: 0.4, 1: 0.2, ...}
iter_colors = {i: [] for i in range(1, max_iterations + 1)}
degree_history = {i: {} for i in range(1, max_iterations + 1)}
betweenness_history = {i: {} for i in range(1, max_iterations + 1)}
closeness_history = {i: {} for i in range(1, max_iterations + 1)}
clustering_history = {i: {} for i in range(1, max_iterations + 1)}

def calculate_metrics_by_iteration(history_log, max_iter):
    """
    This function rebuilds the subgraphs that existed at each
    iteration and calculates the centrality metrics for them.
    
    NOTE: The assignment asked for metrics on the *full* graph 'G'.
    Your code calculates them on the *subgraphs* as they split.
    This version keeps your original (and interesting!) logic.
    """
    color_index = 0
    for i in range(1, max_iter + 1):
        group_red_nodes = []
        group_blue_nodes = []
        
        # Rebuild the groups for this iteration
        for node in history_log.keys():
            # Check the log entry for this iteration
            if i <= len(history_log[node]):
                log_entry = history_log[node][i-1] # (i-1) because lists are 0-indexed
                
                if log_entry[1] == "red":
                    group_red_nodes.append(node)
                    iter_colors[i].append(plot_colors[color_index])
                else:
                    group_blue_nodes.append(node)
                    iter_colors[i].append(plot_colors[color_index + 1])
                        
        # Create the subgraphs from that iteration
        sub_a = nx.subgraph(G, group_red_nodes)
        sub_b = nx.subgraph(G, group_blue_nodes)

        # Calculate metrics for each subgraph and merge the results
        # The | operator (Python 3.9+) is a clean way to merge two dictionaries
        all_deg_c = nx.degree_centrality(sub_a) | nx.degree_centrality(sub_b)
        degree_history[i] = dict(sorted(all_deg_c.items()))
        
        all_bet_c = nx.betweenness_centrality(sub_a) | nx.betweenness_centrality(sub_b)
        betweenness_history[i] = dict(sorted(all_bet_c.items()))
        
        all_clo_c = nx.closeness_centrality(sub_a) | nx.closeness_centrality(sub_b)
        closeness_history[i] = dict(sorted(all_clo_c.items()))
        
        all_clus_co = nx.clustering(sub_a) | nx.clustering(sub_b)
        clustering_history[i] = dict(sorted(all_clus_co.items()))
        
        color_index += 2

# Fill in the metric history dictionaries
calculate_metrics_by_iteration(node_history, max_iterations)

# This is a clever part from your code:
# If a node wasn't in a split at iteration 2, it means
# it was in a "terminal" community. We should carry over
# its value from iteration 1 so the line plot continues.
for node in G.nodes():
    if node not in degree_history[2]:
        degree_history[2][node] = degree_history[1][node]
    if node not in betweenness_history[2]:
        betweenness_history[2][node] = betweenness_history[1][node]
    if node not in closeness_history[2]:
        closeness_history[2][node] = closeness_history[1][node]
    if node not in clustering_history[2]:
        clustering_history[2][node] = clustering_history[1][node]

# 6. Plotting Our Results 

# First, let's show the final communities
mp.figure(figsize=(16, 10))

# Plot 1: Show the final communities, but keep all original edges
mp.subplot(1, 2, 1)
# We use kamada_kawai_layout for a nice "force-directed" look
nx.draw_kamada_kawai(G, with_labels=True, node_color=node_colors)
mp.title("Final Communities (All Edges Shown)")

# Plot 2: Show the final communities with the "cut" edges removed
# This uses the 'drawing_graph' we've been cutting edges from
mp.subplot(1, 2, 2)
nx.draw_spring(drawing_graph, with_labels=True, node_color=node_colors)
mp.title("Final Communities (Inter-Community Edges Removed)")
mp.show()

# --- Now, plot the evolution of metrics for each node ---

# We'll need these to plot against
iterations = [1, 2]
all_nodes = sorted(list(G.nodes())) # Sort the nodes so the plots are tidy

# Get the metric values for each iteration, sorted by node ID
vals_iter1_deg = [degree_history[1][node] for node in all_nodes]
vals_iter2_deg = [degree_history[2][node] for node in all_nodes]

# Plot Degree Centrality
mp.figure(figsize=(16, 10))
for node in all_nodes:
    # Make a list of this node's metric values, e.g., [0.5, 0.4]
    y_values = [vals_iter1_deg[node], vals_iter2_deg[node]]
    mp.plot(iterations, y_values, 'o-', label=f"Node {node}") # 'o-' adds markers
mp.xlabel('Iterations')
mp.ylabel('Degree Centrality')
mp.legend(loc='center left', bbox_to_anchor=(1, 0.5)) # Move legend outside plot
mp.title('Degree Centrality Evolution')
mp.xticks(iterations) # Make sure x-axis only shows 1 and 2
mp.show()

# Get values for Betweenness
vals_iter1_bet = [betweenness_history[1][node] for node in all_nodes]
vals_iter2_bet = [betweenness_history[2][node] for node in all_nodes]

# Plot Betweenness Centrality
mp.figure(figsize=(16, 10))
for node in all_nodes:
    y_values = [vals_iter1_bet[node], vals_iter2_bet[node]]
    mp.plot(iterations, y_values, 'o-', label=f"Node {node}")
mp.xlabel('Iterations')
mp.ylabel('Betweenness Centrality')
mp.legend(loc='center left', bbox_to_anchor=(1, 0.5))
mp.title('Betweenness Centrality Evolution')
mp.xticks(iterations)
mp.show()

# Get values for Closeness
vals_iter1_clo = [closeness_history[1][node] for node in all_nodes]
vals_iter2_clo = [closeness_history[2][node] for node in all_nodes]

# Plot Closeness Centrality
mp.figure(figsize=(16, 10))
for node in all_nodes:
    y_values = [vals_iter1_clo[node], vals_iter2_clo[node]]
    mp.plot(iterations, y_values, 'o-', label=f"Node {node}")
mp.xlabel('Iterations')
mp.ylabel('Closeness Centrality')
mp.legend(loc='center left', bbox_to_anchor=(1, 0.5))
mp.title('Closeness Centrality Evolution')
mp.xticks(iterations)
mp.show()

# Get values for Clustering
vals_iter1_clus = [clustering_history[1][node] for node in all_nodes]
vals_iter2_clus = [clustering_history[2][node] for node in all_nodes]

# Plot Clustering Coefficient
mp.figure(figsize=(16, 10))
for node in all_nodes:
    y_values = [vals_iter1_clus[node], vals_iter2_clus[node]]
    mp.plot(iterations, y_values, 'o-', label=f"Node {node}")
mp.xlabel('Iterations')
mp.ylabel('Clustering Coefficient')
mp.legend(loc='center left', bbox_to_anchor=(1, 0.5))
mp.title('Clustering Coefficient Evolution')
mp.xticks(iterations)
mp.show()

# Plotting the modularity score history 
mp.figure(figsize=(10, 6))
# The x-axis is just the steps: 0 (original), 1 (first split), 2 (second split)
x_axis_modularity = list(range(len(modularity_scores)))
mp.plot(x_axis_modularity, modularity_scores, 'o-')
mp.xlabel('Algorithm Step (0 = Original Graph)')
mp.ylabel('Modularity Score')
mp.title('Modularity Score per Split')
mp.xticks(x_axis_modularity)
mp.show()
