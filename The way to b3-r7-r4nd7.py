#!/usr/bin/env python
# coding: utf-8

# # The way to b3-r7-r4nd7

# This notebook is part of a coding challenge sponsored by get-in-IT and Bertrandt. The task is as follows: 
# 
# "Dein Ziel ist der Planet b3-r7-r4nd7. Es gilt den schnellsten Weg von dem Knotenpunkt "Erde" aus zu finden. Dabei kannst Du nur von Himmelskörper zu Himmelskörper reisen. Alle Wegstrecken und Planeten findest Du in dem JSON-File. Dabei kannst Du Dich von Graph.nodes[Graph.edges[i].source] nach Graph.nodes[Graph.edges[i].target] und umgekehrt bewegen (ungerichteter Graph). Die Graph.edges[i].cost geben die Entfernung zwischen den beiden Planeten an."
# 
# The target is to reach planet b3-r7-r4nd7. You have to find the fastest way from planet earth. We can jump from planet to planet. All paths and planets are located in the json file. We can move from Graph.nodes[Graph.edges[i].source] to Graph.nodes[Graph.edges[i].target] and vice versa (undirected ghraph).  Graph.edges[i].cost are showing the cost between both planets.
# 
# https://www.get-in-it.de/coding-challenge?utm_source=magazin&utm_medium=advertorial&utm_campaign=coding-challenge
# 
# Lets switch to the commando bridge of the space explorer space carrier to see how we can reach b3-r7-r4nd7.

# On the bridge of the space explorer ship, the captain and his first officer are planning their next journey to planet b3-r7-r4nd7

# In[111]:


from collections import defaultdict
from IPython.display import Image
from pyvis import network as net
from pyvis.network import Network
from random import choices
from string import ascii_lowercase
import json
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


# At first the packages need to be loaded into the system.

# ## Data Preperation

# In[116]:


with open('Data/generatedGraph.json') as f:
    d = json.load(f)


# From mission control they received a JSON file with the coordinates to the target. These coordinates need to be loaded into the system. The first officer is asking the computer the transform the file into the readable pandas table format.

# In[117]:


data = pd.DataFrame.from_dict(d, orient='index')
data


# In[118]:


# return the transpose 
result = data.transpose() 
  
# Print the result 
result.head()


# These steps are being done to seperate nodes and edges, there is no allocation being done between the two.

# In[119]:


#We are isolating the edges list
edges = result.iloc[:,1]
edges.head()


# In[120]:


#And the nodes list
nodes = result.iloc[:,0]
nodes.dropna(inplace=True)
#There are 1000 nodes and 1500 edges, so we need to drop the "None" nodes
nodes.head()


# In[121]:


nodes = pd.DataFrame(nodes.values.tolist(), index=nodes.index)
nodes.head()


# In[122]:


#We are looking for earth in the nodes list
nodes[nodes["label"].astype(str).apply(lambda x: 'Erde' in x)]


# The computer has found our starting point which is located at node 18

# In[123]:


#We are looking for the planet b3-r7-r4nd7 in the nodes list
nodes[nodes["label"].astype(str).apply(lambda x: 'b3-r7-r4nd7' in x)]


# We need to go to b3-r7-r4nd7 which is located at node 246

# In[124]:


# The edges list will be transformed....
edges = pd.DataFrame(edges.values.tolist(), index=edges.index)
edges["source"] = "N_" + edges["source"].astype(str)
edges["target"] = "N_" + edges["target"].astype(str)
cols = list(edges.columns.values) #Make a list of all of the columns in the df
cols.pop(cols.index('cost'))
edges = edges[cols+['cost']] #Create new dataframe with columns in the order you want
edges.to_csv('Data/edges.CSV',sep=',')
edges.head()


# This will be the format in which we will use the coordinates to find our way and analyze.

# In[125]:


# We are creating a network to explore the galaxy...
net = Network(height="750px", width="100%", bgcolor="white", font_color="black")

# set the physics layout of the network
net.barnes_hut()

sources = edges['source']
targets = edges['target']
weights = edges['cost']

edge_data = zip(sources, targets, weights)

for e in edge_data:
    src = e[0]
    dst = e[1]
    w = e[2]

    net.add_node(src, src, title=src)
    net.add_node(dst, dst, title=dst)
    net.add_edge(src, dst, value=w)

neighbor_map = net.get_adj_list()

net.show_buttons(filter_=['physics'])
net.show("network.html")


# This is way to complicated! It will be a challenge to navigate through all these planets.

# In[126]:


plt.figure(figsize=(50,60))
g = nx.from_pandas_edgelist(edges, source='source', target='target', edge_attr='cost')
nx.draw(g, width=1, edge_color="black", node_color='blue')
plt.savefig("Data/network.png") # save as png
nx.write_gexf(g, "Data/network.gexf")


# This looks quite complicated, lets startup the navigation system.

# ## Pathfinding

# For finding the shortest path, we will be using the Dijkstra algorithm.
# 
# "Dijkstra’s algorithm, published in 1959 and named after its creator Dutch computer scientist Edsger Dijkstra, can be applied on a weighted graph. The graph can either be directed or undirected. One stipulation to using the algorithm is that the graph needs to have a nonnegative weight on every edge."
# https://brilliant.org/wiki/dijkstras-short-path-finder/

# Steps in the algorithm:
# 
# We step through Dijkstra's algorithm on the graph used in the algorithm above:
# 
# 1. Initialize distances according to the algorithm. 

# In[127]:


Image(filename='Data/Step1.png') 


# 2. Pick first node and calculate distances to adjacent nodes. 

# In[128]:


Image(filename='Data/Step2.png') 


# 3. Pick next node with minimal distance; repeat adjacent node distance calculations. 

# In[129]:


Image(filename='Data/Step3.png') 


# 4. Final result of shortest-path tree 

# In[130]:


Image(filename='Data/Step4.png') 


# Source: Dijkstra's Shortest Path Algorithm. Brilliant.org. Retrieved 11:45, May 23, 2019, from https://brilliant.org/wiki/dijkstras-short-path-finder/

# In[131]:


class Network():
    def __init__(self):
        """
        self.edges is a dict of all possible next nodes
        e.g. {'X': ['A', 'B', 'C', 'E'], ...}
        self.weights has all the weights between two nodes,
        with the two nodes as a tuple as the key
        e.g. {('X', 'A'): 7, ('X', 'B'): 2, ...}
        """
        self.edges = defaultdict(list)
        self.weights = {}
    
    def add_edge(self, from_node, to_node, weight):
        # Note: assumes edges are bi-directional
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.weights[(from_node, to_node)] = weight
        self.weights[(to_node, from_node)] = weight


# In[132]:


network = Network()


# We startup the system and define the graph

# In[133]:


edges_matrix = edges.as_matrix()
edges_matrix


# In[134]:


#Go through edge matrix and fill network class
for edge in edges_matrix:
    network.add_edge(*edge)


# We put the edges into a matrix.

# In[135]:


def dijsktra(network, initial, end):
    # shortest paths is a dict of nodes
    # whose value is a tuple of (previous node, weight)
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()
    
    while current_node != end:
        visited.add(current_node)
        destinations = network.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = network.weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)
        
        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return "Route Not Possible"
        # next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])
    
    # Work back through destinations in shortest path
    path = []
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        current_node = next_node
    # Reverse path
    path = path[::-1]
    return path

#http://benalexkeen.com/implementing-djikstras-shortest-path-algorithm-with-python/


# This is where the calculation happens. It will go through the steps as outlined before. Please note that the nodes can be bi-directional in this case.

# In[136]:


way_to_b3_r7_r4nd7 = dijsktra(network, 'N_18', 'N_246')
way_to_b3_r7_r4nd7


# There is our way! Lets have a look if it is the correct one and make some Analysis!

# In[137]:


way_to_b3_r7_r4nd7 = pd.DataFrame(way_to_b3_r7_r4nd7,columns=['source'])
way_to_b3_r7_r4nd7['target']= way_to_b3_r7_r4nd7 ['source']
way_to_b3_r7_r4nd7['target'] = way_to_b3_r7_r4nd7['target'].shift(-1)
way_to_b3_r7_r4nd7 = pd.merge(way_to_b3_r7_r4nd7, edges, how='left', on=['source','target'])


way_to_b3_r7_r4nd7 ['source_2'] = way_to_b3_r7_r4nd7['source']
way_to_b3_r7_r4nd7['target_2'] = way_to_b3_r7_r4nd7['target']
edges ['source_2'] = edges['target']
edges['target_2'] = edges['source']
way_to_b3_r7_r4nd7 = pd.merge(way_to_b3_r7_r4nd7, edges, how='left', on=['source_2','target_2'])
way_to_b3_r7_r4nd7.drop(columns =["target_2","source_2","source_y","target_y"], inplace = True)


way_to_b3_r7_r4nd7["combined_cost"] = way_to_b3_r7_r4nd7['cost_x'].combine_first(way_to_b3_r7_r4nd7['cost_y'])
way_to_b3_r7_r4nd7["source_x"] = way_to_b3_r7_r4nd7["source_x"].str.replace("N_18", "Earth")
way_to_b3_r7_r4nd7["target_x"] = way_to_b3_r7_r4nd7["target_x"].str.replace("N_246", "b3-r7-r4nd7")
way_to_b3_r7_r4nd7 = way_to_b3_r7_r4nd7.drop(way_to_b3_r7_r4nd7.index[7])
way_to_b3_r7_r4nd7['source_x'] = "Planet " + way_to_b3_r7_r4nd7['source_x']
way_to_b3_r7_r4nd7['target_x'] = "Planet " + way_to_b3_r7_r4nd7['target_x']
way_to_b3_r7_r4nd7["combined_way"] = way_to_b3_r7_r4nd7['source_x'] + " to " + way_to_b3_r7_r4nd7['target_x']

way_to_b3_r7_r4nd7.to_csv('Data/way_to_b3_r7_r4nd7.CSV',sep=',')
way_to_b3_r7_r4nd7


# Lets put the data in one table.

# In[138]:


sum(way_to_b3_r7_r4nd7['combined_cost'])


# Between Earth and b3-r7-r4nd7 the way will be 3 units long.

# In[139]:


way_to_b3_r7_r4nd7.describe()


# In the seven jumps, the average distance between each planet will be 0.43. The lowest distance will be 0.04 (first jump after Earth) and maximum 0.77 (Planet N_519 to Planet N_71).

# In[140]:


plt.figure(figsize=(50,40))

g = nx.convert_matrix.from_pandas_edgelist(way_to_b3_r7_r4nd7, source='source_x', target='target_x', edge_attr='combined_cost')

pos = nx.spring_layout(g)
nx.draw_networkx_nodes(g, pos, node_size=2000)

#edges
nx.draw_networkx_edges(g, pos, width=20, alpha=0.5, edge_color='b', style='dashed')
#nx.draw_networkx_edge_labels(g, pos, font_size=30)

pos_attrs = {}
for node, coords in pos.items():
    pos_attrs[node] = (coords[0], coords[1] + 0.08)

nx.draw_networkx_labels(g, pos_attrs,font_size=40, font_family='sans-serif')

plt.axis('off')
plt.savefig("Data/weighted_graph.png") # save as png
plt.show()


# The graph is shown above but let us have a look on the distance between the jumps.

# In[141]:


sns.set(font_scale = 1)
plt.figure(figsize=(40,12))
plt.xticks(rotation=45)
sns.barplot(x = 'target_x', y = 'combined_cost', data=way_to_b3_r7_r4nd7,  color = "#3182bd")
plt.savefig("Data/barchart.png") # save as png


# There are 4 jumps which are larger than 0.5 distance units. We have to be aware of this in order to plan ahead!

# In[142]:


Image(filename='Data/network_gephi.png') 


# I visualized our way in the program "Gephi" based on the export I did previously ("network.gexf"). The route is logged into the computer. Everything is ready for our journey.
# 
# # Engage!
