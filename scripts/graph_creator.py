import os
import sys
import networkx as nx
import numpy as np
import random

LOWER_BOUND, UPPER_BOUND = -50, 1000

def add_random_edge(G, num_nodes, edges):

    source = random.randint(0, num_nodes - 1)
    target = random.randint(0, num_nodes - 1)
    while (source, target) in edges:
        source = random.randint(0, num_nodes - 1)

        target = random.randint(0, num_nodes - 1)
        while target == source:
            target = random.randint(0, num_nodes - 1)


    
    weight = random.randint(LOWER_BOUND, UPPER_BOUND)  
    while weight == 0:
        weight = random.randint(LOWER_BOUND, UPPER_BOUND)

    G.add_edge(source, target, weight=weight)
    return source, target

def has_negative_cycle(G):
    try:
        nx.bellman_ford_predecessor_and_distance(G, 0)
        return False  
    except nx.NetworkXUnbounded:
        return True   

def generate_graph(num_nodes, num_edges):
    G = nx.DiGraph()
    for i in range(num_nodes):
        G.add_node(i)
    edges = []
    for e in range(num_edges):
        source, target = add_random_edge(G, num_nodes, edges)
        if has_negative_cycle(G):
            G.remove_edge(source, target) 
        else:
            print(f"generated edge {e}/{num_edges}", end='\r')
            edges.append((source, target))
    return G

def count_elements_in_folder(folder_path):
    count = 0
    for _, _, files in os.walk(folder_path):
        count += len(files)
    return count

def graph_to_matrix(G):
    num_nodes = G.number_of_nodes()
    matrix = np.zeros(shape=(num_nodes, num_nodes))
    for edge in G.edges(data=True):
        source = edge[0] - 1
        target = edge[1] - 1
        weight = edge[2]['weight']
        matrix[source,target] = weight
    return matrix

def main(n_nodes:int, n_edges:int):
    if n_edges > (n_nodes **2) - n_nodes:
        print("error. The number of edges should be less than n_nodes^2 - n_nodes")
        return
    nodes = [f"node_{n}" for n in range(n_nodes)]

    graph = generate_graph(n_nodes, n_edges)
    n = count_elements_in_folder("graphs/")
    edges = graph_to_matrix(graph)

    graph_txt = ""
    nodes_txt = ",".join(nodes)
    graph_txt += nodes_txt + "\n"
    for i in range(n_nodes):
        row = [str(int(e)) for e in edges[i,:]]
        graph_txt += ",".join(row) + "\n"
    if graph_txt[-1] == "\n":
        graph_txt = graph_txt[:-1]
    n = count_elements_in_folder("graphs/")
    f = open(f"graphs/graph_{n}.txt","w")
    f.write(graph_txt)
    f.close()

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "auto":
        for n_nodes, n_edges in [(500, 200000)]:  
            print(f"generating graph with {n_nodes} nodes and {n_edges} edges")
            main(n_nodes, n_edges)
            print()
            print("graph generated")

    elif len(sys.argv) < 3:
        print("error. not enough parameters. Usage: python graph_creator.py n_nodes n_edges")

    else:
        n_nodes = int(sys.argv[1])
        n_edges = int(sys.argv[2])
        main(n_nodes, n_edges)
