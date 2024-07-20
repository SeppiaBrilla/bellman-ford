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

def generate_graph(num_nodes, num_edges, allow_negative_cycle):

        adiacent_matrix = np.random.uniform(low=LOWER_BOUND, high=UPPER_BOUND, size=(num_nodes, num_nodes))

        for i in range(num_nodes):
            adiacent_matrix[i,i] = 0
        
        G = nx.from_numpy_array(adiacent_matrix)

        it = 0
        if num_edges != "rand":
            while len(G.edges) != num_edges:
                idx = (0,0)
                print(f"num edges: {len(G.edges)}/{num_edges}. it: {it}", end="\r")
                if len(G.edges) > num_edges:
                    while not G.has_edge(idx[0], idx[1]) or idx[0] == idx[1]:
                        idx = (random.randint(0, num_nodes), random.randint(0, num_nodes))
                    G.remove_edge(idx[0], idx[1])
                elif len(G.edges) < num_edges:
                    while G.has_edge(idx[0], idx[1]) or idx[0] == idx[1]:
                        idx = (random.randint(0, num_nodes), random.randint(0, num_nodes))
                    G.add_edge(idx[0], idx[1], weight=random.randint(LOWER_BOUND, UPPER_BOUND))
                it += 1

        if allow_negative_cycle:
            return G
        while has_negative_cycle(G):
            for edge in G.edges():
                if G.edges[edge]["weight"] < 0:
                    n_negative = 0
                    for neib in [(edge[0] - 1, edge[1] + 1), (edge[0], edge[1] + 1), (edge[0] + 1, edge[1] + 1),
                                 (edge[0] - 1, edge[1]    ), (edge[0], edge[1]    ), (edge[0]    , edge[1] + 1),
                                 (edge[0] - 1, edge[1] - 1), (edge[0], edge[1] - 1), (edge[0] - 1, edge[1] + 1)]:
                        if G.has_edge(neib[0], neib[1]) and G.edges[neib]["weight"] < 0:
                            n_negative += 1
                    if n_negative < 1:
                        mult = 1
                    else:
                        mult = -1
                    w = mult * G.edges[edge]["weight"]
                    G.remove_edge(edge[0], edge[1])
                    G.add_edge(edge[0], edge[1], weight=w)

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

def main(n_nodes:int, n_edges:int|str, negative_cycle:bool = False):
    if type(n_edges) != str and n_edges > (n_nodes **2) - n_nodes:
        print("error. The number of edges should be less than n_nodes^2 - n_nodes")
        return
    nodes = [f"node_{n}" for n in range(n_nodes)]

    graph = generate_graph(n_nodes, n_edges, negative_cycle)
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

    elif len(sys.argv) < 4:
        print("error. not enough parameters. Usage: python graph_creator.py n_nodes n_edges negative_cycles")

    else:
        n_nodes = int(sys.argv[1])
        if sys.argv[2] != "rand":
            n_edges = int(sys.argv[2])
        else:
            n_edges = sys.argv[2]
        negative_cycles = bool(sys.argv[3])
        main(n_nodes, n_edges, negative_cycles)
