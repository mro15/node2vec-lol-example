#! /usr/bin/env python3

import networkx as nx
from node2vec import Node2Vec
import argparse
import matplotlib.pyplot as plt

def read_args():
    parser = argparse.ArgumentParser(description="The parameters are:")
    parser.add_argument("--edges_file", type=str, help="path to the edges file", required=True)
    parser.add_argument("--nodes", action="append", help="nodes to get similarities", required=True)
    return parser.parse_args()

def build_graph(file_path):
    lol_graph=nx.read_edgelist(file_path)
    print(len(lol_graph.nodes()), lol_graph.nodes())
    print(len(lol_graph.edges()), lol_graph.edges())
    return lol_graph

def train_model(lol_graph):
    node2vec = Node2Vec(lol_graph, dimensions=10, walk_length=16, num_walks=50, workers=4, p=0.5, q=2)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    return model

def get_similarities(nodes, model):
    for i in nodes:
        print(i, model.wv.most_similar(i))

def main():
    args = read_args()
    lol_graph = build_graph(args.edges_file)
    nx.draw(lol_graph, with_labels = True)
    plt.show()
    model = train_model(lol_graph)
    get_similarities(args.nodes, model)


if __name__ == "__main__":
    main()        
