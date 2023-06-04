import cv2
import sknw
from skimage.morphology import skeletonize
import networkx as nx
import json
import matplotlib.pyplot as plt
import collections
from skimage import io
import numpy as np

plt.switch_backend("agg")


def remove_isolated_pixels(image):
    connectivity = 8

    output = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)

    num_stats = output[0]
    labels = output[1]
    stats = output[2]

    new_image = image.copy()

    for label in range(num_stats):
        if stats[label, cv2.CC_STAT_AREA] < 50:
            new_image[labels == label] = 0

    return new_image


def binariseImage(I):
    if len(I.shape) > 2:
        ind = I[:, :, 0] > 250
    else:
        ind = I > 250
    Ibin = np.zeros((I.shape[0], I.shape[1])).astype("uint8")
    Ibin[ind] = 255
    Ibin = remove_isolated_pixels(Ibin)
    return Ibin


def getGraph(img, basename):
    img = binariseImage(img) / 255
    ske = skeletonize(img).astype(np.uint16)
    # ske = img.astype('uint16')

    # build graph from skeleton
    graph = sknw.build_sknw(ske)

    # draw image
    plt.figure(figsize=(15, 15))
    plt.imshow(img, cmap="gray")

    # draw edges by pts
    for s, e in graph.edges():
        ps = graph[s][e]["pts"]
        plt.plot([ps[0, 1], ps[-1, 1]], [ps[0, 0], ps[-1, 0]], "green")

    # draw node by o
    nodes = graph.nodes()
    ps = np.array([nodes[i]["o"] for i in nodes])
    plt.plot(ps[:, 1], ps[:, 0], "r.")

    plt.savefig("%s_fig.jpg" % basename, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()
    open("%s_edges.dat" % basename, "w").write(
        str(graph.edges()).replace("(", "[").replace(")", "]")
    )
    open("%s_nodes.dat" % basename, "w").write(
        str(graph.nodes()).replace("(", "[").replace(")", "]")
    )

    edges = np.array(graph.edges())

    return edges, nodes


# build the networkx graph from node and egdes lists
def build_graph(nodes, edges):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    return G


# perform some analysis
def simple_analysis(G):
    no_nodes = G.number_of_nodes()
    no_edges = G.number_of_edges()
    assortativity = nx.degree_assortativity_coefficient(G)
    clustering = nx.average_clustering(G)
    compo = nx.number_connected_components(G)
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G0 = G.subgraph(Gcc[0])
    size_G0_edges = G0.number_of_edges()
    size_G0_nodes = G0.number_of_nodes()
    ratio_nodes = size_G0_nodes / no_nodes
    ratio_edges = size_G0_edges / no_edges
    return (
        no_nodes,
        no_edges,
        assortativity,
        clustering,
        compo,
        ratio_nodes,
        ratio_edges,
    )


# here we generate a histogram of degrees with a specified colour
def degree_histogram(savepath, G, colour="blue"):
    plt.figure()

    degree_sequence = sorted(
        [d for n, d in G.degree()], reverse=True
    )  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    fig1, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color=colour)
    plt.ylabel("Count")
    plt.xlabel("%Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)
    plt.savefig(savepath)
    plt.close()

    degrees = [0] * 6  # ordered from deg. 1 to 6
    for _deg, _cnt in zip(deg, cnt):
        if _deg - 1 < 6:
            degrees[_deg - 1] = _cnt

    return degrees


def graph_image(savepath, G):
    # here we generate an image where nodes are coloured according to their degrees
    plt.figure()
    node_color = [float(G.degree(v)) for v in G]
    nx.draw_spring(G, node_size=10, node_color=node_color)
    plt.savefig(savepath)

    plt.close()


def performGraphProcessing(imgfile, opt, basename, imgid):
    savepath_hist = "%s_hist.png" % basename
    savepath_graph = "%s_graph.png" % basename

    img = io.imread(imgfile)
    edges, nodes = getGraph(img, basename)
    G = build_graph(nodes, edges)
    # metrics: no_nodes,no_edges,assortativity, clustering, compo, ratio_nodes, ratio_edges
    metrics = simple_analysis(G)
    degrees = degree_histogram(savepath_hist, G, "goldenrod")
    graph_image(savepath_graph, G)

    opt.graphfid.write(
        "%s,%d,%d,%0.5f,%0.5f,%0.5f,%0.5f,%0.5f,%d,%d,%d,%d,%d,%d\n"
        % (imgid, *metrics, *degrees)
    )

    plt.close()

    return [savepath_graph, savepath_hist]
