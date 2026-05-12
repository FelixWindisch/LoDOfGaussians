import sqlite3
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import random
import torch
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Not strictly needed in newer versions, but safe to include
from sklearn.neighbors import NearestNeighbors
import torch
import sys
from utils.read_write_model import read_next_bytes


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def construct_distance_graph(images_file, k=100, llff_hold = 10000000000000):
    text = os.path.exists(images_file)
    if not text:
        images_file = images_file[:-4] + ".bin" 

    with open(images_file, "r" if text else "rb") as file:
        if text:
            lines = [line.rstrip() for line in file]
            number_of_images = len(lines) //2 - 2
            positions = np.zeros((number_of_images, 3))
            quats = np.zeros((number_of_images, 4))
            names = []
            i = 0
            for line in lines[4:]:

                if len(line.split(" ")) == 11:
                    if i % llff_hold == 0 and llff_hold > 0 and llff_hold < 1_000_000:
                        i += 1
                        continue
                    split = line.split(" ")
                    positions[i, 0] = split[5]
                    positions[i, 1] = split[6]
                    positions[i, 2] = split[7]
                    quats[i, 0] = split[1]
                    quats[i, 1] = split[2]
                    quats[i, 2] = split[3]
                    quats[i, 3] = split[4]
                    names.append(split[-1]) 
                    i += 1       
        else:
            num_reg_images = read_next_bytes(file, 8, "Q")[0]
            positions = np.zeros((num_reg_images, 3))
            quats = np.zeros((num_reg_images, 4))
            names = []
            i = 0
            for _ in range(num_reg_images):
                if i % llff_hold == 0 and llff_hold > 0 and llff_hold < 1_000_000:
                        i += 1
                        continue
                binary_image_properties = read_next_bytes(
                    file, num_bytes=64, format_char_sequence="idddddddi"
                )
                image_id = binary_image_properties[0]
                quats[i] = np.array(binary_image_properties[1:5])
                positions[i] = np.array(binary_image_properties[5:8])
                camera_id = binary_image_properties[8]
                image_name = ""
                current_char = read_next_bytes(file, 1, "c")[0]
                while current_char != b"\x00":  # look for the ASCII 0 entry
                    image_name += current_char.decode("utf-8")
                    current_char = read_next_bytes(file, 1, "c")[0]
                names.append(image_name)
                num_points2D = read_next_bytes(file, num_bytes=8,
                                           format_char_sequence="Q")[0]
                x_y_id_s = read_next_bytes(file, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
                i += 1    
        # For some bullshit reason, camera objects are sorted in alphabetical order
        sorted_indices = sorted(range(len(names)), key=lambda i: names[i])
        sorted_indices = np.array(sorted_indices)
        positions = positions[sorted_indices]
        quats = quats[sorted_indices]
        
        
        
        for i in range(len(positions)):
            positions[i] = qvec2rotmat(quats[i]).transpose() @ positions[i]
        positions[i, 2] #/= 8
        
        points = positions

        k = 100
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(points)
        distances, indices = nbrs.kneighbors(points)

        view_graph = nx.DiGraph()

        # Add nodes
        for i in range(points.shape[0]):
            view_graph.add_node(i, pos=tuple(points[i]))

        # Add edges (skip the first neighbor since it's the point itself)
        for i in range(points.shape[0]):
            for j in range(1, k+1):
                neighbor_idx = indices[i][j]
                dist = distances[i][j]
                view_graph.add_edge(i, neighbor_idx, weight= float(dist))#float(1000.0/(np.sqrt(dist) + 15)))
    return view_graph



###############################################LEGACY######################################################
def pair_id_to_image_ids(pair_id, num_images):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) / 2147483647
    return image_id1, image_id2

def random_walk_node(G, node, node_count = None):
    neighbors = list(G.neighbors(node))  # Get adjacent nodes
    if not neighbors:
        return None  # No adjacent nodes

    # Get the edge weights
    weights = [(1.0/(G[node][neighbor].get('weight', 1)+20)) for neighbor in neighbors]

    # Normalize weights to sum to 1
    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]

    # Choose a neighbor based on probabilities
    chosen_node = random.choices(neighbors, weights=probabilities)[0]
    return chosen_node

def build_consistency_graph_from_colmap(colmap_database_path):
    # Connect to COLMAP database
    db_path = os.path.join(colmap_database_path + "database.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get the number of images
    cursor.execute("SELECT COUNT(*) FROM images;")
    num_images = cursor.fetchone()[0]

    # Query the co-visibility graph
    cursor.execute("SELECT pair_id, rows FROM two_view_geometries;")
    pairs = cursor.fetchall()

    # Build the graph
    G = nx.Graph()
    for pair_id, matches in pairs:
        img1, img2 = pair_id_to_image_ids(pair_id, num_images)
        if matches > 0:
            G.add_edge(img1, img2, weight=matches)
    return G



def metropolis_hastings_walk(G, node):
    """
    Perform a Metropolis-Hastings random walk on a weighted graph.
    
    Parameters:
    G (networkx.Graph): A weighted graph where edge weights influence transition probabilities.
    start_node: The starting node for the walk.
    num_steps (int): The number of steps to take in the walk.
    
    Returns:
    list: A list of visited nodes.
    """
    current_node = node
    while True:
        neighbors = list(G.neighbors(current_node))
        #if not neighbors:
        #   break  # Stop if there are no neighbors
        
        # Select a neighbor with probability proportional to edge weight
        weights = [G[current_node][nbr]['weight'] * 100 for nbr in neighbors]
        proposed_node = random.choices(neighbors, weights=weights)[0]
        
        # Compute acceptance probability
        deg_current = sum(G[current_node][nbr]['weight'] for nbr in G.neighbors(current_node))
        deg_proposed = sum(G[proposed_node][nbr]['weight'] for nbr in G.neighbors(proposed_node))
        acceptance_prob = min(1, deg_current / deg_proposed)
        
        # Accept or reject the move
        if random.random() < acceptance_prob:
            current_node = proposed_node    
            return current_node