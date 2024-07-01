import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull

def icosahedron_vertices():
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    vertices = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ])
    return vertices / np.linalg.norm(vertices[0])

def plot_icosahedron(vertices):
    hull = ConvexHull(vertices)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for simplex in hull.simplices:
        ax.plot(vertices[simplex, 0], vertices[simplex, 1], vertices[simplex, 2], 'r-')
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='b', s=100)
    plt.show()

def main():
    vertices = icosahedron_vertices()
    plot_icosahedron(vertices)

if __name__ == "__main__":
    main()
