import numpy as np


def construct_graph(img):
    height, width, _ = img.shape

    def loc(i, j):
        return i * width + j

    def valid_point(i, j, ii, jj):
        return (
            ii >= 0 and ii < height and jj >= 0 and jj < width and (ii != i or jj != j)
        )

    def w(i,j,ii,jj):
        return np.exp(-100 * np.linalg.norm(img[i, j] - img[ii, jj]) ** 2)

    length = height * width
    graph = np.zeros((length, length))
    for i in range(height):
        for j in range(width):
            first_loc = loc(i, j)
            # ranges
            for ii_offset in range(-20, 21):
                for jj_offset in range(-20, 21):
                    ii = i + ii_offset
                    jj = j + jj_offset
                    if valid_point(i, j, ii, jj):
                        second_loc = loc(ii, jj)
                        graph[first_loc, second_loc] = w(i, j, ii, jj)
    return graph


def graph_based_segmentation(img):
    graph = construct_graph(img)
    D = np.diag(np.sum(graph, axis=1))
    A = np.identity(graph.shape[0]) - np.linalg.inv(D) @ graph
    eig_vals, eig_vecs = np.linalg.eigh(A)
    second_smallest_eig_vec = eig_vecs[:, np.argsort(eig_vals)[1]]
    return np.reshape(second_smallest_eig_vec, img.shape[:2])