import numpy as np


def construct_graph(img):
    height, width, _ = img.shape

    def loc(i, j):
        return i * width + j
    
    def valid_point(i, j, ii, jj):
        return ii >= 0 and ii < height and jj >= 0 and jj < width and (ii != i or jj != j)

    length = height * width
    graph = np.zeros((length, length))
    get_w = lambda i, j, ii, jj: np.exp(
        -100 * np.linalg.norm(img[i, j] - img[ii, jj]) ** 2
    )
    for i in range(height):
        for j in range(width):
            first_loc = loc(i, j)
            # ranges
            for ii_offset in range(-20, 21):
                for jj_offset in range(-20, 21):
                    ii = i + ii_offset
                    jj = j + jj_offset
                    if valid_point(i,j,ii,jj):
                        second_loc = loc(ii, jj)
                        graph[first_loc, second_loc] = get_w(i, j, ii, jj)
    return graph


def graph_based_segmentation(img):
    graph = construct_graph(img)
    D = np.diag(np.sum(graph, axis=1))
    I = np.identity(graph.shape[0])
    D_inv = np.linalg.inv(D)
    A = I - (D_inv @ graph)
    vals, vecs = np.linalg.eig(A)
    second_smallest_index = np.argsort(vals)[1]
    second_smallest_eig_vec = vecs[:, second_smallest_index]
    return np.reshape(second_smallest_eig_vec, (img.shape[0], img.shape[1]))
