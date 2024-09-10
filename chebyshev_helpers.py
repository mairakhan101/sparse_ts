import torch
import numpy as np
import pandas as pd
import einops

# Define the Chebyshev basis functions
def chebyshev_1d_basis(n, x):
    """ Computes the n-th Chebyshev polynomial of the first kind at x. """
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x
    else:
        T_prev = np.ones_like(x)
        T_curr = x
        for _ in range(2, n+1):
            T_next = 2 * x * T_curr - T_prev
            T_prev = T_curr
            T_curr = T_next
        return T_curr

def chebyshev_1d_basis_matrix(degree, p):
    """ Returns a matrix of Chebyshev basis functions for 1D. """
    x = np.linspace(-1, 1, p)
    basis = np.vstack([chebyshev_1d_basis(n, x) for n in range(degree)])
    return basis

def chebyshev_2d_basis_matrix(degree, p):
    """ Returns a 2D Chebyshev basis matrix. """
    x = np.linspace(-1, 1, p)
    X, Y = np.meshgrid(x, x)
    basis = np.array([[chebyshev_1d_basis(i, X) * chebyshev_1d_basis(j, Y) for j in range(degree)] for i in range(degree)])
    return basis.reshape(degree*degree, p*p)

def czt1d(tensor, chebyshev_basis):
    """ Transforms a tensor into the Chebyshev basis (1D). """
    return tensor @ chebyshev_basis.T

def czt2d(mat, chebyshev_basis):
    """ Transforms a tensor into the Chebyshev basis (2D). """
    shape = mat.shape
    mat = einops.rearrange(mat, '(x y) ... -> x y (...)', x=p, y=p)
    chebyshev_mat = torch.einsum('xyz,fx,Fy->fFz', mat, chebyshev_basis, chebyshev_basis)
    return chebyshev_mat.reshape(shape)

def analyse_chebyshev_2d(tensor, top_k=10):
    """ Analyzes the tensor in the Chebyshev basis. """
    values, indices = tensor.flatten().pow(2).sort(descending=True)
    rows = []
    total = values.sum().item()
    for i in range(top_k):
        rows.append([tensor.flatten()[indices[i]].item(),
                     values[i].item()/total, 
                     values[:i+1].sum().item()/total])
    display(pd.DataFrame(rows, columns=['Coefficient', 'Frac explained', 'Cumulative frac explained']))

def get_2d_chebyshev_component(tensor, x, y, chebyshev_basis):
    """ Projects the tensor onto the 2D Chebyshev Component (x, y). """
    vec = chebyshev_2d_basis_matrix(x, p)[y].flatten()
    return vec[:, None] @ (vec[None, :] @ tensor)
