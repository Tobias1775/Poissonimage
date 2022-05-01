import numpy as np
from scipy.sparse import linalg as linalg
from scipy.sparse import lil_matrix as lil_matrix
from os import path
import cv2
import mpmath

OMEGA = 0
BOUNDARY = 1
OUTSIDE = 2

def in_omega(index,mask):
    return mask[index] == 1

def on_boundary(index,mask):
    for x in surroundings(index):
        if mask[x] == 0:
            return True
    return False

def surroundings(index):
    i,j = index
    return [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]

def point_location(index,mask):
    if in_omega(index,mask) == True:
        if on_boundary(index,mask) == True:
            return BOUNDARY
        return OMEGA
    return OUTSIDE

def get_gradient_x(index,im):
    i,j = index
    dx = int(im[i+1,j]) - int(im[i,j])
    return dx

def get_gradient_y(index,im):
    i,j = index
    dy = int(im[i,j+1]) - int(im[i,j])
    return dy

def compare_gradient(index,target,background):
    f_dx = get_gradient_x(index,target)
    f_dy = get_gradient_y(index,target)
    g_dx = get_gradient_x(index,background)
    g_dy = get_gradient_y(index,background)
    if f_dx**2 + f_dy**2 >= g_dx**2 + g_dy**2:
        return target
    return background

def get_gradient(pt,target):
    dx = np.zeros(len(pt))
    dy = np.zeros(len(pt))
    im = target
    for i,index in enumerate(pt):
        # Mixed_gradient
        dx[i] = get_gradient_x(index,im)
        dy[i] = get_gradient_y(index,im)
    return dx,dy

def get_mixed_gradient(pt,target,background):
    dx = np.zeros(len(pt))
    dy = np.zeros(len(pt))
    for i,index in enumerate(pt):
        # Mixed_gradient
        im = compare_gradient(index,target,background)
        dx[i] = get_gradient_x(index,im)
        dy[i] = get_gradient_y(index,im)
    return dx,dy

def get_flatten_gradient(pt,edge_pt,target):
    dx = np.zeros(len(pt))
    dy = np.zeros(len(pt))
    for index in edge_pt:
        i = pt.index(index)
        dx[i] = get_gradient_x(index,target)
        dy[i] = get_gradient_y(index,target)
    return dx,dy

def get_laplacian(pt,interior_pt,dx,dy):
    lap = np.zeros(len(interior_pt))
    for i,index in enumerate(interior_pt):
        x,y = index
        dx1 = dx[pt.index((x,y))]
        dx2 = dx[pt.index((x-1,y))]
        dy1 = dy[pt.index((x,y))]
        dy2 = dy[pt.index((x,y-1))]
        lapx = dx1 - dx2
        lapy = dy1 - dy2
        lap[i] =lapx+lapy
    return lap

def mask_points(mask):
    indice = np.nonzero(mask)
    return list(zip(indice[0],indice[1]))

def edge_point(edge,pt):
    edge_pt = []
    for index in pt:
        if edge[index] == 1:
            edge_pt.append(index)
    return edge_pt
     
def get_interior_points(pt,mask):
    interior_pt = []
    for index in pt:
        if point_location(index,mask) == OMEGA:
            interior_pt.append(index)
    return interior_pt

def coefficient_matrix(interior_pt):
    n = len(interior_pt)
    A = lil_matrix((n,n))
    for i,index in enumerate(interior_pt):
        # Should have -4's diagonal
        A[i,i] = -4
        # Get all surrounding points
        for x in surroundings(index):
            if x not in interior_pt: continue
            A[i,interior_pt.index(x)] = 1
    return A

def fattal_transforamtion(dx,dy):
    n = len(dx)
    gradient_sum = sum(abs(dx)) + sum(abs(dy))
    avg_gradient = gradient_sum / n 
    alpha = 0.2 * avg_gradient
    beta = 0.2
    for i in range(n):
        if dx[i] != 0 and dy[i] != 0:
            dx[i] = (alpha ** beta) * (mpmath.power(abs(dx[i]),-beta)) * dx[i]
            dy[i] = (alpha ** beta) * (mpmath.power(abs(dy[i]),-beta)) * dy[i]
    return dx,dy

def local_illumination_change(target,mask,background):
    pt = mask_points(mask)
    interior_pt = get_interior_points(pt,mask)
    A = coefficient_matrix(interior_pt)
    composite = np.copy(background).astype(int)
    b = np.zeros(len(interior_pt))
    dx,dy = get_gradient(pt,target)
    new_dx,new_dy = fattal_transforamtion(dx,dy)
    lap = get_laplacian(pt,interior_pt,new_dx,new_dy)
    for i,index in enumerate(interior_pt):
        # Mixed_gradient
        # im = compare_gradient(index,target,background)
        b[i] = lap[i] 
        # If on boundry, add in target intensity
        # Creates constraint lapl source = target at boundary
        for x in surroundings(index):
            if point_location(x,mask) == BOUNDARY:
                b[i] -= background[x]
    x = linalg.cg(A, b)
    for i,index in enumerate(interior_pt):
        composite[index] = x[0][i]
    return composite

def flatten(target,mask,edge):
    pt = mask_points(mask)
    edge_pt = edge_point(edge,pt)
    interior_pt = get_interior_points(pt,mask)
    A = coefficient_matrix(interior_pt)
    composite = np.copy(target).astype(int)
    b = np.zeros(len(interior_pt))
    dx,dy = get_flatten_gradient(pt,edge_pt,target)
    lap = get_laplacian(pt,interior_pt,dx,dy)
    for i,index in enumerate(interior_pt):
        b[i] = lap[i] 
        for x in surroundings(index):
            if point_location(x,mask) == BOUNDARY:
                b[i] -= target[x]
    x = linalg.cg(A, b)
    for i,index in enumerate(interior_pt):
        composite[index] = x[0][i]
    return composite

def poisson_process(target,mask,background):
    pt = mask_points(mask)
    interior_pt = get_interior_points(pt,mask)
    A = coefficient_matrix(interior_pt)
    composite = np.copy(background).astype(int)
    b = np.zeros(len(interior_pt))
    dx,dy = get_gradient(pt,target)
    lap = get_laplacian(pt,interior_pt,dx,dy)
    for i,index in enumerate(interior_pt):
        # Mixed_gradient
        # im = compare_gradient(index,target,background)
        b[i] = lap[i] 
        # If on boundry, add in target intensity
        # Creates constraint lapl source = target at boundary
        for x in surroundings(index):
            if point_location(x,mask) == BOUNDARY:
                b[i] -= background[x]
    x = linalg.cg(A, b)
    for i,index in enumerate(interior_pt):
        composite[index] = x[0][i]
    return composite

def main():    
    scr_dir = 'D:\Program\pythonworkspace\poissonimage\input\\4'
    out_dir = 'D:\Program\pythonworkspace\poissonimage\output'
    target = cv2.imread(path.join(scr_dir, "target.png"))
    edge = cv2.imread(path.join(scr_dir,"edge.png"),
                      cv2.IMREAD_GRAYSCALE)
    # background = cv2.imread(path.join(scr_dir, "target.png"))    
    mask = cv2.imread(path.join(scr_dir, "mask.png"), 
                      cv2.IMREAD_GRAYSCALE)           
    mask[mask!=0] = 1
    # edge[edge!=0] = 1
    channels = target.shape[-1]
    # result_stack = [flatten(target[:,:,i],mask,edge) for i in range(channels)]
    # result_stack = [poisson_process(target[:,:,i],mask,background[:,:,i]) for i in range(channels)]
    # result = cv2.merge(result_stack)
    # cv2.imwrite(path.join(out_dir, "possion9.png"), result)
    


if __name__ == '__main__':
    main()