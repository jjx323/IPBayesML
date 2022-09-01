#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 17:00:22 2020

@author: Junxiong Jia
"""

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from fenics import Function

def mesh2triang(mesh):
    xy = mesh.coordinates()
    return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())

def mplot_cellfunction(cellfn, ax):
    C = cellfn.array()
    tri = mesh2triang(cellfn.mesh())
    return ax.tripcolor(tri, facecolors=C)

def mplot_function(f, ax, vmin, vmax):
    mesh = f.function_space().mesh()
    if (mesh.geometry().dim() != 2):
        raise AttributeError('Mesh must be 2D')
    # DG0 cellwise function
    if f.vector().size() == mesh.num_cells():
        C = f.vector().array()
        if vmin != None and vmax != None:
            return ax.tripcolor(mesh2triang(mesh), C, vmin=vmin, vmax=vmax)
        else:
            return ax.tripcolor(mesh2triang(mesh), C)
    # Scalar function, interpolated to vertices
    elif f.value_rank() == 0:
        C = f.compute_vertex_values(mesh)
        if vmin != None and vmax != None:
            return ax.tripcolor(mesh2triang(mesh), C, shading='gouraud', vmin=vmin, vmax=vmax)
        else:
            return ax.tripcolor(mesh2triang(mesh), C, shading='gouraud')
    # Vector function, interpolated to vertices
    elif f.value_rank() == 1:
        w0 = f.compute_vertex_values(mesh)
        if (len(w0) != 2*mesh.num_vertices()):
            raise AttributeError('Vector field must be 2D')
        X = mesh.coordinates()[:, 0]
        Y = mesh.coordinates()[:, 1]
        U = w0[:mesh.num_vertices()]
        V = w0[mesh.num_vertices():]
        if vmin != None and vmax != None:
            return ax.quiver(X,Y,U,V, vmin=vmin, vmax=vmax)
        else:
            return ax.quiver(X,Y,U,V)

## Plot a generic dolfin object (if supported)
#def plot(obj, ax=None, vmin=None, vmax=None):
#    if ax == None:
#        ax = plt
#    plt.gca().set_aspect('equal')
#    if isinstance(obj, Function):
#        return mplot_function(obj, ax, vmin, vmax)
#    elif isinstance(obj, CellFunctionSizet):
#        return mplot_cellfunction(obj, ax)
#    elif isinstance(obj, CellFunctionDouble):
#        return mplot_cellfunction(obj, ax)
#    elif isinstance(obj, CellFunctionInt):
#        return mplot_cellfunction(obj, ax)
#    elif isinstance(obj, Mesh):
#        if (obj.geometry().dim() != 2):
#            raise AttributeError('Mesh must be 2D')
#    return plt.triplot(mesh2triang(obj), color='#808080')
#
#    raise AttributeError('Failed to plot %s'%type(obj))