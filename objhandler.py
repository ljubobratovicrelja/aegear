"""Functions for reading and writing .obj files."""

import numpy as np


def read_obj(filename):
    """Reads a .obj file and returns the vertices and faces as numpy arrays."""
    vertices = []
    faces = []

    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()

            if not parts:
                continue

            if parts[0] == "v":  # vertex
                vertices.append([float(v) for v in parts[1:]])

            elif parts[0] == "f":  # face
                face = [int(idx.split('/')[0]) for idx in parts[1:]]
                faces.append(face)

    return np.array(vertices), np.array(faces)
