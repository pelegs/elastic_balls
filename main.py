#!/usr/bin/env python3

import numpy as np
import sys


def distance2(v1, v2):
    return np.linalg.norm(v2-v1)

def distance(v1, v2):
    return np.sqrt(distance2(v1, v2))

def normalize(v):
    n = np.linalg.norm(v)
    if n != 0:
        return v / n
    else:
        return np.zeros(3)


class particle:
    def __init__(self,
                 pos=np.zeros(3),
                 vel=np.zeros(3),
                 rad=3,
                 mass=1,
                 color='Xe'):
        self.pos = pos
        self.vel = vel
        self.rad = rad
        self.mass = mass
        self.color = color

        self.cell = []
        self.neighbors = []

    def move(self, dt, L):
        self.pos = self.pos + self.vel * dt
        for i, x in enumerate(self.pos):
            if x < 0:
                self.pos[i] += L
            if x > L:
                self.pos[i] -= L

    def KE(self):
        return 0.5 * self.mass * np.linalg.norm(self.vel)**2

    def reset_neighbors(self):
        self.neighbors = []

    def add_neighbors(self, grid, N):
        for cell in neighbor_indices(self.cell, N):
            for p in grid.cells[cell[0]][cell[1]][cell[2]]:
                if p is not self:
                    self.neighbors.append(p)

    def check_collisions(self):
        for p2 in self.neighbors:
            if distance2(self.pos, p2.pos) <= (self.rad + p2.rad)**2:
                collision(self, p2)

    def print_data(self):
        pos = ' '.join(map(str, self.pos))
        return '{} {} #{}'.format(self.color, pos, self.cell)


class sim_grid:
    def __init__(self, N, L):
        self.cells = [[[[] for _ in range(N)]
                           for _ in range(N)]
                           for _ in range(N)]
        self.N = N
        self.L = L

    def insert(self, p):
        p.cell = np.floor(p.pos/self.L*self.N).astype(int)%self.N
        xcond = (0 <= p.cell[0] < self.N)
        ycond = (0 <= p.cell[1] < self.N)
        zcond = (0 <= p.cell[2] < self.N)
        if xcond and ycond and zcond:
            self.cells[p.cell[0]][p.cell[1]][p.cell[2]].append(p)

    def reset(self):
        self.cells = [[[[] for _ in range(self.N)]
                           for _ in range(self.N)]
                           for _ in range(self.N)]


def neighbor_indices(ind, N):
    indices = np.arange(0, N, 1)
    return [(indices[i], indices[j], indices[k])
            for i in range((ind[0]-1)%N, (ind[0]+2)%N)
            for j in range((ind[1]-1)%N, (ind[1]+2)%N)
            for k in range((ind[2]-1)%N, (ind[2]+2)%N)]

def put_particles_on_mesh(xa, ya, za, particles):
    positions = [np.array([x, y, z]) for x in xa for y in ya for z in za]
    for (i, p), pos in zip(enumerate(particles), positions):
        p.pos = pos

def momentum_center(particles):
    pcenter = np.zeros(3)
    for p in particles:
        pcenter += p.mass * p.vel
    return pcenter

def KE(particles):
    return np.sum([p.KE() for p in particles])

def set_particles_momentum(particles, vmax):
    vcenter = np.zeros(3)
    for p in particles:
        p.vel = np.random.uniform(-vmax, vmax, size=3)
        vcenter += p.mass * p.vel
    vcenter *= 1 / len(particles)
    for p in particles:
        p.vel -= vcenter / p.mass

def set_particles_energy(particles, E):
    Er = np.sqrt(E/KE(particles))
    for p in particles:
        p.vel *= Er


def create_frame_data(data_file, particles, frame):
    with open(data_file, 'a') as f:
        f.write('{}\n'.format(len(particles)))
        f.write('Frame # {}\n'.format(frame))
        for p in particles:
            f.write('{}\n'.format(p.print_data()))

def collision(p1, p2):
    dr = p2.pos - p1.pos
    dv = p2.vel - p1.vel
    M = p1.mass + p2.mass
    dist = distance(p1.pos, p2.pos)
    overlap = p1.rad + p2.rad - dist
    DOT = np.dot(dv, dr)/np.dot(dr, dr) * dr
    p1.vel = p1.vel + 2*p2.mass/M * DOT
    p2.vel = p2.vel - 2*p1.mass/M * DOT
    if p2 in p1.neighbors:
        p1.neighbors.remove(p2)
    if p1 in p2.neighbors:
        p2.neighbors.remove(p1)


ng = 10
particles = [particle() for _ in range(ng**3)]
energy = 50
v_max = 1000
L_box = 20

xgrid = np.linspace(-L_box, L_box, ng)
ygrid = np.linspace(-L_box, L_box, ng)
zgrid = np.linspace(-L_box, L_box, ng)

put_particles_on_mesh(xgrid, ygrid, zgrid, particles)
set_particles_momentum(particles, v_max)
set_particles_energy(particles, energy)

num_cells = ng-3
grid = sim_grid(num_cells, L_box)

tmax = 100
dt = 0.1

data_file = 'test.xyz'
with open(data_file, 'w') as f:
    f.write('')

for frame, t in enumerate(np.arange(0, tmax, dt)):
    grid.reset()
    for p in particles:
        p.move(dt, L_box)
        grid.insert(p)
    for p in particles:
        p.reset_neighbors()
        p.add_neighbors(grid, num_cells)
    for p in particles:
        p.check_collisions()

    create_frame_data(data_file, particles, frame)

    sys.stderr.write('\rframe {} out of {}: KE={:10.2f}'.format(frame,
                                                                int(tmax/dt),
                                                                KE(particles)
                                                                )
                    )
