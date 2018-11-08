#!/usr/bin/env python3

import numpy as np
import sys

def distance2(v1, v2):
    return np.linalg.norm(v2-v1)


class particle:
    def __init__(self,
                 pos=np.zeros(3),
                 vel=np.zeros(3),
                 rad=1,
                 mass=1,
                 color='Xe'):
        self.pos = pos
        self.vel = vel
        self.rad = rad
        self.mass = mass
        self.color = color

    def move(self, dt, L):
        self.pos = self.pos + self.vel * dt
        for i, x in enumerate(self.pos):
            if x < 0:
                self.pos[i] += L
            if x > L:
                self.pos[i] -= L

    def KE(self):
        return 0.5 * self.mass * np.linalg.norm(self.vel)**2

    def print_data(self):
        pos = ' '.join(map(str, self.pos))
        return '{} {}'.format(self.color, pos)


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

def set_particles_velocity(particles, vmax):
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

ng = 10
particles = [particle() for _ in range(ng**3)]
energy = 2.3
v_max = 5000
L_box = 20

xgrid = np.linspace(-L_box, L_box, ng)
ygrid = np.linspace(-L_box, L_box, ng)
zgrid = np.linspace(-L_box, L_box, ng)

put_particles_on_mesh(xgrid, ygrid, zgrid, particles)
set_particles_velocity(particles, v_max)
set_particles_energy(particles, energy)

tmax = 100
dt = 0.1

data_file = 'test.xyz'
with open(data_file, 'w') as f:
    f.write('')

for frame, t in enumerate(np.arange(0, tmax, dt)):
    for p in particles:
        p.move(dt, L_box)

    create_frame_data(data_file, particles, frame)

    sys.stderr.write('\rframe {} out of {}'.format(frame, int(tmax/dt)))
