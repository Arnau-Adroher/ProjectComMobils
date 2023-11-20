import math
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon

def calc_distance(punto1, punto2):
    x1, y1 = punto1
    x2, y2 = punto2

    distancia = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distancia
    
def generate_hexagon(center_x, center_y, size=1):
    hexagon = []
    for i in range(6):
        angle = 2 * math.pi / 6 * i
        x = center_x + size * math.cos(angle)
        y = center_y + size * math.sin(angle)
        hexagon.append((x, y))
    return hexagon


def get_sectors(x, y):
    sector1 = [(1.0 + x, 0.0 + y), (0.5 + x, 0.866 + y), (-0.5 + x, 0.866 + y), (0 + x, 0 + y)]
    sector2 = [(1.0 + x, 0.0 + y), (0.5 + x, -0.866 + y), (-0.5 + x, -0.866 + y), (0 + x, 0 + y)]
    sector3 = [(-0.5 + x, 0.866 + y), (-1.0 + x, 0.0 + y), (-0.5 + x, -0.866 + y), (0 + x, 0 + y)]
    sectors = [sector1, sector2, sector3]
    return sectors


def get_random_points_in_sectors(sectors): 
    random_points = []
    # posible sol; hacer que verifique que esta dentro
    # de los margenes del hexagono, si no esta que genere otro punto aleatorio hasta que este demtro del rango
    for vertices in sectors:
        polygon = Polygon(vertices)

        point_fount = False

        while not point_fount:

            x_values, y_values = zip(*vertices)
            min_x, max_x = min(x_values), max(x_values)
            min_y, max_y = min(y_values), max(y_values)

            random_x = random.uniform(min_x, max_x)
            random_y = random.uniform(min_y, max_y)

            random_point = Point(random_x, random_y)

            if polygon.contains(random_point):
                random_points.append((random_x, random_y))
                point_fount = True

    return random_points


def plot_hexagon_and_sectors(center_x, center_y):
    hexagon = generate_hexagon(center_x, center_y)
    sectors = get_sectors(center_x, center_y)
    random_points = get_random_points_in_sectors(sectors)  # Get the random points

    # Unpack the coordinates for plotting
    hex_x, hex_y = zip(*hexagon)
    sector_x = [np.array(sector)[:, 0] for sector in sectors]
    sector_y = [np.array(sector)[:, 1] for sector in sectors]

    plt.plot(hex_x + (hex_x[0],), hex_y + (hex_y[0],), marker='o', linestyle='-', color='b', label='Hexagon')

    for i in range(3):
        plt.fill(sector_x[i], sector_y[i], alpha=0.5, label=f'Sector {i + 1}')

    random_x, random_y = zip(*random_points)
    plt.scatter(random_x, random_y, c='r', marker='x', label='Random Points')

    plt.gca().set_aspect('equal', adjustable='box')


def calculate_hexagon_centers(n_layers, size=1):
    # Start with the center hexagon
    centers = [(0, 0)]
    if n_layers == 0:
        return centers

    angle_degrees = 30  # Each hexagon has 60 degrees between its vertices
    angle_radians = math.radians(angle_degrees)

    # Calculate centers of surrounding hexagons
    for i in range(1, n_layers + 1):
        # Distance from the center increases with each layer
        dist = 2 * i * size * math.cos(angle_radians)

        for j in range(0, 6 * i):
            if i == 2 and j % 2 != 0:
                dist = 3
            else:
                dist = 2 * i * size * math.cos(angle_radians)

            offset = (j * (360 / (i * 6)) + 30)

            x = centers[0][0] + math.cos(math.radians(offset)) * dist
            y = centers[0][1] + math.sin(math.radians(offset)) * dist
            print(offset)

            # Add center if unique and still need more hexagons
            if (x, y) not in centers:
                centers.append((x, y))

    print(centers)
    return centers



def plot_hexagons(n_layers):
    hexagon_centers = calculate_hexagon_centers(n_layers)

    for center in hexagon_centers:
        plot_hexagon_and_sectors(center_x=center[0], center_y=center[1])

    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.legend()

    plt.grid(True)
    plt.show()

def generate_shadow_fading(sigma_dB):
    sigma_linear = sigma_dB * np.log(10)/10

    shadow_fading = np.random.lognormal(mean=0, sigma=sigma_linear, size=1) #Standard normal, do we need to use the def on slide 13 on Cellular Network design?

    return shadow_fading

def lineal_to_db(number):
    x = 10 * np.log10(number)
    return x

def db_to_lineal(number):
    x = 10 ** (number/10)
    return x
def ex_1():
    SIR = 0
    centers = calculate_hexagon_centers(2)
    
    for center in centers:
        sectors = get_sectors(center[0],center[1])
        random_points = get_random_points_in_sectors(sectors) 
        
    print(random_points)   
def main():
    ###Values###
    v = 3.8
    layers = 2
    sigma_dB = 8
    ############
    plot_hexagons(2)

    # center_x = 0
    # center_y = 0
    # plot_hexagon_and_sectors(center_x, center_y)
    
if __name__ == '__main__':
    sys.exit(main())

