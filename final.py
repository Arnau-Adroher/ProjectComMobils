import math
import random
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


def calculate_hexagon_centers(num_hexagons, size=1):
    # Start with the center hexagon
    centers = [(0, 0)]
    if num_hexagons == 1:
        return centers

    angle_degrees = 30
    angle_radians = math.radians(angle_degrees)

    cosine_of_angle = math.cos(angle_radians)

    dist = 2 * size * cosine_of_angle

    # Calculate centers of surrounding hexagons
    layer = 1
    while len(centers) < num_hexagons:
        for i in range(6):
            angle = angle_radians * i
            for j in range(layer):
                x = centers[-1][0] + math.cos(angle)
                y = centers[-1][1] + math.sin(angle)
            if len(centers) < num_hexagons:
                centers.append((x, y))
            else:
                break
        if len(centers) >= num_hexagons:
            break
    layer += 1

    return centers


def plot_hexagons(num_hexagons):
    hexagon_centers = calculate_hexagon_centers(num_hexagons)

    for center in hexagon_centers:
        plot_hexagon_and_sectors(center_x=center[0], center_y=center[1])

    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.legend()

    plt.grid(True)
    plt.show()

def main():
    ###Values###
    v = 3.8
    ############
   plot_hexagons(2)

    # center_x = 0
    # center_y = 0
    # plot_hexagon_and_sectors(center_x, center_y)
    
if __name__ == '__main__':
    sys.exit(main())

