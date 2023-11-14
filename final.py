import math
import random
import matplotlib.pyplot as plt
import numpy as np

def generate_hexagon(center_x, center_y):
    hexagon = []
    for i in range(6):
        angle = 2 * math.pi / 6 * i
        x = round(center_x + math.cos(angle), 3)
        y = round(center_y + math.sin(angle), 3)
        hexagon.append((x, y))
    
    return hexagon

def get_sectors(x,y):
    sector1 = [(1.0+x, 0.0+y), (0.5+x, 0.866+y),(-0.5+x, 0.866+y),(0+x,0+y)]
    sector2 = [(1.0+x, 0.0+y),(0.5+x, -0.866+y),(-0.5+x, -0.866+y),(0+x,0+y)]
    sector3 = [(-0.5+x, 0.866+y), (-1.0+x, 0.0+y), (-0.5+x, -0.866+y),(0+x,0+y)]
    sectors = [sector1,sector2,sector3]
    return sectors
    
def get_random_points_in_sectors(sectors):      ######DOES NOT WORK, NEED TO BE IMPLMENTED
    random_points = []
    
    for sector in sectors:
        
        vertices = sector
        x_values, y_values = zip(*vertices)
        
        min_x, max_x = min(x_values), max(x_values)
        min_y, max_y = min(y_values), max(y_values)
        
        random_x = random.uniform(min_x, max_x)
        random_y = random.uniform(min_y, max_y)
        
        random_points.append((random_x, random_y))
    
    return random_points
    

def plot_hexagon_and_sectors(center_x, center_y):
    hexagon = generate_hexagon(center_x, center_y)
    sectors = get_sectors(center_x, center_y)
    random_points = get_random_points_in_sectors(sectors)  # Get the random points

    # Unpack the coordinates for plotting
    hex_x, hex_y = zip(*hexagon)
    sector_x = [np.array(sector)[:,0] for sector in sectors]
    sector_y = [np.array(sector)[:,1] for sector in sectors]

    plt.plot(hex_x + (hex_x[0],), hex_y + (hex_y[0],), marker='o', linestyle='-', color='b', label='Hexagon')

    for i in range(3):
        plt.fill(sector_x[i], sector_y[i], alpha=0.5, label=f'Sector {i+1}')

    random_x, random_y = zip(*random_points)
    plt.scatter(random_x, random_y, c='r', marker='x', label='Random Points')
    
    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    plt.grid(True)
    plt.show()

center_x = 0
center_y = 0
plot_hexagon_and_sectors(center_x, center_y)
