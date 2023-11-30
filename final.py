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
    
def calculate_angle(punto1, punto2):
    x1, y1 = punto1
    x2, y2 = punto2
    delta_x = x2 - x1
    delta_y = y2 - y1
    angle_radians = math.atan2(delta_y, delta_x)
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees    
    
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
            #print(offset)

            # Add center if unique and still need more hexagons
            if (x, y) not in centers:
                centers.append((x, y))

    #print(centers)
    return centers



def plot_hexagons(n_layers):
    plt.figure(1)
    hexagon_centers = calculate_hexagon_centers(n_layers)

    for center in hexagon_centers:
        plot_hexagon_and_sectors(center_x=center[0], center_y=center[1])

    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.legend()

    plt.grid(True)
    

def generate_shadow_fading(sigma_dB):

    shadow_fading = np.random.normal(loc=0, scale=sigma_dB) #Standard normal, do we need to use the def on slide 13 on Cellular Network design?

    return shadow_fading

def lineal_to_db(number):
    x = 10 * np.log10(number)
    return x

def db_to_lineal(number):
    x = 10 ** (number/10)
    return x

def calculate_percentage(sorted_SIR, threshold=-5):
    val = 0
    for i in sorted_SIR:
        val += 1
        if i >= threshold:
            percentage = (1 - (val / len(sorted_SIR))) * 100
            #print(f'SIR >= {threshold}: {percentage:.2f}%')
            break

    


    
def ex_1_2(v, sigma_dB, n):
    centers = calculate_hexagon_centers(2)
    #print(centers)
    ref_cent = 0,0
    list_of_SIR = []
    list_of_SIR_3 = []
    list_of_SIR_9 = []
    list_of_SIR_frac = []
    list_of_SIR_3_frac = []
    list_of_SIR_9_frac = []


    for i in range (1,50001):
        SIR = 0
        SIR_3 = 0
        SIR_9 = 0
        d_all = 0
        d_all_3 = 0
        d_all_9 = 0
        center_id = 0
        P=0
        a = 0
        a_all =0
        a_all_3 = 0
        a_all_9 = 0
        for center in centers:
            ####s'ha de comprobar si angle del punt respecte 0,0 és més petit = 120º
            c_x = center[0]
            c_y = center[1]
            sectors = get_sectors(c_x,c_y)           
            random_points = get_random_points_in_sectors(sectors)   
            if c_y == 0 and c_x == 0:
                x= db_to_lineal(generate_shadow_fading(sigma_dB))
                P=(calc_distance(random_points[0],ref_cent))**(n*v)/(x**(n))
                d = x/(calc_distance(random_points[0],ref_cent)**(v))
                a = d*P
            else:
                for i in range(0,3):
                    angle = calculate_angle(ref_cent,random_points[i])
                    #print(angle)
                    if angle <= 120 and angle >= 0:
                        x1= db_to_lineal(generate_shadow_fading(sigma_dB))
                        x2= db_to_lineal(generate_shadow_fading(sigma_dB))

                        P_k = (calc_distance(random_points[i],center))**(v*n)/(x1**(n))
                        if (calc_distance(random_points[i],center))>1:
                            print("error")
                        val =x2/(calc_distance(random_points[i],ref_cent)**(v))
                        d_all += val
                        a_all += val*P_k
                        if i == 0:
                            d_all_3 += val
                            a_all_3 += P_k * val
                            if center_id == 8 or center_id == 18:
                                d_all_9 += val
                                a_all_9 += P_k * val

                                #print(center)
            center_id += 1
        
        SIR = lineal_to_db(d/d_all)
        SIR_3 = lineal_to_db(d/d_all_3)
        SIR_9 = lineal_to_db(d/d_all_9)
        list_of_SIR.append(SIR)
        list_of_SIR_3.append(SIR_3)  
        list_of_SIR_9.append(SIR_9)   

        SIR_frac = lineal_to_db(a/a_all)
        SIR_3_frac = lineal_to_db(a/a_all_3)
        SIR_9_frac = lineal_to_db(a/a_all_9)
        list_of_SIR_frac.append(SIR_frac)
        list_of_SIR_3_frac.append(SIR_3_frac)  
        list_of_SIR_9_frac.append(SIR_9_frac)
        
              
        
    sorted_SIR = np.sort(list_of_SIR)
    sorted_SIR_3 = np.sort(list_of_SIR_3)
    sorted_SIR_9 = np.sort(list_of_SIR_9)

    sorted_SIR_frac = np.sort(list_of_SIR_frac)
    sorted_SIR_3_frac = np.sort(list_of_SIR_3_frac)
    sorted_SIR_9_frac = np.sort(list_of_SIR_9_frac)
    
    # Ex1
   # print('sorted_SIR')
    calculate_percentage(sorted_SIR)

   # print('sorted_SIR_3')
    calculate_percentage(sorted_SIR_3)

   # print('sorted_SIR_9')
    calculate_percentage(sorted_SIR_9)
    
    '''
    val = 0
    for i in sorted_SIR:
        val += 1
        if i >= -5:
            #print('SIR >= -5 frac 1: ',(1-(val/1000))*100,'%')
            break
    val = 0
    
    for i in sorted_SIR_3:
        val += 1
        if i >= -5:
            #print('SIR >= -5 frac 3: ',(1-(val/1000))*100,'%')
            break
    val = 0

    for i in sorted_SIR_9:
        val += 1
        if i >= -5:
            #print('SIR >= -5 frac 9: ',(1-(val/1000))*100,'%')
            break
            
    val = 0
    '''
    #EX2
    x = 0
    for i in sorted_SIR_3_frac:
        val += 1
        if i >= -5:
            x = (1-(val/1000))*100
            #print('SIR >= -5 frac 3: ',(1-(val/1000))*100,'%')
            break
  
    return x
    '''        
    # Calculate the cumulative distribution function for both arrays
    cumulative_prob = np.linspace(0, 1, len(sorted_SIR))
    cumulative_prob_3 = np.linspace(0, 1, len(sorted_SIR_3))
    cumulative_prob_9 = np.linspace(0, 1, len(sorted_SIR_9))

    cumulative_prob_frac = np.linspace(0, 1, len(sorted_SIR_frac))
    cumulative_prob_3_frac = np.linspace(0, 1, len(sorted_SIR_3_frac))
    cumulative_prob_9_frac = np.linspace(0, 1, len(sorted_SIR_9_frac))

    # Plot the CDF curves for both arrays
    plt.figure(2)
    plt.plot(sorted_SIR, cumulative_prob, label='CDF reuse factor 1', color='blue')
    plt.plot(sorted_SIR_3, cumulative_prob_3, label='CDF reuse factor 3', color='red')
    plt.plot(sorted_SIR_9, cumulative_prob_9, label='CDF reuse factor 9', color='green')

    plt.plot(sorted_SIR_frac, cumulative_prob_frac, label='CDF reuse factor 1', color='pink')
    plt.plot(sorted_SIR_3_frac, cumulative_prob_3_frac, label='CDF reuse factor 3', color='yellow')
    plt.plot(sorted_SIR_9_frac, cumulative_prob_9_frac, label='CDF reuse factor 9', color='orange')

    # Add labels and title
    plt.title('Cumulative Distribution Function (CDF) of Random Data')
    plt.xlabel('SIR(dB)')
    plt.ylabel('Cumulative Probability')

    plt.xlim(-20, 40)

    plt.grid(True)
    plt.legend()  # Show legend if multiple curves are plotted
   
    '''
    
def main():
    ###Values###
    v = 3.8
    layers = 2
    sigma_dB = 8
    ############
    #plot_hexagons(2)
    #ex_1(v,sigma_dB)
    
    max_n = 0
    max_x = 0
    array___r = []
    for i in range (1,21):
        n = i*(1/20)
        print(n)
        x = ex_1_2(v,sigma_dB,n)
        array___r.append(x)
        if x > max_x:
            max_x = x
            max_n = n
    
    print(max_n, max_x)
    print(array___r)

    x_values, y_values = zip(*array___r)

    # Plotting the points
    plt.scatter(x_values, y_values, label='Points', color='blue', marker='o')

    # Adding labels and title
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.title('Scatter Plot of Points')

    # Adding a legend
    plt.legend()

    # Display the plot
    plt.show()

    # center_x = 0
    # center_y = 0
    # plot_hexagon_and_sectors(center_x, center_y)
    
if __name__ == '__main__':
    sys.exit(main())
