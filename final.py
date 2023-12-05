import math
import random
import sys
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon
import time
import threading

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
    plt.figure()
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
    return percentage

    


    
def simulator(v, sigma_dB, n,num_samples):
    centers = calculate_hexagon_centers(2)
    #print(centers)
    ref_cent = 0,0
    list_of_SIR = []
    list_of_SIR_3 = []
    list_of_SIR_9 = []
    list_of_SIR_frac = []
    list_of_SIR_3_frac = []
    list_of_SIR_9_frac = []


    for i in range (1,num_samples+1):
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
    p1 = calculate_percentage(sorted_SIR)

   # print('sorted_SIR_3')
    p2 = calculate_percentage(sorted_SIR_3)

   # print('sorted_SIR_9')
    p3 = calculate_percentage(sorted_SIR_9)
    
    # print('sorted_SIR_3_frac')
    p4 = calculate_percentage(sorted_SIR_3_frac)
    
    return p1,p2,p3,p4, sorted_SIR, sorted_SIR_3, sorted_SIR_9, sorted_SIR_frac, sorted_SIR_3_frac, sorted_SIR_9_frac

def calculate_throughput(effective_bandwidth, total_interference, snr_gap_dB, bandwidth):
    # Replace this with your actual function to calculate throughput
    # This could involve Shannon's Capacity Formula or any other appropriate model
    snr = snr_gap_dB + lineal_to_db(effective_bandwidth / total_interference)
    throughput = bandwidth * np.log2(1 + 10**(snr / 10))
    return throughput

def simulator_power_control_off(v, sigma_dB, num_samples, bandwidth, SNR_gap_dB, reuse_factor):
    centers = calculate_hexagon_centers(2)
    ref_cent = 0, 0
    list_of_throughput = []

    for i in range(1, num_samples + 1):
        throughput = 0
        d_all = 0
        d_all_3 = 0
        d_all_9 = 0
        center_id = 0

        for center in centers:
            c_x = center[0]
            c_y = center[1]
            sectors = get_sectors(c_x, c_y)
            random_points = get_random_points_in_sectors(sectors)

            if c_y == 0 and c_x == 0:
                x = db_to_lineal(generate_shadow_fading(sigma_dB))
                d = x / (calc_distance(random_points[0], ref_cent) ** (v))
            else:
                for j in range(0, 3):
                    angle = calculate_angle(ref_cent, random_points[j])

                    if angle <= 120 and angle >= 0:
                        x2 = db_to_lineal(generate_shadow_fading(sigma_dB))

                        if (calc_distance(random_points[j], center)) > 1:
                            print("error")

                        val = x2 / (calc_distance(random_points[j], ref_cent) ** (v))
                        d_all += val

                        if j == 0:
                            d_all_3 += val
                            if center_id == 8 or center_id == 18:
                                d_all_9 += val

            center_id += 1

        # Calculate throughput
        effective_bandwidth = bandwidth / reuse_factor  # Replace with actual reuse factor
        if(reuse_factor==3):
            d_all =  d_all_3   
        if(reuse_factor==9):
            d_all =  d_all_9       
        throughput = calculate_throughput(effective_bandwidth, d_all, SNR_gap_dB, bandwidth)
        list_of_throughput.append(throughput)

    sorted_throughput = np.sort(list_of_throughput)

    # Calculate average bitrate and bitrate attained by 97% of users
    average_bitrate = np.mean(sorted_throughput)
    bitrate_97 = np.percentile(sorted_throughput, 97)

    return average_bitrate, bitrate_97, sorted_throughput



    
def simulate_single(v, sigma_dB, n, num_samples, result_list):
    p1, p2, p3, p4, sorted_SIR, sorted_SIR_3, _, _, sorted_SIR_3_frac, _ = simulator(v, sigma_dB, n, num_samples)
    result_list.append((n, p4,sorted_SIR,sorted_SIR_3_frac))
 

def ex1(num_samples):
    v = 3.8
    sigma_dB = 8
    n= 0
    p1,p2,p3,p4, sorted_SIR, sorted_SIR_3, sorted_SIR_9, _, _, _ = simulator(v, sigma_dB, n,num_samples)
    
    print('F1, P(SIR >= -5dB): ', p1)
    print('F3, P(SIR >= -5dB): ', p2) 
    print('F9, P(SIR >= -5dB): ', p3)     
    
    #plot
    cumulative_prob = np.linspace(0, 1, len(sorted_SIR))
    cumulative_prob_3 = np.linspace(0, 1, len(sorted_SIR_3))
    cumulative_prob_9 = np.linspace(0, 1, len(sorted_SIR_9))
    
    plt.figure()
    plt.plot(sorted_SIR, cumulative_prob, label='CDF reuse factor 1', color='blue')
    plt.plot(sorted_SIR_3, cumulative_prob_3, label='CDF reuse factor 3', color='red')
    plt.plot(sorted_SIR_9, cumulative_prob_9, label='CDF reuse factor 9', color='green')
    plt.title('Cumulative Distribution Function (CDF) of Random Data')
    plt.xlabel('SIR(dB)')
    plt.ylabel('Cumulative Probability')

    plt.xlim(-20, 40)

    plt.grid(True)
    plt.legend()  
    #plt.show()   
    
def ex2(num_samples):
    v = 3.8
    sigma_dB = 8
    max_p4 = 0
    array___r = []
    
    results = []
    threads = []
    
    for i in range(1, 21):
        n = round(i * (1 / 20), 2)
        thread = threading.Thread(target=simulate_single, args=(v, sigma_dB, n, num_samples, results))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    for n_val, p4,sorted_SIR_3,sorted_SIR_3_frac in results:
        array___r.append((n_val, p4))
        if p4 > max_p4:
            max_p4 = p4
            max_n = n_val
            max_SIR = sorted_SIR_3
            max_SIR_frac = sorted_SIR_3_frac
    
    print('Max eta: ',max_n, ', P(SIR>=-5dB): ', max_p4)
    print(array___r)

    x_values, y_values = zip(*array___r)
    
    plt.figure()
    # Plotting the points
    plt.scatter(x_values, y_values, label='P(SIR>=-5dB)/Eta', color='blue', marker='o')

    # Adding labels and title
    plt.xlabel('Eta value')
    plt.ylabel('P(SIR>=-5dB)')
    plt.title('Scatter Plot of Points')

    # Adding a legend
    plt.legend()
    
    # Once max eta is found plot the CDFs SIRs
    '''
    p1,p2,p3,p4, sorted_SIR, sorted_SIR_3, sorted_SIR_9, sorted_SIR_frac, sorted_SIR_3_frac, sorted_SIR_9_frac = simulator(v,sigma_dB,max_n,num_samples)
    '''
    cumulative_prob_3 = np.linspace(0, 1, len(max_SIR))
    cumulative_prob_3_frac = np.linspace(0, 1, len(max_SIR_frac))
    
    plt.figure(4)
    plt.plot(max_SIR, cumulative_prob_3, label='CDF reuse factor 3', color='red')
    plt.plot(max_SIR_frac, cumulative_prob_3_frac, label='CDF reuse factor 3 fractional power', color='green')
    
    plt.title('Cumulative Distribution Function (CDF) of Random Data')
    plt.xlabel('SIR(dB)')
    plt.ylabel('Cumulative Probability')

    plt.xlim(-20, 40)

    plt.grid(True)
    plt.legend()
    
    return max_n
    
def ex3(num_samples,max_n_v3_8):
    v = 3.8
    sigma_dB = 8
    v_values = (3, 4.5)
    save_val = []
    
    p1,p2,p3,p4, sorted_SIR, sorted_SIR_3_v3_8, sorted_SIR_9, sorted_SIR_frac, sorted_SIR_3_frac_v3_8, sorted_SIR_9_frac = simulator(v, sigma_dB, max_n_v3_8,num_samples)
    

    for j in v_values:
        max_p4 = 0
        max_n = 0
        array___r = []

        results = []
        threads = []

        for i in range(1, 21):
            n = round(i * (1 / 20), 2)
            thread = threading.Thread(target=simulate_single, args=(j, sigma_dB, n, num_samples, results))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        for n_val, p4,sorted_SIR_3,sorted_SIR_3_frac in results:
            array___r.append((n_val, p4))
            if p4 > max_p4:
                max_p4 = p4
                max_n = n_val
                max_SIR = sorted_SIR_3
                max_SIR_frac = sorted_SIR_3_frac

        print('V value: ', j, ', Max eta: ', max_n, ', P(SIR>=-5dB): ', max_p4)
        save_val.append((max_SIR, max_SIR_frac))
    
    
    
    cumulative_prob_3_v3 = np.linspace(0, 1, len(save_val[0][0]))
    cumulative_prob_3_frac_v3 = np.linspace(0, 1, len(save_val[0][1]))
    cumulative_prob_3_v3_8 = np.linspace(0, 1, len(sorted_SIR_3_v3_8))
    cumulative_prob_3_frac_v3_8 = np.linspace(0, 1, len(sorted_SIR_3_frac_v3_8))
    cumulative_prob_3_v4_5 = np.linspace(0, 1, len(save_val[1][0]))
    cumulative_prob_3_frac_v4_5 = np.linspace(0, 1, len(save_val[1][1]))
    
    plt.figure()
    
    plt.plot(save_val[0][0], cumulative_prob_3_v3, label='CDF reuse factor 3 v=3', color='red')
    plt.plot(save_val[0][1], cumulative_prob_3_frac_v3, label='CDF reuse factor 3 fractional power v=3', color='green')
    plt.plot(sorted_SIR_3_v3_8, cumulative_prob_3_v3_8, label='CDF reuse factor 3 v=3.8', color='blue')
    plt.plot(sorted_SIR_3_frac_v3_8, cumulative_prob_3_frac_v3_8, label='CDF reuse factor 3 fractional power v=3.8', color='orange')
    plt.plot(save_val[1][0], cumulative_prob_3_v4_5, label='CDF reuse factor 3', color='purple')
    plt.plot(save_val[1][1], cumulative_prob_3_frac_v4_5, label='CDF reuse factor 3 fractional power v=4.5', color='brown')

    
    plt.title('Cumulative Distribution Function (CDF) of Random Data')
    plt.xlabel('SIR(dB)')
    plt.ylabel('Cumulative Probability')

    plt.xlim(-20, 40)

    plt.grid(True)
    plt.legend()
    
def ex4(num_samples):
    v = 3.8
    sigma_dB = 8
    b = 100e6  # Bandwidth in Hz
    SNR_gap_dB = 4

    reuse_factors = [1, 3, 9]
    plt.figure()

    for reuse_factor in reuse_factors:
        average_bitrate, bitrate_97, sorted_throughput = simulator_power_control_off(v, sigma_dB, num_samples, b, SNR_gap_dB, reuse_factor)

        print(f'Average bitrate for reuse factor {reuse_factor}: {round(average_bitrate/1e6,2)} Mbps')
        print(f'Bitrate attained by 97% of users for reuse factor {reuse_factor}: {round(bitrate_97/1e6,2)} Mbps')

        # Plot
        cumulative_prob = np.linspace(0, 1, len(sorted_throughput))
        plt.plot(sorted_throughput, cumulative_prob, label=f'CDF reuse factor {reuse_factor}')
        
    plt.title('Cumulative Distribution Function (CDF) of Throughput')
    plt.xlabel('Throughput (bps)')
    plt.ylabel('Cumulative Probability')
    plt.grid(True)
    plt.legend()

def act1(num_samples):
    print('-------------EX1-------------')
    start_time = time.time()
    ex1(num_samples)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Time for ex 1: ',elapsed_time)

def act2(num_samples):
    print('-------------EX2-------------')
    start_time = time.time()
    eta_for_3_8 = ex2(num_samples)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Time for ex 2: ',elapsed_time)

    return eta_for_3_8

def act3(num_samples, eta_for_3_8):
    print('-------------EX3-------------')
    start_time = time.time()
    ex3(num_samples, eta_for_3_8)
    end_time = time.time()    
    elapsed_time = end_time - start_time
    print('Time for ex 3: ',elapsed_time)

def act4(num_samples):
    print('-------------EX4-------------')
    start_time = time.time()
    ex4(num_samples)
    end_time = time.time()    
    elapsed_time = end_time - start_time
    print('Time for ex 4: ',elapsed_time)

def main():
    ###Values###
    layers = 2
    num_samples = 100
    ############
    
    plot_hexagons(layers)
    
   # act1(num_samples)
    
   # eta_for_3_8 = act2(num_samples)  

   # act3(num_samples, eta_for_3_8)

    act4(num_samples)

    #plots
    plt.show()
    

    
if __name__ == '__main__':
    sys.exit(main())
