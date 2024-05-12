#======================================================================
#this script is used to find the transformation coefficients between the camera and the world coordinate system

#@@@@@ THE CAMERA COORDINATE SYSTEM IS DEFINED AS FOLLOWS @@@@@#
#the origin is the center of the camera lens (middle of the frame in 2D)

#horizontal view angle = the angle between the left and right edges of the frame
#vertical view angle = the angle between the top and bottom edges of the frame

#pixel x angle = it is the horizontal angle between the pixel and the center of the frame (zero at the middle of the frame in 2D, incerase to the right)
#pixel y angle = it is the vertical angle between the pixel and the center of the frame (zero at the middle of the frame in 2D, incerase to the up)
#the angles are normalized to [-0.5, 0.5 ] and calculated as follows:
# x_angle = ( (pixel_x_coordinate / frame width) -0.5 )*horizontal_view_angle
# y_angle = ( 0.5 - (pixel_y_coordinate / frame height) )*vertical_view_angle

#Lets say camera base vector is V = [V1, V2, V3] where V1, V2, V3 are the orthagonal (i.e. Vi*Vj^T=0) unit vectors of the camera coordinate system
#The unit vector (V = a1*V1 + a2*V2+ a3*V3) = P_c(a1,a2,a3) pointing from the camera origin to the pixel is calculated as follows:

# lenght = 1:
# a1*|V1| = a1 = 1*cos(angle_y)*sin(angle_x)
# a2*|V2| = a2 = 1*sin(angle_y)
# a3*|V3| = a3 = 1*cos(angle_y)*cos(angle_x)

#Note that angles are bounded to [-pi/2, pi/2] thus a3 is always positive (i.e. the pixel is always in front of the camera)
#Then according to the camera coordinate system, the unit vector pointing from the camera origin to the pixel is:
# V = [a1, a2, a3] = [cos(angle_y)*sin(angle_x), sin(angle_y), cos(angle_y)*cos(angle_x)]

#@@@@@ CHANGING THE BASE OF THE CAMERA COORDINATE FRAME TO WORLD COORDINATE FRAME @@@@@#
#the v1 axis is the horizontal axis of the camera. (zero at the middle of the frame in 2D, incerase to the right)
#the v2 axis is the vertical axis of the camera. (zero at the middle of the frame in 2D, incerase to the up)
#the v3 axis is the axis that is perpendicular to the camera lens. (zero at the middle of the frame in 2D, incerase to the front)

#To express the unit vector in the world coordinate system, we need to change the base of the camera coordinate system to the world coordinate system

#Lets say the a base vector B = [B1, B2, B3] where B1, B2, B3 are the orthagonal (i.e. Bi*Bj^T=0) unit vectors is oriented same as the world coordinate system and 
#located at the same place as camera coordinate system in space

#Lets say P is a point in the space
# P = a1*V1+ a2*V2+ a3*V3 = b1*B1+ b2*B2+ b3*B3

#We can express each base vector of V in terms of the base vectors of B as follows:
#V1 = α11*B1 + α12*B2 + α13*B3
#V2 = α21*B1 + α22*B2 + α23*B3
#V3 = α31*B1 + α32*B2 + α33*B3

#Then we can express P in terms of the base vectors of B as follows:
#P = a1*(α11*B1 + α12*B2 + α13*B3) + a2*(α21*B1 + α22*B2 + α23*B3) + a3*(α31*B1 + α32*B2 + α33*B3)

#|V1|   | α11 α12 α13 |   |B1|
#|V2| = | α21 α22 α23 | * |B2|
#|V3|   | α31 α32 α33 |   |B3|

#       |_____________|
#        A_rotate = A_r
#
#       V = A_r * B

# A_rotate : rotation matrix that rotates the base B to the base V

# now lets say the world coordinate system is noted as W with base vectors W1, W2, W3

# Vbp : vector from the camera origin to the point p in the camera coordinate system in base B
# Vwp : vector from the world coordinate system to the point p in base W  
# T : (translation) vector from the world coordinate system in base W to the camera center

# T = [T1, T2, T3] : translation vector from the world coordinate system to the camera coordinate system in base W
# T + Vbp = Vwp  -> Vbp = Vwp - T

# Then V = A_r * B = A_r * (Vwp - T) = A_r * Vwp - A_r * T
# Note that A_r * T is a constant vector and it is the same for all points in the space

#===================
# V = A_r *Vwp + C |
#===================
# Even though V is an unit vector, This transformation transforms any point from camera coordinate system to the world coordinate system

#@@@@@ FINDING THE UNKNOWNS PRACTICALLY @@@@@#

# V = A_r *Vwp + C

# Unknowns are: 
# α11, α12, α13, α21, α22, α23, α31, α32, α33, c1, c2 ,c3 (12 unknowns)

# We need at least 4 points to find the unknowns each point gives us 3 equations, can go more than 4 points to increase accuracy since it will be approximated solution (LEAST-SQUARES)
# Lets say we have a measurement of a point P in the world coordinate system and camera coordinate system

# P = Pv(v1,v2,v3) = Pb(b1,b2,b3) = Pw(w1,w2,w3)
# Since each base vectors are orthagonal to each other, we can write the following equations:

#  v1 = w1*α11 + w2*α12 + w3*α13 + c1
#  v2 = w1*α21 + w2*α22 + w3*α23 + c2
#  v3 = w1*α31 + w2*α32 + w3*α33 + c3

# note that vx,bx,wx are indicating how many of respective vectors are used

# PROCEDURE: Using the detection data and the world data, collect points so that you have ( v1_i, v2_i, v3_i, w1_i, w2_i, w3_i )
# Then you can write the following equation for each point:

# α11   α12   α13   α21   α22   α23   α31   α32   α33   c1   c2   c3 | V 
# w1_i  w2_i  w3_i  0     0     0     0     0     0     1    0    0  | v1_i
# 0     0     0     w1_i  w2_i  w3_i  0     0     0     0    1    0  | v2_i
# 0     0     0     0     0     0     w1_i  w2_i  w3_i  0    0    1  | v3_i

# Then you can stack all the equations and solve the following equation:
# A * X = B 
# using least squares method which will give you the unknowns

def calculate_transformation_coefficients():
    import numpy as np
    import pprint

    number_of_points = 0
    row_matrix = None
    result_matrix = None

    while True:
        camera_coordinates = input(f"point {number_of_points+1}: Enter the camera coordinates of the point as -> v1,v2,v3: ")
        world_coordinates = input(f"point {number_of_points+1}: Enter the world coordinates of the point as -> w1,w2,w3: ")
        camera_coordinates = np.array(camera_coordinates.split(","), dtype=np.float32)
        world_coordinates = np.array(world_coordinates.split(","), dtype=np.float32)

        new_rows = np.array(
            [
            [world_coordinates[0], world_coordinates[1], world_coordinates[2], 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, world_coordinates[0], world_coordinates[1], world_coordinates[2], 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, world_coordinates[0], world_coordinates[1], world_coordinates[2], 0, 0, 1]
            ]
            )
        new_results = np.array(
            [
            [camera_coordinates[0]], 
            [camera_coordinates[1]],
            [camera_coordinates[2]]
            ])

        if row_matrix is None:
            row_matrix = new_rows
            result_matrix = new_results
        else:
            row_matrix = np.vstack((row_matrix, new_rows))
            result_matrix = np.vstack((result_matrix, new_results))

        number_of_points +=1

        pprint.pprint("ROW MATRIX:")
        pprint.pprint(row_matrix)
        pprint.pprint("RESULT MATRIX:")
        pprint.pprint(result_matrix)

        if number_of_points >= 4:
            continue_input = input("Do you want to continue? (y/n): ")
            if continue_input != "y":

                #solve the equation using least squares method
                # A * X = B                
                # X = [α11, α12, α13, α21, α22, α23, α31, α32, α33, c1, c2, c3]

                # Using numpy's lstsq function to solve for x
                x, residuals, rank, s = np.linalg.lstsq(row_matrix, result_matrix, rcond=None)
                row_names = ["alpha_11", "alpha_12", "alpha_13", "alpha_21", "alpha_22", "alpha_23", "alpha_31", "alpha_32", "alpha_33", "c1", "c2", "c3"]
                for row_name, row in zip(row_names, x):
                    print(f"{row_name:<15}: {row[0]:.4f}")


                A_matrix_pretty = f"A = np.array([\n\t[{x[0][0]:.4f}, {x[1][0]:.4f},{x[2][0]:.4f}],\n\t[{x[3][0]:.4f}, {x[4][0]:.4f},{x[5][0]:.4f}],\n\t[{x[6][0]:.4f}, {x[7][0]:.4f},{x[8][0]:.4f}]\n])"

                B_matrix_pretty = f"B = np.array([\n\t[{x[9][0]:.4f}],\n\t[{x[10][0]:.4f}],\n\t[{x[11][0]:.4f}]\n])"

                print(A_matrix_pretty)
                print(B_matrix_pretty)

                break
          










