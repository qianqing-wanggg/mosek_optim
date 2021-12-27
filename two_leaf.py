class ContactSurfaceBase():
    def __init__(self, R1, R2, t1, n1, mu):
        self.R1 = R1
        self.R2 = R2
        self.R3 = self.R1
        self.R4 = self.R2
        self.t1 = t1
        self.n1 = n1
        self.t4 = self.t3 = self.t2 = self.t1
        self.n4 = self.n3 = self.n2 = self.n1
        self.A1 = [[-self.t1[0], -self.t1[1], (self.R1[0]*self.t1[1]-self.R1[1]*self.t1[0])],
                    [-self.n1[0], -self.n1[1], (self.R1[0]*self.n1[1]-self.R1[1]*self.n1[0])]]
        self.A2 = [[-self.t2[0], -self.t2[1], (self.R2[0]*self.t2[1]-self.R2[1]*self.t2[0])],
                    [-self.n2[0], -self.n2[1], (self.R2[0]*self.n2[1]-self.R2[1]*self.n2[0])]]
        self.A3 = [[-self.t3[0], -self.t3[1], (self.R3[0]*self.t3[1]-self.R3[1]*self.t3[0])],
                    [-self.n3[0], -self.n3[1], (self.R3[0]*self.n3[1]-self.R3[1]*self.n3[0])]]
        self.A4 = [[-self.t4[0], -self.t4[1], (self.R4[0]*self.t4[1]-self.R4[1]*self.t4[0])],
                    [-self.n4[0], -self.n4[1], (self.R4[0]*self.n4[1]-self.R4[1]*self.n4[0])]]
        self.y = [[1, -1, 0],
                    [-mu, -mu, -1]]
    def printinfo(self):
        print(self.A1)
        print(self.A2)
        print(self.A3)
        print(self.A4)

class ContactSurface():
    def __init__(self, R1, R2, t1, n1, mu):
        self.R1 = R1
        self.R2 = R2
        self.R3 = self.R1
        self.R4 = self.R2
        self.t1 = t1
        self.n1 = n1
        self.t3 = -self.t1
        self.n3 = -self.n1
        self.t2 = self.t1
        self.n2 = self.n1
        self.t4 = -self.t2
        self.n4 = -self.n2

        self.A1 = [[-self.t1[0], -self.t1[1], (self.R1[0]*self.t1[1]-self.R1[1]*self.t1[0])],
                    [-self.n1[0], -self.n1[1], (self.R1[0]*self.n1[1]-self.R1[1]*self.n1[0])]]
        self.A2 = [[-self.t2[0], -self.t2[1], (self.R2[0]*self.t2[1]-self.R2[1]*self.t2[0])],
                    [-self.n2[0], -self.n2[1], (self.R2[0]*self.n2[1]-self.R2[1]*self.n2[0])]]
        self.A3 = [[-self.t3[0], -self.t3[1], (self.R3[0]*self.t3[1]-self.R3[1]*self.t3[0])],
                    [-self.n3[0], -self.n3[1], (self.R3[0]*self.n3[1]-self.R3[1]*self.n3[0])]]
        self.A4 = [[-self.t4[0], -self.t4[1], (self.R4[0]*self.t4[1]-self.R4[1]*self.t4[0])],
                    [-self.n4[0], -self.n4[1], (self.R4[0]*self.n4[1]-self.R4[1]*self.n4[0])]]
        self.y = [[1, -1, 0],
                    [-mu, -mu, -1]]

class BrickLeft():
    def __init__(self, length, height, mu):
        brick_length = length
        brick_height = height

        #brick 1
        brick_center = [brick_length/2, brick_height/2] 
        ##contact surface 1
        R1 = [-brick_length/2, -brick_height/2]
        R2 = [brick_length/2, -brick_height/2]
        t1 = [1,0]
        n1 = [0,1]
        self.consur1 = ContactSurface(R1, R2, t1, n1, mu)
        ##contact surface 2
        R1 = [brick_length/2, -brick_height/2]
        R2 = [brick_length/2, brick_height/2]
        t1 = [0,1]
        n1 = [-1,0]
        self.consur2 = ContactSurface(R1, R2, t1, n1, mu)
        ##contact surface 3
        R1 = [-brick_length/2, brick_height/2]
        R2 = [brick_length/2, brick_height/2]
        t1 = [1,0]
        n1 = [0,-1]
        self.consur3 = ContactSurface(R1, R2, t1, n1, mu)
class BrickRight():
    def __init__(self, length, height, mu):
        brick_length = 0.25
        brick_height = 3.9/30
        #brick 2
        brick_center = [brick_length*(1+0.5), brick_height/2] 
        ##contact surface 1
        R1 = [-brick_length/2, -brick_height/2]
        R2 = [brick_length/2, -brick_height/2]
        t1 = [1,0]
        n1 = [0,1]
        self.consur1 = ContactSurface(R1, R2, t1, n1, mu)
        ##contact surface 2
        R1 = [-brick_length/2, -brick_height/2]
        R2 = [-brick_length/2, brick_height/2]
        t1 = [0,-1]
        n1 = [1,0]
        self.consur2 = ContactSurface(R1, R2, t1, n1, mu)
        ##contact surface 3
        R1 = [-brick_length/2, brick_height/2]
        R2 = [brick_length/2, brick_height/2]
        t1 = [1,0]
        n1 = [0,-1]
        self.consur3 = ContactSurface(R1, R2, t1, n1, mu)

# brick_length = 0.25
# brick_height = 3.9/30
# nodes = dict{}
# for node_number in range(4):
#     nodes.update({node_number: [[1,0],[0,1],[-brick_length*pow(-1,node_number%2)/2, -brick_height/2],0,node_number]})#t,n,R,A_row_brick, A_col_brick
import numpy as np
#define nodes:
brick_length = 0.25
brick_height = 3.9/30
#brick_length = 0.2
#brick_height = 1
nodes = dict()
nodes = {
    'A' : [0, brick_height],
    'B' : [brick_length, brick_height],
    'C' : [brick_length*2, brick_height],
    'D' : [0, 0],
    'E' : [brick_length, 0],
    'F' : [brick_length*2, 0]
}

#define elements:
ro = 18.0
elems = dict()
elems = {
    'ABED': [0],#FX
    'BCFE': [0]
}
# elems = {
#     'ABED': [0]
# }

for key, value in elems.items():
    length = np.linalg.norm(np.array(nodes[key[0]])-np.array(nodes[key[1]]))
    height = np.linalg.norm(np.array(nodes[key[1]])-np.array(nodes[key[2]]))
    mass = ro*length*height
    value.append(mass)
    center = 0.5*(np.array(nodes[key[0]])+np.array(nodes[key[2]]))
    value.append(center)

#define contacts:
mu = 0.58
#mu = 0.7
conts = dict()
conts = {
    'DE': [0, [1,0], [0,1]],#ground or not, t,n
    'BE': [1, [0,1], [1,0]],
    'EF': [0, [1,0], [0,1]]
}
# conts = {
#     'DE': [0, [1,0], [0,1]]
# }

def Aelem_node(node, elem_center, t, n, reverse = False):
    t = np.array(t)
    n = np.array(n)
    if reverse:
        t = t*(-1)
        n = n*(-1)
    R = np.array(node) - np.array(elem_center)
    #print(R)
    Alocal = np.matrix([
       [t[0], n[0]],
       [t[1], n[1]],
       [-1*float(np.cross(R,t)), -1*float(np.cross(R,n))]
    ])
    #print(float(np.cross(R,t)))
    return Alocal

# #define A matrix for each element-cont pair
# elemAM = dict()

# for key, value in elems.items():
#     for i in range(-1,len(key)-1):
#         if key[i]+key[i+1] in conts:
#             Alocal_1 = Aelem_node(nodes[key[i]], value[-1], conts[key[i]+key[i+1]][1], conts[key[i]+key[i+1]][2])
#             Alocal_2 = Aelem_node(nodes[key[i+1]], value[-1], conts[key[i]+key[i+1]][1], conts[key[i]+key[i+1]][2])
#             Alocal_3 = Aelem_node(nodes[key[i]], value[-1], conts[key[i]+key[i+1]][1], conts[key[i]+key[i+1]][2])
#             Alocal_4 = Aelem_node(nodes[key[i+1]], value[-1], conts[key[i]+key[i+1]][1], conts[key[i]+key[i+1]][2])
#             Alocal = np.concatenate(Alocal_1, Alocal_2, Alocal_3, Alocal_4, axis = 1)
#             elemAM.update(key+key[i]+key[i+1]: Alocal)
#         if key[i+1]+key[i] in conts:
#             #t, n reverse
#             Alocal_1 = Aelem_node(nodes[key[i]], value[-1], conts[key[i]+key[i+1]][1], conts[key[i]+key[i+1]][2], reverse = True)
#             Alocal_2 = Aelem_node(nodes[key[i+1]], value[-1], conts[key[i]+key[i+1]][1], conts[key[i]+key[i+1]][2], reverse = True)
#             Alocal_3 = Aelem_node(nodes[key[i]], value[-1], conts[key[i]+key[i+1]][1], conts[key[i]+key[i+1]][2], reverse = True)
#             Alocal_4 = Aelem_node(nodes[key[i+1]], value[-1], conts[key[i]+key[i+1]][1], conts[key[i]+key[i+1]][2], reverse = True)
#             Alocal = np.concatenate(Alocal_2, Alocal_1, Alocal_4, Alocal_3, axis = 1)
#             elemAM.update(key+key[i+1]+key[i]: Alocal)

#global A matrix
Aglobal = np.zeros((3*len(elems), 2*4*len(conts)))
row = 0
for key_e, value_e in elems.items():
    col = 0
    for key_c, value_c in conts.items():
        #print(key_c)
        #print(key_e)
        
        if key_c in key_e:
            Alocal_1 = Aelem_node(nodes[key_c[0]], value_e[-1], value_c[1], value_c[2])
            Alocal_2 = Aelem_node(nodes[key_c[1]], value_e[-1], value_c[1], value_c[2])
            Alocal_3 = Aelem_node(nodes[key_c[0]], value_e[-1], value_c[1], value_c[2])
            Alocal_4 = Aelem_node(nodes[key_c[1]], value_e[-1], value_c[1], value_c[2])
            Alocal = np.concatenate((Alocal_1, Alocal_2, Alocal_3, Alocal_4), axis = 1)
            Aglobal[row:row+3, col:col+8] = Alocal
        elif key_c[1]+key_c[0] in key_e or key_c == key_e[0]+key_e[-1]:
            Alocal_1 = Aelem_node(nodes[key_c[0]], value_e[-1], value_c[1], value_c[2], reverse = True)
            Alocal_2 = Aelem_node(nodes[key_c[1]], value_e[-1], value_c[1], value_c[2], reverse = True)
            Alocal_3 = Aelem_node(nodes[key_c[0]], value_e[-1], value_c[1], value_c[2], reverse = True)
            Alocal_4 = Aelem_node(nodes[key_c[1]], value_e[-1], value_c[1], value_c[2], reverse = True)
            Alocal = np.concatenate((Alocal_1, Alocal_2, Alocal_3, Alocal_4), axis = 1)
            Aglobal[row:row+3, col:col+8] = Alocal
        else:
            Aglobal[row:row+3, col:col+8] = np.zeros((3,8))
        col+=8
    row+=3

#print(Aglobal)

#global Y matrix
Y = np.zeros((3*4*len(conts), 2*4*len(conts)))
yunit = np.matrix([
    [1, -mu],
    [-1, -mu],
    [0,-1]
])
index = 0
for key, value in conts.items():
    for i in range(4):
        Y[index*3:index*3+3, index*2:index*2+2] = yunit
        index+=1
#print(Y)

# brick_length = 0.25
# brick_height = 3.9/30
# # brick_length = 0.2
# # brick_height = 1
# mu = 0.58
# ro = 18.0
weight = ro*brick_length*brick_height
#weight = 3.6
# #weight = 3.6
# class BrickRightBottom():
#     '''
#     for element with right and bottom contact
#     '''
#     ##contact surface 1
#     R1 = [-brick_length/2, -brick_height/2]
#     R2 = [brick_length/2, -brick_height/2]
#     t1 = [1,0]
#     n1 = [0,1]
#     consur_bottom = ContactSurfaceBase(R1, R2, t1, n1, mu)
#     ##contact surface 2
#     R1 = [brick_length/2, -brick_height/2]
#     R2 = [brick_length/2, brick_height/2]
#     t1 = [0,1]
#     n1 = [-1,0]
#     consur_right = ContactSurface(R1, R2, t1, n1, mu)

# class BrickLeftBottom():
#     '''
#     for element with left and bottom contact
#     '''
#     ##contact surface bottom
#     R1 = [-brick_length/2, -brick_height/2]
#     R2 = [brick_length/2, -brick_height/2]
#     t1 = [1,0]
#     n1 = [0,1]
#     consur_bottom = ContactSurface(R1, R2, t1, n1, mu)
#     ##contact surface left
#     R1 = [-brick_length/2, -brick_height/2]
#     R2 = [-brick_length/2, brick_height/2]
#     t1 = [0,-1]
#     n1 = [1,0]
#     consur_left = ContactSurface(R1, R2, t1, n1, mu)

# brick0 = BrickRightBottom()
import sys
import mosek
import math

# Since the value of infinity is ignored, we define it solely
# for symbolic purposes
inf = 0.0

# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def main():
    # Make mosek environment
    with mosek.Env() as env:
        # Create a task object
        with env.Task(0, 0) as task:
            # Attach a log stream printer to the task
            task.set_Stream(mosek.streamtype.log, streamprinter)

            # Bound keys and values for constraints -- force equilibrium
            bkc = []
            blc = []
            buc = []
            for key, value in elems.items():
                bkc.extend([mosek.boundkey.fx, #-x1-x3-x5-x7-3.6x9 = 0
                        mosek.boundkey.fx, #-x2-x4-x6-x8 = -3.6
                        mosek.boundkey.fx]) #0.5x1-0.1x2+0.5x3+0.1x4+0.5x5-0.1x6+0.5x7+0.1x8=0
                blc.extend([0.0, -value[1], 0.0])
                buc.extend([0.0, -value[1], 0.0])
                #    mosek.boundkey.up, #x1-0.7x2<=0
                #    mosek.boundkey.up, #-x1-0.7x2<=0
                #    mosek.boundkey.up, #-x2<=0
                #    mosek.boundkey.up, #x3-0.7x4<=0
                #    mosek.boundkey.up, #-x3-0.7x4<=0
                #    mosek.boundkey.up, #-x4<=0
                #    mosek.boundkey.up, #x5-0.7x6<=0
                #    mosek.boundkey.up, #-x5-0.7x6<=0
                #    mosek.boundkey.up, #-x6<=0
                #    mosek.boundkey.up, #x7-0.7x8<=0
                #    mosek.boundkey.up, #-x7-0.7x8<=0
                #    mosek.boundkey.up] #-x8<=0

            # # Bound values for constraints
            # blc = [0.0, -weight, 0.0]
            # buc = [0.0, -weight, 0.0]
            for key, value in conts.items():
                for i in range(3*4):
                    bkc.append(mosek.boundkey.up)
                    blc.append(-inf)
                    buc.append(0.0)
            #for i in range(3*4*4):#3degrees * 4 nodes * 2 contact faces
                

            # # Bound keys for variables
            bkx = []
            blx = []
            bux = []
            for key, value in conts.items():
                for i in range(2*4):
                    bkx.append(mosek.boundkey.fr)
                    blx.append(-inf)
                    bux.append(+inf)
            bkx.append(mosek.boundkey.fr)
            blx.append(-inf)
            bux.append(+inf)
            # for i in range(2*4*2+1):
            #     bkx.append(mosek.boundkey.fr)
            #     blx.append(-inf)
            #     bux.append(+inf)

            # # Bound values for variables
            # blx = [-inf, -inf,-inf, -inf, -inf, -inf,-inf, -inf, -inf]
            # bux = [+inf, +inf, +inf, +inf, +inf, +inf, +inf, +inf, +inf]

            c = []
            # Objective coefficients
            for key, value in conts.items():
                for i in range(2*4):#2variables(t,n)*4nodes*2contact faces
                    c.append(0.0)
            c.append(1.0)

            # Below is the sparse representation of the A
            # matrix stored by column.
            asub = []
            aval = []
            for col in range(len(conts)*2*4):
                col_index = []
                col_value = []
                for row in range(len(elems)*3):
                    if Aglobal[row][col] != 0:
                        col_index.append(row)
                        col_value.append(Aglobal[row][col])
                for row in range(3*4*len(conts)):
                    if Y[row][col] != 0:
                        col_index.append(row+len(elems)*3)
                        col_value.append(Y[row][col])
                asub.append(col_index)
                aval.append(col_value)
            #for col in range(len(conts)*2*4):
                
            asub.append([0])
            aval.append([weight])
            print(asub)
            print(aval)


            # aval = [
            #         brick0.consur_bottom.A1[0] + brick0.consur_bottom.y[0],
            #         brick0.consur_bottom.A1[1] + brick0.consur_bottom.y[1],
            #         brick0.consur_bottom.A2[0] + brick0.consur_bottom.y[0],
            #         brick0.consur_bottom.A2[1] + brick0.consur_bottom.y[1],
            #         brick0.consur_bottom.A3[0] + brick0.consur_bottom.y[0],
            #         brick0.consur_bottom.A3[1] + brick0.consur_bottom.y[1],
            #         brick0.consur_bottom.A4[0] + brick0.consur_bottom.y[0],
            #         brick0.consur_bottom.A4[1] + brick0.consur_bottom.y[1],
            #         brick0.consur_right.A1[0] + brick0.consur_right.y[0],
            #         brick0.consur_right.A1[1] + brick0.consur_right.y[1],
            #         brick0.consur_right.A2[0] + brick0.consur_right.y[0],
            #         brick0.consur_right.A2[1] + brick0.consur_right.y[1],
            #         brick0.consur_right.A3[0] + brick0.consur_right.y[0],
            #         brick0.consur_right.A3[1] + brick0.consur_right.y[1],
            #         brick0.consur_right.A4[0] + brick0.consur_right.y[0],
            #         brick0.consur_right.A4[1] + brick0.consur_right.y[1],
            #         [-weight]
            #         ]

            numvar = len(bkx)
            numcon = len(bkc)
            print(len(asub))
            print(len(aval))

            # Append 'numcon' empty constraints.
            # The constraints will initially have no bounds.
            task.appendcons(numcon)

            # Append 'numvar' variables.
            # The variables will initially be fixed at zero (x=0).
            task.appendvars(numvar)

            for j in range(numvar):
                # Set the linear term c_j in the objective.
                task.putcj(j, c[j])

                # Set the bounds on variable j
                # blx[j] <= x_j <= bux[j]
                task.putvarbound(j, bkx[j], blx[j], bux[j])

                # Input column j of A
                print(asub[j])
                print(aval[j])
                task.putacol(j,                  # Variable (column) index.
                             asub[j],            # Row index of non-zeros in column j.
                             aval[j])            # Non-zero Values of column j.

            # Set the bounds on constraints.
             # blc[i] <= constraint_i <= buc[i]
            for i in range(numcon):
                task.putconbound(i, bkc[i], blc[i], buc[i])

            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.maximize)

            # Solve the problem
            task.writedata("data.opf")
            task.optimize()
            # Print a summary containing information
            # about the solution for debugging purposes
            task.solutionsummary(mosek.streamtype.msg)

            # Get status information about the solution
            solsta = task.getsolsta(mosek.soltype.bas)

            if (solsta == mosek.solsta.optimal or
                    solsta == mosek.solsta.near_optimal):
                xx = [0.] * numvar
                task.getxx(mosek.soltype.bas, # Request the basic solution.
                           xx)
                print("Optimal solution: ")
                for i in range(numvar):
                    print("x[" + str(i) + "]=" + str(xx[i]))
            elif (solsta == mosek.solsta.dual_infeas_cer or
                  solsta == mosek.solsta.prim_infeas_cer or
                  solsta == mosek.solsta.near_dual_infeas_cer or
                  solsta == mosek.solsta.near_prim_infeas_cer):
                print("Primal or dual infeasibility certificate found.\n")
            elif solsta == mosek.solsta.unknown:
                print("Unknown solution status")
            else:
                print("Other solution status")

# call the main function
try:
    main()
except mosek.Error as e:
    print("ERROR: %s" % str(e.errno))
    if e.msg is not None:
        print("\t%s" % e.msg)
        sys.exit(1)
except:
    import traceback
    traceback.print_exc()
    sys.exit(1)

