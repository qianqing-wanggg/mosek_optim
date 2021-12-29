import numpy as np
#define nodes:
brick_length = 0.25
brick_height = 3.9/30
#brick_length = 0.2
#brick_height = 1
nodes = dict()
nodes = {
    1 : [0, brick_height],
    2 : [brick_length, brick_height],
    3 : [brick_length*2, brick_height],
    4 : [0, 0],
    5 : [brick_length, 0],
    6 : [brick_length*2, 0]
}

#define elements:
ro = 18.0
elems = dict()
elems = {
    (1,2,5,4): [0],#FX
    (2,3,6,5): [0]
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
    (4,5): [0, [1,0], [0,1]],#ground or not, t,n
    (2,5): [1, [0,1], [1,0]],
    (5,6): [0, [1,0], [0,1]]
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

#global A matrix
def is_subtuple_2(tupA, tupB):
    '''
    check if tuple A(two elements) is a sub tuple of tuple B
    '''
    for i in range(-1,len(tupB)-1):
        if (tupB[i],tupB[i+1]) == tupA:
            return True
    
    return False

Aglobal = np.zeros((3*len(elems), 2*4*len(conts)))
row = 0
for key_e, value_e in elems.items():
    col = 0
    for key_c, value_c in conts.items():
        #print(key_c)
        #print(key_e)
        
        #if key_c in key_e or key_c == key_e[-1]+key_e[0]:
        if is_subtuple_2(key_c, key_e) or key_c == (key_e[-1]+key_e[0]):
            Alocal_1 = Aelem_node(nodes[key_c[0]], value_e[-1], value_c[1], value_c[2])
            Alocal_2 = Aelem_node(nodes[key_c[1]], value_e[-1], value_c[1], value_c[2])
            Alocal_3 = Aelem_node(nodes[key_c[0]], value_e[-1], value_c[1], value_c[2])
            Alocal_4 = Aelem_node(nodes[key_c[1]], value_e[-1], value_c[1], value_c[2])
            Alocal = np.concatenate((Alocal_1, Alocal_2, Alocal_3, Alocal_4), axis = 1)
            Aglobal[row:row+3, col:col+8] = Alocal
        #elif key_c[1]+key_c[0] in key_e or key_c == key_e[0]+key_e[-1]:
        elif is_subtuple_2((key_c[1],key_c[0]), key_e) or key_c == key_e[0]+key_e[-1]:
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

#liveload
liveload = ro*brick_length*brick_height

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
                bkc.extend([mosek.boundkey.fx,
                        mosek.boundkey.fx,
                        mosek.boundkey.fx])
                blc.extend([0.0, -value[1], 0.0])
                buc.extend([0.0, -value[1], 0.0])

            # Bound keys and values for constraints -- contact failure condition
            for key, value in conts.items():
                for i in range(3*4):
                    bkc.append(mosek.boundkey.up)
                    blc.append(-inf)
                    buc.append(0.0)           

            # Bound keys for variables
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

            # Objective coefficients
            c = []
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
                
            asub.append([0])
            aval.append([liveload])#the live load is applied to the first element in the x direction


            numvar = len(bkx)
            numcon = len(bkc)

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

