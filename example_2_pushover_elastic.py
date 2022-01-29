import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
import math

nb_course = 10
nb_unit = 10
unit_length = 2
unit_height = 1.75
unit_weigh = 10
mass_unit = unit_weigh*unit_length*unit_height
print_detail = False
kn = 1e5
poisson_ratio = 0.2
kt = kn/(2*(1+poisson_ratio))
Cn = 1/kn
Ct = 1/kt

nodes = dict()
for i in range(0,nb_course+1):
    for col in range(1, nb_unit+2):
        nodes[(nb_unit+1)*i+col] = [(col-1)*unit_length, (nb_course-i)*unit_height]

#define elements:
elems = dict()#Fx
for j in range(int(nb_course/2)):
    for i in range(5):
        elems[(2*i+1+22*j, 2*i+2+22*j, 2*i+3+22*j, 2*i+14+22*j, 2*i+13+22*j, 2*i+12+22*j, 2*i+1+22*j)] = [2]
    elems[(12+22*j,13+22*j,24+22*j,23+22*j,12+22*j)] = [1]
    for i in range(4):
        elems[(2*i+13+22*j, 2*i+14+22*j, 2*i+15+22*j, 2*i+26+22*j, 2*i+25+22*j, 2*i+24+22*j, 2*i+13+22*j)] = [2]
    elems[(21+22*j,22+22*j,33+22*j,32+22*j,21+22*j)] = [1]

for key, value in elems.items():
    if len(key) == 5:
        #length = np.linalg.norm(np.array(nodes[key[0]])-np.array(nodes[key[1]]))
        #height = np.linalg.norm(np.array(nodes[key[1]])-np.array(nodes[key[2]]))
        mass = mass_unit
        value.append(mass)
        center = 0.5*(np.array(nodes[key[0]])+np.array(nodes[key[2]]))
        value.append(center)
    else:
        mass = 2*mass_unit
        value.append(mass)
        center = 0.5*(np.array(nodes[key[0]])+np.array(nodes[key[3]]))
        value.append(center)

#define contacts:
mu = 0.65
#mu = 0.7
conts = dict()#ground or not, t,n
for j in range(int(nb_course/2)):
    for i in range(4):
        conts[(2*i+14+22*j, 2*i+3+22*j)] = [1, [0,-1], [1,0]]
    for i in range(5):
        conts[(2*i+13+22*j, 2*i+12+22*j)] = [0, [1,0], [0,1]]
        conts[(2*i+14+22*j, 2*i+13+22*j)] = [0, [1,0], [0,1]]
    for i in range(5):
        conts[(2*i+24+22*j, 2*i+13+22*j)] = [1, [0,-1], [1,0]]
    for i in range(5):
        conts[(2*i+24+22*j, 2*i+23+22*j)] = [0, [1,0], [0,1]]
        conts[(2*i+25+22*j, 2*i+24+22*j)] = [0, [1,0], [0,1]]



def plot_elements(elems, title = 'initial elements'):
    lines = []
    d = 0
    for key, value in elems.items():
        boundary_points = []
        for k in key:
            boundary_points.append((nodes[k][0], nodes[k][1]))
        for i in range(len(boundary_points)-1):
            lines.append([boundary_points[i], boundary_points[i+1]])
        d+=1
    lc = mc.LineCollection(lines, linewidths=2)
    fig, ax = plt.subplots()
    ax.set_xlim(-0.2*unit_length*nb_unit, 1.2*unit_length*nb_unit)
    ax.set_ylim(-0.2*unit_height*nb_course, 1.2*unit_height*nb_course)
    ax.add_collection(lc)
    ax.set_title(title)
    plt.show()

def plot_conts(conts, title = 'initial contacts'):
    lines = []
    for key, value in conts.items():
        boundary_points = []
        #translate all the boundary point of the element
        for k in key:
            boundary_points.append((nodes[k][0], nodes[k][1]))
        for i in range(len(boundary_points)-1):
            lines.append([boundary_points[i], boundary_points[i+1]])
    lc = mc.LineCollection(lines, linewidths=2)
    fig, ax = plt.subplots()
    ax.set_xlim(-0.2*unit_length*nb_unit, 1.2*unit_length*nb_unit)
    ax.set_ylim(-0.2*unit_height*nb_course, 1.2*unit_height*nb_course)
    ax.add_collection(lc)
    ax.set_title(title)
    plt.show()
plot_conts(conts)

def normalize_disp_factor(elems, disps = []):
    d = 0
    for key, value in elems.items():
        center = value[-1]
        trans_x = disps[3*d]
        trans_y = disps[3*d+1]
        rot = disps[3*d+2]
        boundary_points = []
        #translate all the boundary point of the element
        for k in key:
            node_x = nodes[k][0]-center[0]
            node_y = nodes[k][1]-center[1]
            new_x = node_x*math.cos(rot)-node_y*math.sin(rot)+trans_x+center[0]
            new_y = node_x*math.sin(rot)+node_y*math.cos(rot)+trans_y+center[1]
            if k == node_crutial:
                return (new_x - node_x - center[0])/disp_incre
        d+=1

def get_node_disp(elems, disps = []):
    d = 0
    for key, value in elems.items():
        center = value[-1]
        trans_x = disps[3*d]
        trans_y = disps[3*d+1]
        rot = disps[3*d+2]
        boundary_points = []
        #translate all the boundary point of the element
        for k in key:
            node_x = nodes[k][0]-center[0]
            node_y = nodes[k][1]-center[1]
            new_x = node_x*math.cos(rot)-node_y*math.sin(rot)+trans_x+center[0]
            new_y = node_x*math.sin(rot)+node_y*math.cos(rot)+trans_y+center[1]
            if k == node_crutial:
                return new_x - node_x - center[0]
        d+=1

def update_node_coor(elems, disps=[]):
    if not disps:
        disps = [0] * len(elems)*3
    d = 0
    for key, value in elems.items():
        center = value[-1]
        trans_x = disps[3*d]
        trans_y = disps[3*d+1]
        rot = disps[3*d+2]
        boundary_points = []
        #translate all the boundary point of the element
        node_index = 0
        for k in key:
            if node_index == len(key)-1:
                break#not reupdate the first node
            node_x = nodes[k][0]-center[0]
            node_y = nodes[k][1]-center[1]
            new_x = node_x*math.cos(rot)-node_y*math.sin(rot)+trans_x+center[0]
            new_y = node_x*math.sin(rot)+node_y*math.cos(rot)+trans_y+center[1]
            nodes[k] = [new_x,new_y]
            node_index+=1
            #print(f"the node {k}, originally at {nodes[k][0]}, {nodes[k][1]},  is updated to {new_x}, {new_y}")
        d+=1


def init_elem_center_mass(elems):
    for key, value in elems.items():
        if len(key) == 5:
            #length = np.linalg.norm(np.array(nodes[key[0]])-np.array(nodes[key[1]]))
            #height = np.linalg.norm(np.array(nodes[key[1]])-np.array(nodes[key[2]]))
            mass = mass_unit
            value.append(mass)
            center = 0.5*(np.array(nodes[key[0]])+np.array(nodes[key[2]]))
            value.append(center)
        else:
            mass = 2*mass_unit
            value.append(mass)
            center = 0.5*(np.array(nodes[key[0]])+np.array(nodes[key[4]]))
            value.append(center)

def cal_elem_center(elems):
    for key, value in elems.items():
        if len(key) == 5:
            center = 0.5*(np.array(nodes[key[0]])+np.array(nodes[key[2]]))
            value[2] = center
        else:
            center = 0.5*(np.array(nodes[key[0]])+np.array(nodes[key[4]]))
            value[2] = center
init_elem_center_mass(elems)
plot_elements(elems)


def dist_point_line(point, line_p1, line_p2):
    #print(point, line_p1, line_p2)
    point = np.array(point)
    line_p1 = np.array(line_p1)
    line_p2 = np.array(line_p2)
    cb_2 = np.sum(np.power(point - line_p2,2))
    ab_dot_cb_2 = np.power(np.dot((line_p1 - line_p2), (point - line_p2)),2)
    ab_2 = np.sum(np.power(line_p1-line_p2, 2))
    if cb_2 < ab_dot_cb_2/ab_2:
        return 0
    return np.sqrt(cb_2 - ab_dot_cb_2/ab_2)
#g matrix
g = []

def cal_gap(conts):
    g.clear()
    for key, value in conts.items():
        g.append(0)
        g.append(dist_point_line(nodes[key[0]],nodes[key[2]], nodes[key[3]]))
        g.append(0)
        g.append(dist_point_line(nodes[key[1]],nodes[key[2]], nodes[key[3]]))
        g.append(0)
        g.append(dist_point_line(nodes[key[2]],nodes[key[0]], nodes[key[1]]))
        g.append(0)
        g.append(dist_point_line(nodes[key[3]],nodes[key[0]], nodes[key[1]]))
    #print(g)


def Aelem_node(node, elem_center, t, n, reverse = False):
    t = np.array(t)
    n = np.array(n)
    #print(t)
    #print(n)
    #print(reverse)
    if reverse:
        t = t*(-1)
        n = n*(-1)
    #print(t)
    #print(n)
    R = np.array(node) - np.array(elem_center)
    #print(R)
    Alocal = np.matrix([
       [-1*t[0], -1*n[0]],
       [-1*t[1], -1*n[1]],
       [-1*float(np.cross(R,t)), -1*float(np.cross(R,n))]
    ])
    return Alocal

#global A matrix
def is_subtuple_2(tupA, tupB):
    '''
    check if tuple A(two elements) is a sub tuple of tuple B
    '''
    for i in range(0,len(tupB)-1):
        if (tupB[i],tupB[i+1]) == tupA:
            return True
    
    return False

Aglobal = np.zeros((3*len(elems), 2*4*len(conts)))
def cal_Agloabl(elems, conts):
    row = 0
    for key_e, value_e in elems.items():
        col = 0
        for key_c, value_c in conts.items():
            #print(key_c)
            #print(key_e)
            
            #the order of node sequence change the orientation of t and n
            if is_subtuple_2((key_c[0],key_c[1]), key_e) or is_subtuple_2((key_c[2],key_c[3]), key_e):
                Alocal_1 = Aelem_node(nodes[key_c[0]], value_e[-1], value_c[1], value_c[2])
                Alocal_2 = Aelem_node(nodes[key_c[1]], value_e[-1], value_c[1], value_c[2])
                Alocal_3 = Aelem_node(nodes[key_c[2]], value_e[-1], value_c[1], value_c[2])
                Alocal_4 = Aelem_node(nodes[key_c[3]], value_e[-1], value_c[1], value_c[2])
                Alocal = np.concatenate((Alocal_1, Alocal_2, Alocal_3, Alocal_4), axis = 1)
                Aglobal[row:row+3, col:col+8] = Alocal

            elif is_subtuple_2((key_c[1],key_c[0]), key_e) or is_subtuple_2((key_c[3],key_c[2]), key_e):
                Alocal_1 = Aelem_node(nodes[key_c[0]], value_e[-1], value_c[1], value_c[2], reverse = True)
                Alocal_2 = Aelem_node(nodes[key_c[1]], value_e[-1], value_c[1], value_c[2], reverse = True)
                Alocal_3 = Aelem_node(nodes[key_c[2]], value_e[-1], value_c[1], value_c[2], reverse = True)
                Alocal_4 = Aelem_node(nodes[key_c[3]], value_e[-1], value_c[1], value_c[2], reverse = True)
                Alocal = np.concatenate((Alocal_1, Alocal_2, Alocal_3, Alocal_4), axis = 1)
                Aglobal[row:row+3, col:col+8] = Alocal
            else:
                Aglobal[row:row+3, col:col+8] = np.zeros((3,8))
            col+=8
        row+=3

#print(Aglobal[-3:])

#global Y matrix
Y = np.zeros((3*4*len(conts), 2*4*len(conts)))
yunit = np.matrix([
    [1, -mu],
    [-1, -mu],
    [0,-1]
])

yunit_no_fric = np.matrix([
    [1, 0],
    [-1, 0],
    [0,-1]
])

index = 0
for key, value in conts.items():
    if value[0]>0:
        for i in range(4):
            Y[index*3:index*3+3, index*2:index*2+2] = yunit
            index+=1
    else:
        for i in range(4):
            Y[index*3:index*3+3, index*2:index*2+2] = yunit_no_fric
            index+=1        
#print(Y)

#liveload
liveload = mass_unit

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

def cal_disp_update_x(force_multi):
    # Make mosek environment
    with mosek.Env() as env:
        # Create a task object
        with env.Task(0, 0) as task:
            # Attach a log stream printer to the task
            task.set_Stream(mosek.streamtype.log, streamprinter)

            # Bound keys and values for constraints -- energy normalization
            bkc = []
            blc = []
            buc = []
            # bkc.extend([mosek.boundkey.fx])
            # blc.extend([1.0])
            # buc.extend([1.0])
            # Bound keys and values for constraints -- flow rule
            g_index = 0
            for key, value in conts.items():
                for i in range(4):
                    bkc.extend([mosek.boundkey.fx,
                            mosek.boundkey.fx])
                    blc.extend([g[2*g_index], g[2*g_index+1]])
                    buc.extend([g[2*g_index], g[2*g_index+1]])
                    g_index+=1

            # Bound keys for variables
            bkx = []
            blx = []
            bux = []
            for key, value in elems.items():
                for i in range(3):
                    bkx.append(mosek.boundkey.fr)
                    blx.append(-inf)
                    bux.append(+inf)
            for key, value in conts.items():
                for i in range(3*4):
                    bkx.append(mosek.boundkey.lo)
                    blx.append(0.0)
                    bux.append(+inf)
            for key, value in conts.items():
                for i in range(2*4):
                    bkx.append(mosek.boundkey.fr)
                    blx.append(-inf)
                    bux.append(+inf)

            # Objective coefficients
            c = []
            for key, value in elems.items():
                c.extend([-force_multi*value[-2], value[-2], 0])##############################both ft and fn
            for key, value in conts.items():
                for i in range(3*4):
                    c.append(0.0)
            for ke, value in conts.items():
                for i in range(2*4):
                    c.append(0.0)

            # Below is the sparse representation of the A
            # matrix stored by column.
            asub = []
            aval = []
            elem_index = 0
            for key, value in elems.items():
                for i in range(3):
            #for col in range(len(elems)*3):
                    col_index = []
                    col_value = []
                    # if i==0:#x direction
                    #     col_index.append(0)
                    #     col_value.append(liveload*value[0])
                    for row in range(len(conts)*4*2):
                        if Aglobal[elem_index*3+i][row] != 0:
                            #col_index.append(1+row)
                            col_index.append(row)
                            col_value.append(Aglobal[elem_index*3+i][row])
                    asub.append(col_index)
                    aval.append(col_value)
                elem_index+=1

            for col in range(len(conts)*4*3):
                col_index = []
                col_value = []
                for row in range(len(conts)*4*2):
                    if Y[col][row] != 0:
                        #col_index.append(1+row)
                        col_index.append(row)
                        col_value.append(-1*Y[col][row])
                asub.append(col_index)
                aval.append(col_value)
            
            for col in range(len(conts)*4*2):
                col_index = [col]
                col_value = [-1]
                asub.append(col_index)
                aval.append(col_value)

            #print(asub)
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

            # Set up and input quadratic objective
            qsubi = []
            qsubj = []
            qval = []
            cont_index = len(elems)*3+len(conts)*4*3
            for key, value in conts.items():
                for point in key:
                    qsubi.extend([cont_index, cont_index+1])
                    qsubj.extend([cont_index, cont_index+1])
                    qval.extend([kt, kn])
                    cont_index+=2
            print(numvar)
            task.putqobj(qsubi, qsubj, qval)

            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.minimize)

            # Solve the problem
            task.writedata("data.opf")
            task.optimize()
            if print_detail:
                # Print a summary containing information
                # about the solution for debugging purposes
                task.solutionsummary(mosek.streamtype.msg)

            # Get status information about the solution
            solsta = task.getsolsta(mosek.soltype.itr)

            if (solsta == mosek.solsta.optimal):
                xx = [0.] * numvar
                task.getxx(mosek.soltype.itr, # Request the itr solution.
                           xx)
                if print_detail:
                    print("Optimal solution: ")
                    for i in range(numvar):
                        print("x[" + str(i) + "]=" + str(xx[i]))
                #plot_elements(elems, xx, title = 'mechanism under horizontal load')

                update_node_coor(elems, xx)
                cal_elem_center(elems)
                #plot_elements(elems)
                
                return get_node_disp(elems, xx)
            else:
                if print_detail:
                    print("Other solution status")
                return 0


def main():
    alphas_plan = [0.1,0.15,0.2,0.3,0.4,0.5,0.6]
    alphas = []
    disps = []
    i = 0
    for alpha in alphas_plan:
        cal_gap(conts)
        cal_Agloabl(elems, conts)
        #force_opti = update_e(alpha)
        # if not force_opti:
        #     break
        disp = cal_disp_update_x(alpha)
        if disp == 0:
            break
        disps.append(disp)
        alphas.append(alpha)
        plot_elements(elems,title  = f'displacement at d={disp}')


    fig, ax = plt.subplots()
    ax.plot(disps, alphas)
    ax.set_xlabel('displacemnet')
    ax.set_ylabel('force multiplier')
    plt.show()
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