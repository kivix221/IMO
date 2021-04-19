import numpy as np

def steep_lm(matrix, cycle1, cycle2,n=0):
    cycle1, cycle2 = steep_swap_vertices_between_cycle_lm(matrix, cycle1[:-1], cycle2[:-1])
    cycle1,cycle2 = steep_swap_edges_in_cycle_lm(matrix,cycle1),steep_swap_edges_in_cycle_lm(matrix,cycle2)
    
    cycle1, cycle2 = np.concatenate([cycle1, [cycle1[0]]]), np.concatenate([cycle2, [cycle2[0]]])
    return cycle1, cycle2 

#edges
def steep_swap_edges_in_cycle_lm(matrix,cycle):
    lm = init_swap_edges(matrix,cycle)
    m = lm.pop(0)
    cycle = swap_edges(cycle,m) 
    
    while True:
        add_m_after_edges(matrix,cycle,m,lm)
        lm.sort(key=lambda movement: movement[0],reverse=True)
        
        m = None
        m_to_remove = []
        for i in range(len(lm)):
            is_applicable = check_swap_edges(cycle,lm[i])
            if  is_applicable== 0:
                m_to_remove.append(lm[i])
                continue
            elif is_applicable == 1:
                continue
            elif is_applicable == 2:
                m = lm[i]
                m_to_remove.append(lm[i])
                break
        
        lm[:] = [x for x in lm if x not in m_to_remove]
        if not m:
            break
        
        cycle = swap_edges(cycle,m)
            
    return cycle


def check_swap_edges(cycle,m):
    is_applicable = 0 #0-usun 1-pomin 2-aplikuj
    n1,succ_n1,n2,succ_n2 = np.where(cycle[:-1] == m[1][0])[0],np.where(cycle[:-1] == m[1][1])[0], \
         np.where(cycle[:-1] == m[1][2])[0], np.where(cycle[:-1] == m[1][3])[0] #indeksy!
         
    if n1 and succ_n1 and n2 and succ_n2:
        is_edge1_exist = (cycle[(n1 + 1) % len(cycle)] == cycle[succ_n1])  
        is_edge1_exist_revers = (cycle[n1 - 1] == cycle[succ_n1])    

        is_edge2_exist = (cycle[(n2 + 1) % len(cycle)] == cycle[succ_n2])  
        is_edge2_exist_revers = (cycle[n2 - 1] == cycle[succ_n2])
        
        if (is_edge1_exist or  is_edge1_exist_revers) and (is_edge2_exist or is_edge2_exist_revers):
            is_applicable = 1
        if (is_edge1_exist and is_edge2_exist):
            is_applicable = 2
       
           
    return is_applicable

def add_m_after_edges(matrix,cycle,m,lm):
    n1,succ_n2 = int(np.where(cycle == m[1][0])[0]),int(np.where(cycle == m[1][-1])[0])
    for i in range(n1-1):
        if n1 < 2:
            break
        for j in range(n1,succ_n2 +1):
            old_distance = matrix[cycle[i]][cycle[(i + 1) % len(cycle)]] + matrix[cycle[j]][cycle[(j + 1) % len(cycle)]]
            new_distance = matrix[cycle[i]][cycle[j]] + matrix[cycle[(i + 1) % len(cycle)]][cycle[(j + 1) % len(cycle)]]

            delta = new_distance - old_distance
            if new_distance - old_distance < 0:
                swap = [cycle[i],cycle[(i+1) % len(cycle)],cycle[j],cycle[(j+1) % len(cycle)]] #n1,succ(n1),n2,succ(n2)
                new_m = (abs(delta),swap)
                lm.append(new_m)
                    
    for i in range(n1-1,succ_n2 + 1):
            for j in range(n1 + 2,len(cycle)):
                old_distance = matrix[cycle[i]][cycle[(i + 1) % len(cycle)]] + matrix[cycle[j]][cycle[(j + 1) % len(cycle)]]
                new_distance = matrix[cycle[i]][cycle[j]] + matrix[cycle[(i + 1) % len(cycle)]][cycle[(j + 1) % len(cycle)]]

                delta = new_distance - old_distance
                if new_distance - old_distance < 0:
                    swap = [cycle[i],cycle[(i+1) % len(cycle)],cycle[j],cycle[(j+1) % len(cycle)]] #n1,succ(n1),n2,succ(n2)
                    new_m = (abs(delta),swap)
                    lm.append(new_m)

def swap_edges(cycle,m):
    i,j = int(np.where(cycle == m[1][0])[0]),int(np.where(cycle == m[1][-2])[0])
    cycle[i + 1], cycle[j] = cycle[j], cycle[i + 1]
    cycle[i + 2:j] = cycle[i + 2:j][::-1]
    
    return cycle


def init_swap_edges(matrix,cycle):
    lm = []
    for i in range(len(cycle)):
        for j in range(i + 2, len(cycle)):
            old_distance = matrix[cycle[i]][cycle[(i + 1) % len(cycle)]] + matrix[cycle[j]][cycle[(j + 1) % len(cycle)]]
            new_distance = matrix[cycle[i]][cycle[j]] + matrix[cycle[(i + 1) % len(cycle)]][cycle[(j + 1) % len(cycle)]]

            delta = new_distance - old_distance
           
            if new_distance - old_distance < 0:    
                swap = [cycle[i],cycle[(i+1) % len(cycle)],cycle[j],cycle[(j+1) % len(cycle)]] #n1,succ(n1),n2,succ(n2)
                m = (abs(delta),swap)
                lm.append(m)
                
    return sorted(lm, key=lambda movement: movement[0],reverse=True)  

#vertices
def steep_swap_vertices_between_cycle_lm(matrix,cycle1,cycle2):
    lm = init_swap_vertices(matrix,cycle1,cycle2)
    m = lm.pop(0)
    cycle1,cycle2 = swap_vertices(cycle1,cycle2,m) 
    
    while True:
        add_m_after_vert(matrix,cycle1,cycle2,m,lm)
        lm.sort(key=lambda movement: movement[0],reverse=True)
        
        m = None
        m_to_remove = []
        for i in range(len(lm)):
            is_applicable = check_swap_vert(matrix,cycle1,cycle2,lm[i])
            if  is_applicable == 0:
                m_to_remove.append(lm[i])
            elif is_applicable == 1:
                continue
            elif is_applicable == 2:
                m = lm[i]
                m_to_remove.append(lm[i])
                break
        
        lm[:] = [x for x in lm if x not in m_to_remove]
        if not m:
            break
        
        cycle1,cycle2 = swap_vertices(cycle1,cycle2,m)
  
    return cycle1, cycle2

def swap_vertices(cycle1,cycle2,m):
    i,j = int(np.where(cycle1 == m[1][0])[0]),int(np.where(cycle2 == m[1][-1])[0])
    cycle1[i],cycle2[j] = cycle2[j],cycle1[i]
  
    return cycle1,cycle2

def add_m_after_vert(matrix,cycle1,cycle2,m,lm):
    n1,n2 = m[1][-1],m[1][0]
    i1,i2 = int(np.where(cycle1 == n1)[0]),int(np.where(cycle2 == n2)[0])
    
    #cycle1
    for j, d in enumerate(cycle2):
        tb = matrix[cycle1[i1 - 1]][n1] + matrix[n1][cycle1[(i1 + 1) % len(cycle1)]] + matrix[cycle2[j - 1]][d] + matrix[d][
            cycle2[(j + 1) % len(cycle2)]]
        ta = matrix[cycle1[i1 - 1]][d] + matrix[d][cycle1[(i1 + 1) % len(cycle1)]] + matrix[cycle2[j - 1]][n1] + matrix[n1][
            cycle2[(j + 1) % len(cycle2)]]  
        delta = ta - tb
        if delta < 0:
            new_m = (abs(delta),(cycle1[i1],cycle2[j]))
            lm.append(new_m)
    #cycle2
    for j, d in enumerate(cycle1):
            tb = matrix[cycle1[j - 1]][d] + matrix[d][cycle1[(j + 1) % len(cycle1)]] + matrix[cycle2[i2 - 1]][n2] + matrix[n2][
                cycle2[(i2 + 1) % len(cycle2)]]
            ta = matrix[cycle1[j - 1]][n2] + matrix[n2][cycle1[(j + 1) % len(cycle1)]] + matrix[cycle2[i2 - 1]][d] + matrix[d][
                cycle2[(i2 + 1) % len(cycle2)]]
            
            delta = ta - tb
            if delta < 0:
                new_m = (abs(delta),(cycle1[j],cycle2[i2])) #n1, prec(n1),succ(n1),prec(n2),succ(n2),n2
                lm.append(new_m)
 
def check_swap_vert(matrix,cycle1,cycle2,m):
    is_applicable = 0
    n1 = np.where(cycle1 == m[1][0])[0] 
    n2 = np.where(cycle2 == m[1][-1])[0]

    if n1 and n2:
        is_applicable =1
        n1,n2 = int(n1),int(n2)
        tb = matrix[cycle1[n1 - 1]][m[1][0]] + matrix[m[1][0]][cycle1[(n1 + 1) % len(cycle1)]] + \
            matrix[cycle2[n2 - 1]][m[1][-1]] + matrix[m[1][-1]][cycle2[(n2 + 1) % len(cycle2)]]
        ta = matrix[cycle1[n1 - 1]][m[1][-1]] + matrix[m[1][-1]][cycle1[(n1 + 1) % len(cycle1)]] + \
            matrix[cycle2[n2 - 1]][m[1][0]] + matrix[m[1][0]][cycle2[(n2 + 1) % len(cycle2)]]
        if ta - tb == -m[0]:
            is_applicable = 2
        
    return is_applicable


def init_swap_vertices(matrix,cycle1,cycle2):
    lm = []
    for i, c in enumerate(cycle1):
        for j, d in enumerate(cycle2):
            tb = matrix[cycle1[i - 1]][c] + matrix[c][cycle1[(i + 1) % len(cycle1)]] + matrix[cycle2[j - 1]][d] + matrix[d][
                cycle2[(j + 1) % len(cycle2)]]
            ta = matrix[cycle1[i - 1]][d] + matrix[d][cycle1[(i + 1) % len(cycle1)]] + matrix[cycle2[j - 1]][c] + matrix[c][
                cycle2[(j + 1) % len(cycle2)]]
            
            delta = ta - tb
            if delta < 0:
                m = (abs(delta),(cycle1[i],cycle2[j])) #n1,,n2
                lm.append(m)
    return sorted(lm, key=lambda movement: movement[0],reverse=True)      

   
                
