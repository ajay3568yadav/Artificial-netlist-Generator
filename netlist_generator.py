from liberty.parser import parse_liberty
from classes.lut import LUT
from classes.cell import Cell
from classes.net import Net
from classes.pin import Pin
from classes.node import Node
import copy
import random
import pickle
import os

# location of the liberty file
liberty_file = # you lib file location here example: "/home/ajayyadav/ANG/PDK_130nm/sky130_fd_sc_hd__tt_025C_1v80.lib"
library = parse_liberty(open(liberty_file).read())

def get_index_1(pin):
    index_1 = []
    if pin.get_groups('timing'):
        for val in pin.get_groups('timing')[0].get_groups('cell_fall')[0]['index_1'][0].value.split(","):
            index_1.append(float(val))
    return index_1
    
def get_index_2(pin):
    index_2 = []
    if pin.get_groups('timing'):
        for val in pin.get_groups('timing')[0].get_groups('cell_fall')[0]['index_2'][0].value.split(","):
            index_2.append(float(val.strip()))
    return index_2
    
def get_fall_delay(pin, i):
    if pin.get_groups('timing'):
        if pin.get_groups('timing')[i].get_groups('cell_fall'):
            values = pin.get_groups('timing')[i].get_groups('cell_fall')[0]['values']
            fall_delay = [ [float(num.strip()) for num in row.value.split(',')] for row in values ]
            return fall_delay
    return None
    
def get_rise_delay(pin, i):
    if pin.get_groups('timing'):
        if pin.get_groups('timing')[i].get_groups('cell_rise'):
            values = pin.get_groups('timing')[i].get_groups('cell_rise')[0]['values']
            rise_delay = [ [float(num.strip()) for num in row.value.split(',')] for row in values ]
            return rise_delay
    return None
    
def get_fall_transition(pin, i):
    if pin.get_groups('timing'):
        if pin.get_groups('timing')[i].get_groups('fall_transition'):
            values = pin.get_groups('timing')[i].get_groups('fall_transition')[0]['values']
            fall_transition = [ [float(num.strip()) for num in row.value.split(',')] for row in values ]
            return fall_transition
    return None
    
def get_rise_transition(pin, i):
    if pin.get_groups('timing'):
        if pin.get_groups('timing')[i].get_groups('rise_transition'):
            values = pin.get_groups('timing')[i].get_groups('rise_transition')[0]['values']
            rise_transition = [ [float(num.strip()) for num in row.value.split(',')] for row in values ]
            return rise_transition
    return None
    
def get_related_pin_luts(pin):
    related_pin_luts = {}
    related_timings = pin.get_groups('timing')
    for i in range(len(related_timings)):
        related_pin = related_timings[i]['related_pin']
        timing_sense = related_timings[i]['timing_sense']
        timing_type = related_timings[i]['timing_type']
        lut_dict = {
            'cell_fall': LUT(values = get_fall_delay(pin,i), timing_sense = timing_sense, timing_type=timing_type),
            'cell_rise': LUT(values = get_rise_delay(pin, i), timing_sense = timing_sense, timing_type=timing_type),
            'fall_transition': LUT(values = get_fall_transition(pin, i), timing_sense = timing_sense, timing_type=timing_type),
            'rise_transition': LUT(values = get_rise_transition(pin, i), timing_sense = timing_sense, timing_type=timing_type)
        }
        if related_pin not in related_pin_luts:
            related_pin_luts[related_pin] = {}
        related_pin_luts[related_pin] = lut_dict
    return related_pin_luts


cells_dict = {}

for cell_group in library.get_groups('cell'):
    cell_name = cell_group.args[0]
    output_pin_related_pins = {}
    pins = cell_group.get_groups('pin')
    related_pin_lut = []
    input_pins = []
    output_pins = []
    is_seq = False
    for pin in pins:
        pin_name = pin.args[0]
        if pin['clock'] == "true":
            is_seq = True
        if pin['direction'] == 'input':
            capacitance = pin['capacitance']
            is_input = True        
            input_pins.append(Pin(name = pin_name, capacitance=capacitance, is_input=is_input))
        elif pin['direction'] == 'output':
            is_output = True
            capacitance = pin['max_capacitance']
            input_transition_time = get_index_1(pin) #index 1
            total_output_net_capacitance = get_index_2(pin) #index 2
            related_pin_luts = get_related_pin_luts(pin) #related_pin_lut
            ## note: there can be more than one output so make it a list
            related_pin_lut.append(related_pin_luts)
            output_pins.append(Pin(name = pin_name, capacitance=capacitance, input_transition_time=input_transition_time,
                                  total_output_net_capacitance=total_output_net_capacitance,
                                  related_pin_luts=related_pin_luts, is_output=is_output))
            
    if cell_name not in cells_dict:
        cells_dict[cell_name] = Cell(name = cell_name, input_pins=input_pins, output_pins=output_pins, 
                                     is_seq= is_seq, num_inputs = len(input_pins), 
                                     num_outputs = len(output_pins), input_pins_names = list(related_pin_luts.keys()),
                                    output_pins_names = [p.name for p in output_pins], related_pin_lut= related_pin_lut)


        
def filter_cells(cells_dict):
    deleted_cells = []
    for cell in cells_dict:
        if cells_dict[cell].name!="input" and cells_dict[cell].num_inputs == 0:
            deleted_cells.append(cell)
    for cell in deleted_cells:
        del(cells_dict[cell])


filter_cells(cells_dict)

cells = list(cells_dict.values())

cell_input_output_dict = {cell_name:[cell.num_inputs,cell.num_outputs] for cell_name, cell in cells_dict.items()}
# cell_input_output_dict["input"] = [0,1]

cell_input_size_dict = {cell_name:cell_input_output_dict[cell_name][0] for cell_name in cell_input_output_dict}

# number of outputs for each cell
cell_min_output_dict = {cell:cell_input_output_dict[cell][1] for cell in cell_input_output_dict}

comb_cells = []
for cell in cells:
    if cell.is_seq == False:
        comb_cells.append(cell)

comb_cells_for_output = []

for cell in comb_cells:
    if cell_min_output_dict[cell.name]>=1:
        comb_cells_for_output.append(cell)
        
        
with open('comb_cells_for_output.pkl', 'wb') as file:
    pickle.dump(comb_cells_for_output, file)
    
    
def is_cyclic_util(v, visited, rec_stack, graph, path):
    # Mark the current node as visited and add to recursion stack
    visited.add(v)
    rec_stack.add(v)
    path.append(v)
    
    # Recur for all neighbors
    for neighbor in graph.get(v, []):
        # If neighbor is not visited, recurse on it
        if neighbor not in visited:
            if is_cyclic_util(neighbor, visited, rec_stack, graph, path):
                return True
        # If neighbor is in rec_stack, then cycle detected
        elif neighbor in rec_stack:
            # Extract the cycle from the path
            cycle_index = path.index(neighbor)
            cycle = path[cycle_index:]  # cycle from the first occurrence of neighbor to the end of path
            for p in cycle:
                print(p)
            return True
    
    # Remove the node from the recursion stack and path
    rec_stack.remove(v)
    path.pop()
    return False

def is_cyclic(graph):
    visited = set()
    rec_stack = set()
    path = []
    
    # Call the helper function for each node
    for node in graph:
        if node not in visited:
            if is_cyclic_util(node, visited, rec_stack, graph, path):
                return True
    return False


def get_comb_cells(comb_cells_for_output, n):
    comb_cells_list = []
    c_cells = random.choices(comb_cells_for_output, k=n)
    for cell in c_cells:
        comb_cells_list.append(copy.copy(cells_dict[cell.name]))
    return comb_cells_list

def get_input_cells(n):
    i=0
    input_nodes = []
    while i<n:
        name = "input"+str(i)
        input_nodes.append(Cell(name = name, num_outputs = 1, is_input = True))
        i+=1
    return input_nodes

def get_output_cells(comb_cells_for_output, n):
    comb_cells_list = []
    c_cells = random.choices(comb_cells_for_output, k=n)
    for cell in c_cells:
        cell_copy = copy.copy(cells_dict[cell.name])
        cell_copy.is_output = True
        comb_cells_list.append(cell_copy)
    return comb_cells_list


def inputs_available(nodes_list):
    for node in nodes_list:
        available_connects = node.num_inputs - len(node.input_nets)
        if available_connects !=0:
            return True
    return False

def fanout_net_check(netlist, next_list):
    for node in next_list:
        for in_net in node.input_nets:
            if in_net.node.is_input:
                print("ERROR")
            if len(netlist[in_net.node]) >1:
                return True
    return False
            
def create_cell_list():
    cell_list = []
    rent_parameter = random.uniform(0.5,1)
    t_constant = 1
    num_gates = 20
    T_terminals = int(t_constant*(num_gates)**rent_parameter)
    
    num_inputs = int(0.6*T_terminals)
    num_outputs = T_terminals - num_inputs
    num_comb_cells = num_gates - num_outputs
                
    
    comb_cells_list = get_comb_cells(comb_cells_for_output, num_comb_cells)
    input_cells = get_input_cells(num_inputs)
    output_cells = get_output_cells(comb_cells_for_output, num_outputs)
    cell_list = comb_cells_list+output_cells+input_cells
    return cell_list


def net_gen():
        
    cells_list = create_cell_list()
    print("length of cell list", len(cells_list))
    
    t=0
    n=0
    in_nets = 0
    out_nets = 0
    net_to_node_dict = {}
    
    input_nodes_list = []
    output_nodes_list = []
    node_list = []
            
    for cell in cells_list:
        node_index = 0
        node = Node(cell = cell, name = ("C"+str(n)), 
                    input_pins = cell.input_pins, 
                    output_pins = cell.output_pins,
                    num_inputs = cell.num_inputs, 
                    num_outputs=cell.num_outputs, 
                    output_nets = [],
                    input_nets = [],
                    is_input = cell.is_input,
                    is_output = cell.is_output,
                    index = node_index if not is_input else 0)
        n+=1
        #input nets
        if node.is_input:
            input_net = "in"+str(in_nets)
            node.output_nets.append(Net(name = input_net, node = node, connected_nodes =[]))
            in_nets+=1
            input_nodes_list.append(node)
        elif node.cell.is_output:
            node.is_output = True
            out_net = "out"+str(out_nets)
            node.output_nets.append(Net(name = out_net, node = node, connected_nodes =[]))
            out_nets+=1
            for i in range(node.num_outputs-1):
                net = "net"+str(t)
                t+=1
                node.output_nets.append(Net(name = net, node = node, connected_nodes =[]))
            output_nodes_list.append(node)
        else:
            #comb nets
            for i in range(node.num_outputs):
                net = "net"+str(t)
                t+=1
                node.output_nets.append(Net(name = net, node= node, connected_nodes =[]))
            node_list.append(node)
            
        node_index+=1
            
    all_nodes = input_nodes_list+node_list+output_nodes_list
        
    netlist = {}
    net_dict = {}

    for current_node in all_nodes:
        #print(current_node, current_node.cell.name, current_node.output_nets)
        current_node.input_nets = []
        netlist[current_node] = []
    
    # Connect inputs first
    for node in input_nodes_list:
        connect_to_node = random.choices(node_list,k=1)[0]
        
        #checking how many connections are left for the selected node
        if connect_to_node.num_inputs == len(connect_to_node.input_nets): #or node in connect_to_node.input_nets:
            while True:
                connect_to_node = random.choices(node_list,k=1)[0]
                connections_left = len(connect_to_node.input_nets)
                if connect_to_node.num_inputs - connections_left: #and node not in connect_to_node.input_nets:
                    break
        
        random_net = random.choices(node.output_nets,k=1)[0]
        connect_to_node.input_nets.append(random_net)
        random_net.connected_nodes.append(connect_to_node)
        netlist[node].append(connect_to_node)
        
        comb_cell_first_index = len(input_nodes_list)
        current_index = comb_cell_first_index
        
    ##########################  ##########################  ##########################
    
    # Connecting combinational cells
    for node in all_nodes[comb_cell_first_index:]:
        # fullfill input connections
        num_inputs_remaining = node.num_inputs - len(node.input_nets)
        
        if num_inputs_remaining:
            in_node = random.choices(all_nodes[:current_index],k=num_inputs_remaining)
            
            for n in in_node:  
                random_net = random.choices(n.output_nets,k=1)[0]
                node.input_nets.append(random_net)
                netlist[n].append(node)
                random_net.connected_nodes.append(node)
    
        #fullfill output connections
        if current_index+1 != len(all_nodes):
    
            # Choose a random output node
            out_node = random.choices(all_nodes[current_index+1:],k=1)[0]
    
            # Check the remaining connections for the selected random node
            out_node_remaining_inputs = (out_node.num_inputs - len(out_node.input_nets))
    
            # if selected node has no connections left but there are connectable nets ahead... 
            if out_node_remaining_inputs == 0 and inputs_available(all_nodes[current_index+1:]):
    
                #keep looking for that net
                while out_node_remaining_inputs == 0:
                    out_node = random.choices(all_nodes[current_index+1:],k=1)[0]
                    out_node_remaining_inputs = (out_node.num_inputs - len(out_node.input_nets))
                    if out_node_remaining_inputs<0:
                        print(out_node, out_node.cell.name, "errorring")

                random_net = random.choices(node.output_nets,k=1)[0]
                netlist[node].append(out_node)
                out_node.input_nets.append(random_net)
                random_net.connected_nodes.append(out_node)
            # if no input is available ahead
            elif not inputs_available(all_nodes[current_index+1:]):
                
                # and the current node is an input node, then no need to connect it
                if node.is_output:
                    netlist[node] = []
                # if its not then search for a node whose input has been connected several times and replace it with the current node
                else:
                    net_with_wanted_fanout_available = fanout_net_check(netlist, all_nodes[current_index+1:])
                    if net_with_wanted_fanout_available:
                        flag = 0
                        while flag == 0:
                            out_node = random.choices(all_nodes[current_index+1:],k=1)[0]
                            for out_node_input_net in out_node.input_nets:
                                if len(netlist[out_node_input_net.node])>1:
                                    netlist[out_node_input_net.node].remove(out_node)
                                    out_node.input_nets.remove(out_node_input_net)
                                    random_net = random.choices(node.output_nets,k=1)[0]
                                    out_node.input_nets.append(random_net)
                                    random_net.connected_nodes.append(out_node)
                                    netlist[node].append(out_node)
                                    flag = 1
                                    break
                    else:
                        node.is_output = True
                        netlist[node] = []
                        
            else:
                random_net = random.choices(node.output_nets,k=1)[0]
                netlist[node].append(out_node)
                out_node.input_nets.append(random_net)
                random_net.connected_nodes.append(out_node)
        else:
            if node not in netlist:
                netlist[node] = []
        current_index+=1
        
    if not is_cyclic(netlist):
        print("No cycle detected")
    
    return [netlist, input_nodes_list, node_list, output_nodes_list]



def adjust_connections(netlist):
    for node in netlist:
        output_nets = node.output_nets
        for net in output_nets:
            if len(net.connected_nodes) == 0 and not node.is_output:
                return True
    return False        

def netlist_writer(netlist, input_nodes_list, output_nodes_list, node_list, f_name, folder_path):
    # Ensure the directory exists
    os.makedirs(folder_path, exist_ok=True)

    # Construct the full file path
    file_name = os.path.join(folder_path, f_name + ".txt")
    
    with open(file_name, "w") as file:
        # Module section
        file.write("module ANG (\n")
    
        # Input output section
        i_o_list = input_nodes_list + output_nodes_list
        for n in i_o_list:
            if n.is_input:
                file.write(f"input {n.output_nets[0].name},\n")
            if n.is_output:
                file.write(f"output {n.output_nets[0].name},\n")
        file.write(");\n\n")
    
        # Wires and nets section
        for n in input_nodes_list + output_nodes_list + node_list:
            for net in n.output_nets:
                file.write(f"wire {net.name};\n")
    
        file.write("\n\n")
        
        # Nodes section
        for node in netlist:
            if node.is_input:
                continue
            file.write(f"{node.cell.name.value} {node.name} (\n")
            for i in range(len(node.input_nets)):
                #for k in node.input_nets[i].output_nets:
                file.write(f".{node.cell.input_pins_names[i].value}({node.input_nets[i].name}),\n")
            for j in range(len(node.output_nets)):
                file.write(f".{node.cell.output_pins_names[j].value}({node.output_nets[j].name}),\n")
            file.write(");\n\n")
        file.write("\nendmodule")
        
netlist_list_dataset_1 = {}
i=0
num_netlists_needed = 100
while i<num_netlists_needed:
    f_name = "gen_netlist_"+str(i)
    folder_path = "Netlists"
    netlist, input_nodes_list, node_list, output_nodes_list = net_gen()
    if adjust_connections(netlist):
        print(f_name,"has adjusted connections")
        continue
    netlist_list_dataset_1[f_name] = netlist
    netlist_writer(netlist, input_nodes_list, output_nodes_list, node_list, f_name, folder_path)
    i+=1
