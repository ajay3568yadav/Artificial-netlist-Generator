class Node:
    """
    A class to represent a Node in a digital circuit design.

    Attributes:
        cell (object): The cell associated with the node.
        name (str): The name of the node.
        input_pins (list): A list of input pins connected to the node.
        output_pins (list): A list of output pins connected to the node.
        num_inputs (int): The number of input pins.
        num_outputs (int): The number of output pins.
        input_nets (list): A list of input nets connected to the node.
        output_nets (list): A list of output nets connected to the node.
        is_input (bool): Specifies if the node is an input node.
        is_output (bool): Specifies if the node is an output node.
        pin_to_net_dict (dict): A mapping of pins to nets.
        input_slew (float): The input slew of the node for timing analysis.
        cell_delay (float): The delay of the cell in the node.
        output_slew (float): The output slew of the node.
        load_capacitance (float): The load capacitance of the node.
        fanin (int): The fan-in of the node, representing the number of inputs connected to it.
        fanout (int): The fan-out of the node, representing the number of outputs driven by it.
        index (int): An optional index for the node.
    """

    def __init__(self, cell=None, name="", input_pins=None, output_pins=None, num_inputs=0,
                 num_outputs=0, input_nets=None, output_nets=None, is_input=False, is_output=False,
                 pin_to_net_dict=None, input_slew=None, cell_delay=None, output_slew=None,
                 max_cell_delay=None, load_capacitance=None, fanout=None, fanin=None, index=None):
        """
        Initializes the Node object.

        Args:
            cell (object, optional): The cell associated with the node. Defaults to None.
            name (str, optional): The name of the node. Defaults to an empty string.
            input_pins (list, optional): A list of input pins connected to the node. Defaults to an empty list.
            output_pins (list, optional): A list of output pins connected to the node. Defaults to an empty list.
            num_inputs (int, optional): The number of input pins. Defaults to 0.
            num_outputs (int, optional): The number of output pins. Defaults to 0.
            input_nets (list, optional): A list of input nets connected to the node. Defaults to an empty list.
            output_nets (list, optional): A list of output nets connected to the node. Defaults to an empty list.
            is_input (bool, optional): Indicates if the node is an input node. Defaults to False.
            is_output (bool, optional): Indicates if the node is an output node. Defaults to False.
            pin_to_net_dict (dict, optional): A mapping of pins to nets. Defaults to an empty dictionary.
            input_slew (float, optional): The input slew for timing analysis. Defaults to 0.
            cell_delay (float, optional): The delay of the cell in the node. Defaults to 0.
            output_slew (float, optional): The output slew for timing analysis. Defaults to 0.
            load_capacitance (float, optional): The load capacitance. Defaults to 0.
            fanin (int, optional): The number of inputs connected to the node. Defaults to 0.
            fanout (int, optional): The number of outputs driven by the node. Defaults to 0.
            index (int, optional): An index for the node. Defaults to 0.
        """
        self.cell = cell
        self.name = name
        self.input_pins = input_pins if input_pins is not None else []
        self.output_pins = output_pins if output_pins is not None else []
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.input_nets = input_nets if input_nets is not None else []
        self.output_nets = output_nets if output_nets is not None else []
        self.is_input = is_input
        self.is_output = is_output
        self.pin_to_net_dict = pin_to_net_dict if pin_to_net_dict is not None else {}
        self.input_slew = input_slew if input_slew is not None else 0
        self.cell_delay = cell_delay if cell_delay is not None else 0
        self.output_slew = output_slew if output_slew is not None else 0
        self.load_capacitance = load_capacitance if load_capacitance is not None else 0
        self.fanin = fanin if fanin is not None else 0
        self.fanout = fanout if fanout is not None else 0
        self.index = index if index is not None else 0
