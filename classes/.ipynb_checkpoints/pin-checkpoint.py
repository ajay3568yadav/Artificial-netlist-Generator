class Pin:
    """
    A class to represent a Pin in a digital circuit design.

    Attributes:
        name (str): The name of the pin.
        capacitance (float): The capacitance value associated with the pin.
        input_transition_time (list): A list representing the input transition times.
        total_output_net_capacitance (list): A list representing the total output net capacitance.
        is_input (bool): Specifies if the pin is an input pin.
        is_output (bool): Specifies if the pin is an output pin.
        related_pin_luts (dict): A dictionary of related Look-Up Tables (LUTs) for this pin.
        connected_node (object): The node that this pin is connected to.
    """

    def __init__(self, name=None, capacitance=0.0, input_transition_time=None, total_output_net_capacitance=None, 
                 is_input=False, is_output=False, related_pin_luts=None):
        """
        Initializes the Pin object.

        Args:
            name (str, optional): The name of the pin. Defaults to an empty string.
            capacitance (float, optional): The capacitance of the pin. Defaults to 0.0.
            input_transition_time (list, optional): A list of input transition times. Defaults to an empty list.
            total_output_net_capacitance (list, optional): A list of total output net capacitance values. Defaults to an empty list.
            is_input (bool, optional): Specifies if the pin is an input pin. Defaults to False.
            is_output (bool, optional): Specifies if the pin is an output pin. Defaults to False.
            related_pin_luts (dict, optional): A dictionary of related Look-Up Tables (LUTs). Defaults to an empty dictionary.
        """
        self.name = name if name is not None else ""
        self.capacitance = capacitance
        self.input_transition_time = input_transition_time if input_transition_time is not None else []
        self.total_output_net_capacitance = total_output_net_capacitance if total_output_net_capacitance is not None else []
        self.is_input = is_input
        self.is_output = is_output
        self.related_pin_luts = related_pin_luts if related_pin_luts is not None else {}
        self.connected_node = None