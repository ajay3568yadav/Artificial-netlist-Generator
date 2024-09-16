# Class to represent a Cell
class Cell:
    """
    A class to represent a Cell in a digital circuit design.

    Attributes:
        name (str): The name of the cell.
        input_pins (list): A list of input pins associated with the cell.
        output_pins (list): A list of output pins associated with the cell.
        input_pins_names (list): A list of names for the input pins.
        output_pins_names (list): A list of names for the output pins.
        is_seq (bool): Specifies whether the cell is sequential.
        num_inputs (int): The number of input pins.
        num_outputs (int): The number of output pins.
        is_input (bool): Specifies if the cell is an input cell.
        is_output (bool): Specifies if the cell is an output cell.
        related_pin_lut (list): A list of related Look-Up Tables (LUTs) associated with the cell.
    """
    def __init__(self, name="", input_pins=None, output_pins=None, input_pins_names=None, output_pins_names=None, 
                 is_seq = False, num_inputs = 0, num_outputs = 0, is_input=False,is_output = False,related_pin_lut=None):
        """
        Initializes the Cell object.

        Args:
            name (str, optional): The name of the cell. Defaults to an empty string.
            input_pins (list, optional): A list of input pins associated with the cell. Defaults to an empty list.
            output_pins (list, optional): A list of output pins associated with the cell. Defaults to an empty list.
            input_pins_names (list, optional): A list of names for the input pins. Defaults to an empty list.
            output_pins_names (list, optional): A list of names for the output pins. Defaults to an empty list.
            is_seq (bool, optional): Indicates whether the cell is sequential. Defaults to False.
            num_inputs (int, optional): The number of input pins. Defaults to 0.
            num_outputs (int, optional): The number of output pins. Defaults to 0.
            is_input (bool, optional): Indicates if the cell is an input cell. Defaults to False.
            is_output (bool, optional): Indicates if the cell is an output cell. Defaults to False.
            related_pin_lut (list, optional): A list of related Look-Up Tables (LUTs) associated with the cell. Defaults to an empty list.
        """
        self.name = name
        self.input_pins = input_pins if input_pins is not None else []
        self.output_pins = output_pins if output_pins is not None else []
        self.input_pins_names = input_pins_names if input_pins_names is not None else []
        self.output_pins_names = output_pins_names if output_pins_names is not None else []
        self.is_seq = is_seq
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.is_input = is_input
        self.is_output = is_output
        self.related_pin_lut = related_pin_lut if related_pin_lut is not None else []