# Class to represent a LUT (Look-Up Table)
class LUT:
    """
    A class to represent a Look-Up Table (LUT) used in digital circuit design.

    Attributes:
        values (list): A list of values representing the LUT contents.
        timing_sense (str): A string indicating the timing sense of the LUT.
        timing_type (str): A string indicating the timing type of the LUT.
    """

    def __init__(self, values=None, timing_sense=None, timing_type=None):
        """
        Initializes the LUT object.

        Args:
            values (list, optional): A list of values representing the LUT contents. Defaults to an empty list.
            timing_sense (str, optional): Specifies the timing sense of the LUT. Defaults to an empty string.
            timing_type (str, optional): Specifies the timing type of the LUT. Defaults to an empty string.
        """
        self.values = values if values is not None else []
        self.timing_sense = timing_sense if timing_sense is not None else ""
        self.timing_type = timing_type if timing_type is not None else ""
