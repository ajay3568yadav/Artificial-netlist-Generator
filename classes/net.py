class Net:
    """
    A class to represent a Net in a digital circuit design.

    Attributes:
        name (str): The name of the net.
        node (object): The node associated with the net.
        fanout (int): The fanout of the net, representing the number of nodes driven by this net.
        connected_nodes (list): A list of nodes that are connected to this net.
    """

    def __init__(self, name="", node=None, fanout=0, connected_nodes=None):
        """
        Initializes the Net object.

        Args:
            name (str, optional): The name of the net. Defaults to an empty string.
            node (object, optional): The node associated with the net. Defaults to None.
            fanout (int, optional): The fanout of the net. Defaults to 0.
            connected_nodes (list, optional): A list of nodes connected to this net. Defaults to an empty list.
        """
        self.name = name
        self.node = node
        self.fanout = fanout
        self.connected_nodes = connected_nodes if connected_nodes is not None else []
