class Error(Exception):
    """Base Exceptions class"""

    def __init__(self):
        Exception.__init__(self)
        self.message = 'MDPToolbox'

    def __str__(self):
        return repr(self.message)

class InvalidError(Error):
    """Invalid Exceptions"""

    def __init__(self, msg):
        Error.__init__(self)
        self.message += msg
        self.args = tuple(msg)

class NonNegativeError(Error):
    """Stochastic Errors"""

    default_msg = "The transition probability matrix is negative."

    def __init__(self, msg=None):
        if msg is None:
            msg = self.default_msg
        Error.__init__(self)
        self.message += msg
        self.args = tuple(msg)

class SquareError(Error):
    """Square Errors in transision matrix"""

    default_msg = "The transition probability matrix is not square."

    def __init__(self, msg=None):
        if msg is None:
            msg = self.default_msg
        Error.__init__(self)
        self.message += msg
        self.args = tuple(msg)

class StochasticError(Error):
    """Stochastic transition matrix errors"""

    default_msg = "The transition probability matrix is not stochastic."

    def __init__(self, msg=None):
        if msg is None:
            msg = self.default_msg
        Error.__init__(self)
        self.message += msg
        self.args = tuple(msg)