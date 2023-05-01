import dataclasses
import typing
import numpy

@dataclasses.dataclass 
class AntoineParameters():
    A: float
    B: float
    C: float
    log_type: typing.Literal["ln", "log"] = "ln"

    def P(self, T):
        if self.log_type == "ln":
            return numpy.exp(self.A - self.B / (T + self.C))
        elif self.log_type == "log":
            return 10**(self.A - self.B / (T + self.C))