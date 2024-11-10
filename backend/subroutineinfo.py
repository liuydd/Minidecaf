from utils.label.funclabel import FuncLabel
from frontend.symbol.varsymbol import VarSymbol
from typing import Dict

"""
SubroutineInfo: collect some info when selecting instr which will be used in SubroutineEmitter
"""


class SubroutineInfo:
    # def __init__(self, funcLabel: FuncLabel) -> None:
    #     self.funcLabel = funcLabel
    def __init__(self, funcLabel: FuncLabel, numArgs: int) -> None:
        self.funcLabel = funcLabel
        self.numArgs = numArgs
        
        self.offsets: Dict[str, int] = {}
        self.size = 0

        # for name, symbol in self.arrays.items():
        #     self.offsets[name] = self.size
        #     self.size += symbol.type.size

    def __str__(self) -> str:
        return "funcLabel: {}".format(
            self.funcLabel.name,
        )
