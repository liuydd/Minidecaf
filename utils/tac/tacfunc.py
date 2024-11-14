from utils.label.funclabel import FuncLabel
from frontend.symbol.varsymbol import VarSymbol
from typing import List, Dict
from .tacinstr import TACInstr


class TACFunc:
    def __init__(self, entry: FuncLabel, numArgs: int, arrays: Dict[str, VarSymbol], p_arrays: Dict[int, VarSymbol]) -> None:
        self.entry = entry
        self.numArgs = numArgs
        self.arrays = arrays
        self.p_arrays = p_arrays
        self.instrSeq = []
        self.tempUsed = 0

    def getInstrSeq(self) -> list[TACInstr]:
        return self.instrSeq

    def getUsedTempCount(self) -> int:
        return self.tempUsed

    def add(self, instr: TACInstr) -> None:
        self.instrSeq.append(instr)

    def printTo(self) -> None:
        for instr in self.instrSeq:
            if instr.isLabel():
                print(instr)
            else:
                print("    " + str(instr))
