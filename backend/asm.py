from backend.dataflow.cfg import CFG
from backend.dataflow.cfgbuilder import CFGBuilder
from backend.dataflow.livenessanalyzer import LivenessAnalyzer
from backend.reg.bruteregalloc import BruteRegAlloc
from backend.riscv.riscvasmemitter import RiscvAsmEmitter
from utils.tac.tacprog import TACProg
from utils.riscv import Riscv

"""
Asm: we use it to generate all the asm code for the program
"""

class Asm:
    def __init__(self) -> None:
        pass

    def transform(self, prog: TACProg):
        analyzer = LivenessAnalyzer()
        
        for func in prog.funcs:
            emitter = RiscvAsmEmitter(Riscv.AllocatableRegs, Riscv.CallerSaved)
            reg_alloc = BruteRegAlloc(emitter)
            pair = emitter.selectInstr(func) #指令选择将中端TAC代码转换为riscv汇编代码
            builder = CFGBuilder()
            cfg: CFG = builder.buildFrom(pair[0])
            analyzer.accept(cfg)
            reg_alloc.accept(cfg, pair[1])

        return emitter.emitEnd()
