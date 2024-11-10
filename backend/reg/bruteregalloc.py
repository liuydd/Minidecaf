import random

from backend.dataflow.basicblock import BasicBlock, BlockKind
from backend.dataflow.cfg import CFG
from backend.dataflow.loc import Loc
from backend.reg.regalloc import RegAlloc
from backend.riscv.riscvasmemitter import RiscvAsmEmitter
from backend.riscv.riscvasmemitter import RiscvSubroutineEmitter
from backend.subroutineinfo import SubroutineInfo
from utils.riscv import Riscv
from utils.tac.reg import Reg
from utils.tac.temp import Temp
from utils.tac.tacop import InstrKind
from utils.tac.tacinstr import TACInstr
from utils.tac.backendinstr import BackendInstr

"""
BruteRegAlloc: one kind of RegAlloc

bindings: map from temp.index to Reg

we don't need to take care of GlobalTemp here
because we can remove all the GlobalTemp in selectInstr process

1. accept：根据每个函数的 CFG 进行寄存器分配，寄存器分配结束后生成相应汇编代码
2. bind：将一个 Temp 与寄存器绑定
3. unbind：将一个 Temp 与相应寄存器解绑定
4. localAlloc：根据数据流对一个 BasicBlock 内的指令进行寄存器分配
5. allocForLoc：每一条指令进行寄存器分配
6. allocRegFor：根据数据流决定为当前 Temp 分配哪一个寄存器
"""

class BruteRegAlloc(RegAlloc):
    def __init__(self, emitter: RiscvAsmEmitter) -> None:
        super().__init__(emitter)
        self.bindings = {}
        self.maxNumParams = 8
        for reg in emitter.allocatableRegs:
            reg.used = False

    def accept(self, graph: CFG, info: SubroutineInfo) -> None:
        self.numArgs = info.numArgs
        self.functionParams = []
        self.callerSavedRegs = {}
        subEmitter = RiscvSubroutineEmitter(self.emitter, info)
        
        for index in range(min(self.numArgs, self.maxNumParams)):
            self.bind(Temp(index), Riscv.ArgRegs[index])
            subEmitter.emitStoreToStack(Riscv.ArgRegs[index])
            
        available_block = graph.findAvailableBlock()
        for bb in graph.iterator():
            # you need to think more here
            # maybe we don't need to alloc regs for all the basic blocks
            if bb.id not in available_block: continue
            if bb.label is not None:
                subEmitter.emitLabel(bb.label)
            self.localAlloc(bb, subEmitter)
        subEmitter.emitFunc()

    def bind(self, temp: Temp, reg: Reg):
        reg.used = True
        self.bindings[temp.index] = reg
        reg.occupied = True
        reg.temp = temp

    def unbind(self, temp: Temp): #解除所有caller-saved寄存器与临时变量的绑定关系
        if temp.index in self.bindings:
            self.bindings[temp.index].occupied = False
            self.bindings.pop(temp.index)
            
    def callerParamCount(self):
        return len(self.functionParams)

    def localAlloc(self, bb: BasicBlock, subEmitter: RiscvSubroutineEmitter):
        for reg in self.emitter.allocatableRegs:
            reg.occupied = False

        # in step9, you may need to think about how to store callersave regs here
        for loc in bb.allSeq():
            subEmitter.emitComment(str(loc.instr))

            self.allocForLoc(loc, subEmitter)

        for tempindex in bb.liveOut:
            if tempindex in self.bindings:
                subEmitter.emitStoreToStack(self.bindings.get(tempindex))

        if (not bb.isEmpty()) and (bb.kind is not BlockKind.CONTINUOUS):
            self.allocForLoc(bb.locs[len(bb.locs) - 1], subEmitter)
        
        self.bindings.clear()  

    def allocForLoc(self, loc: Loc, subEmitter: RiscvSubroutineEmitter):
        instr = loc.instr
        srcRegs: list[Reg] = []
        dstRegs: list[Reg] = []

        for i in range(len(instr.srcs)):
            temp = instr.srcs[i]
            if isinstance(temp, Reg):
                srcRegs.append(temp)
            else:
                srcRegs.append(self.allocRegFor(temp, True, loc.liveIn, subEmitter))

        for i in range(len(instr.dsts)):
            temp = instr.dsts[i]
            if isinstance(temp, Reg):
                dstRegs.append(temp)
            else:
                dstRegs.append(self.allocRegFor(temp, False, loc.liveIn, subEmitter))
        # instr.fillRegs(dstRegs, srcRegs)
        # subEmitter.emitAsm(instr)
        if instr.kind == InstrKind.PARAM:
            self.allocForParam(instr, srcRegs, subEmitter)
        elif instr.kind == InstrKind.CALL:
            self.allocForCall(instr, srcRegs, dstRegs, subEmitter)
        else:
            instr.fillRegs(dstRegs, srcRegs)
            subEmitter.emitAsm(instr)

    def allocForParam(self, instr: BackendInstr, srcRegs: list[Reg], subEmitter: RiscvSubroutineEmitter):
        # 保存前八个参数到寄存器中
        if self.callerParamCount() < self.maxNumParams:
            reg = Riscv.ArgRegs[self.callerParamCount()]
            # 将寄存器解绑, 稍后恢复
            if reg.occupied:
                subEmitter.emitStoreToStack(reg)
                self.callerSavedRegs[reg] = reg.temp
                self.unbind(reg.temp)
            subEmitter.emitReg(reg, srcRegs[0])
        self.functionParams.append(instr.srcs[0])

    def allocForCall(self, instr: BackendInstr, srcRegs: list[Reg], dstRegs: list[Reg], subEmitter: RiscvSubroutineEmitter):
        # 调用前保存 caller-saved 寄存器
        for reg in Riscv.CallerSaved:
            if reg.occupied:
                subEmitter.emitStoreToStack(reg)
                self.callerSavedRegs[reg] = reg.temp
                self.unbind(reg.temp)

        # 保存多余的参数到栈中
        if self.callerParamCount() > self.maxNumParams:
            for (index, temp) in enumerate(self.functionParams[self.maxNumParams:][::-1]):
                subEmitter.emitStoreParamToStack(temp, index)
            subEmitter.emitNative(instr)
            subEmitter.emitRestoreStackPointer(4 * (self.callerParamCount() - self.maxNumParams))
        else:
            # breakpoint()
            subEmitter.emitNative(instr)
        self.functionParams = []

        # 调用后恢复 caller-saved 寄存器
        for reg, temp in self.callerSavedRegs.items():
            # 返回值寄存器不需要恢复, 否则会覆盖
            if reg != Riscv.A0:
                self.bind(temp, reg)
                subEmitter.emitLoadFromStack(reg, temp)
        self.callerSavedRegs = {}

    def allocRegFor(
        self, temp: Temp, isRead: bool, live: set[int], subEmitter: RiscvSubroutineEmitter
    ):
        if temp.index in self.bindings:
            return self.bindings[temp.index]

        for reg in self.emitter.allocatableRegs:
            if (not reg.occupied) or (not reg.temp.index in live):
                subEmitter.emitComment(
                    "  allocate {} to {}  (read: {}):".format(
                        str(temp), str(reg), str(isRead)
                    )
                )
                if isRead:
                    # subEmitter.emitLoadFromStack(reg, temp)
                    # 如果是存储在栈上的参数, 利用 FP 从栈中加载
                    if (self.maxNumParams <= temp.index < self.numArgs):
                        subEmitter.emitLoadParamFromStack(reg, temp.index)
                    # 否则, 利用 SP 从栈中加载
                    else:
                        subEmitter.emitLoadFromStack(reg, temp)
                if reg.occupied:
                    self.unbind(reg.temp)
                self.bind(temp, reg)
                return reg

        reg = self.emitter.allocatableRegs[
            random.randint(0, len(self.emitter.allocatableRegs) - 1)
        ]
        subEmitter.emitStoreToStack(reg)
        subEmitter.emitComment("  spill {} ({})".format(str(reg), str(reg.temp)))
        self.unbind(reg.temp)
        self.bind(temp, reg)
        subEmitter.emitComment(
            "  allocate {} to {} (read: {})".format(str(temp), str(reg), str(isRead))
        )
        if isRead:
            subEmitter.emitLoadFromStack(reg, temp)
        return reg
