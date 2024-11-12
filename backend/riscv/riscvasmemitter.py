from typing import Sequence, Tuple

from utils.error import IllegalArgumentException
from utils.label.label import Label, LabelKind
from utils.riscv import Riscv, RvBinaryOp, RvUnaryOp
from utils.tac.reg import Reg
from utils.tac.tacfunc import TACFunc
from utils.tac.tacinstr import *
from utils.tac.tacinstr import Assign
from utils.tac.tacvisitor import TACVisitor
from utils.asmcodeprinter import AsmCodePrinter
from utils.tac.backendinstr import BackendInstr
from ..subroutineinfo import SubroutineInfo
from frontend.ast.tree import *

"""
RiscvAsmEmitter: an AsmEmitter for RiscV
"""


class RiscvAsmEmitter():
    def __init__(
        self,
        allocatableRegs: list[Reg],
        callerSaveRegs: list[Reg],
        globalVars: dict[str, Declaration],
    ):
        self.allocatableRegs = allocatableRegs
        self.callerSaveRegs = callerSaveRegs
        self.printer = AsmCodePrinter()
    
        # the start of the asm code
        # int step10, you need to add the declaration of global var here
        self.printer.println(".data")
        for symbol, decl in globalVars.items():
            self.printer.printGlobalVar(symbol, decl.getattr("symbol").initValue)
        self.printer.println("")
        
        self.printer.println(".text")
        self.printer.println(".global main")
        self.printer.println("")

    # transform tac instrs to RiscV instrs
    # collect some info which is saved in SubroutineInfo for SubroutineEmitter
    def selectInstr(self, func: TACFunc) -> tuple[list[str], SubroutineInfo]:
        info = SubroutineInfo(func.entry, func.numArgs)

        selector: RiscvAsmEmitter.RiscvInstrSelector = (
            RiscvAsmEmitter.RiscvInstrSelector(func.entry)
        )
        # print(selector.__dict__)
        for instr in func.getInstrSeq():
            # print(instr.__dict__)
            instr.accept(selector)

        # breakpoint()
        return (selector.seq, info)

    # return all the string stored in asmcodeprinter
    def emitEnd(self):
        return self.printer.close()

    class RiscvInstrSelector(TACVisitor):
        def __init__(self, entry: Label) -> None:
            self.entry = entry
            self.seq = []

        def visitOther(self, instr: TACInstr) -> None:
            raise NotImplementedError("RiscvInstrSelector visit{} not implemented".format(type(instr).__name__))

        def visitAssign(self, instr: Assign) -> None:
            # print("here")
            self.seq.append(Riscv.Move(instr.dst, instr.src))
            
        def visitParam(self, instr: Param) -> None:
            self.seq.append(Riscv.Param(instr.param))

        def visitCall(self, instr: Call) -> None:
            self.seq.append(Riscv.Call(instr.label))
            self.seq.append(Riscv.Move(instr.param, Riscv.A0))

        # in step11, you need to think about how to deal with globalTemp in almost all the visit functions. 
        def visitReturn(self, instr: Return) -> None:
            if instr.value is not None:
                self.seq.append(Riscv.Move(Riscv.A0, instr.value))
            else:
                self.seq.append(Riscv.LoadImm(Riscv.A0, 0))
            self.seq.append(Riscv.JumpToEpilogue(self.entry))

        def visitMark(self, instr: Mark) -> None:
            self.seq.append(Riscv.RiscvLabel(instr.label))

        def visitLoadImm4(self, instr: LoadImm4) -> None:
            self.seq.append(Riscv.LoadImm(instr.dst, instr.value))
            
        def visitLoadAddress(self, instr: LoadAddress) -> None:
            self.seq.append(Riscv.LoadAddress(instr.symbol.name, instr.dsts[0]))
        
        def visitLoadData(self, instr: LoadData) -> None:
            self.seq.append(Riscv.LoadData(instr.dsts[0], instr.srcs[0], instr.offset)) 
                
        def visitStoreData(self, instr: StoreData) -> None:
            self.seq.append(Riscv.StoreData(instr.srcs[0], instr.srcs[1], instr.offset))

        def visitUnary(self, instr: Unary) -> None:
            op = {
                TacUnaryOp.NEG: RvUnaryOp.NEG,
                TacUnaryOp.BITNOT: RvUnaryOp.NOT,
                TacUnaryOp.LOGICNOT: RvUnaryOp.SEQZ,
                # You can add unary operations here.
            }[instr.op]
            self.seq.append(Riscv.Unary(op, instr.dst, instr.operand))

        def visitBinary(self, instr: Binary) -> None:
            """
            For different tac operation, you should translate it to different RiscV code
            A tac operation may need more than one RiscV instruction
            """
            if instr.op == TacBinaryOp.LOR: #||
                self.seq.append(Riscv.Binary(RvBinaryOp.OR, instr.dst, instr.lhs, instr.rhs))
                self.seq.append(Riscv.Unary(RvUnaryOp.SNEZ, instr.dst, instr.dst))
            elif instr.op == TacBinaryOp.AND: #&&
                self.seq.append(Riscv.Unary(RvUnaryOp.SNEZ, instr.dst, instr.lhs))
                self.seq.append(Riscv.Binary(RvBinaryOp.SUB, instr.dst, Riscv.ZERO, instr.dst))
                self.seq.append(Riscv.Binary(RvBinaryOp.AND, instr.dst, instr.dst, instr.rhs))
                self.seq.append(Riscv.Unary(RvUnaryOp.SNEZ, instr.dst, instr.dst))
            elif instr.op == TacBinaryOp.LEQ: #<=
                self.seq.append(Riscv.Binary(RvBinaryOp.SGT, instr.dst, instr.lhs, instr.rhs))
                self.seq.append(Riscv.Unary(RvUnaryOp.SEQZ, instr.dst, instr.dst))
            elif instr.op == TacBinaryOp.GEQ: #>=
                self.seq.append(Riscv.Binary(RvBinaryOp.SLT, instr.dst, instr.lhs, instr.rhs))
                self.seq.append(Riscv.Unary(RvUnaryOp.SEQZ, instr.dst, instr.dst))
            elif instr.op == TacBinaryOp.EQU: #==
                self.seq.append(Riscv.Binary(RvBinaryOp.SUB, instr.dst, instr.lhs, instr.rhs))
                self.seq.append(Riscv.Unary(RvUnaryOp.SEQZ, instr.dst, instr.dst))
            elif instr.op == TacBinaryOp.NEQ: #!=
                self.seq.append(Riscv.Binary(RvBinaryOp.SUB, instr.dst, instr.lhs, instr.rhs))
                self.seq.append(Riscv.Unary(RvUnaryOp.SNEZ, instr.dst, instr.dst))
            else:
                op = {
                    TacBinaryOp.ADD: RvBinaryOp.ADD,
                    TacBinaryOp.SUB: RvBinaryOp.SUB,
                    TacBinaryOp.MUL: RvBinaryOp.MUL,
                    TacBinaryOp.DIV: RvBinaryOp.DIV,
                    TacBinaryOp.MOD: RvBinaryOp.REM,
                    TacBinaryOp.SLT: RvBinaryOp.SLT,
                    TacBinaryOp.SGT: RvBinaryOp.SGT,
                    # TacBinaryOp.ASSIGN: RvBinaryOp.MV,
                    # You can add binary operations here.
                }[instr.op]
                self.seq.append(Riscv.Binary(op, instr.dst, instr.lhs, instr.rhs))

        def visitCondBranch(self, instr: CondBranch) -> None:
            self.seq.append(Riscv.Branch(instr.cond, instr.label))
        
        def visitBranch(self, instr: Branch) -> None:
            self.seq.append(Riscv.Jump(instr.target))

        # in step9, you need to think about how to pass the parameters and how to store and restore callerSave regs
        # in step11, you need to think about how to store the array 
"""
RiscvAsmEmitter: an SubroutineEmitter for RiscV
"""

class RiscvSubroutineEmitter():
    def __init__(self, emitter: RiscvAsmEmitter, info: SubroutineInfo) -> None:
        self.info = info
        self.printer = emitter.printer
        
        # + 8 is for the RA and S0 reg 
        self.nextLocalOffset = 4 * len(Riscv.CalleeSaved) + + self.info.size + 8
        
        # the buf which stored all the NativeInstrs in this function
        self.buf: list[BackendInstr] = []

        # from temp to int
        # record where a temp is stored in the stack
        self.offsets = {}
        # print(info.funcLabel.__dict__)
        # print("before")
        # print(self.printer.debug_buffer())
        # print("end before")
        self.printer.printLabel(info.funcLabel)
        # print("after")
        # print(self.printer.debug_buffer())
        # print("end after")

        # in step9, step11 you can compute the offset of local array and parameters here

    def emitComment(self, comment: str) -> None:
        # you can add some log here to help you debug
        pass
        # print(comment)
    
    def emitNative(self, instr: BackendInstr):
        self.buf.append(instr)
        
    def emitReg(self, dst: Reg, src: Temp):
        self.buf.append(Riscv.Move(dst, src))
        
    # restore stack after calling a function
    def emitRestoreStackPointer(self, offset:int) -> None:
        self.buf.append(Riscv.SPAdd(offset))
    
    # store some param to stack
    def emitStoreParamToStack(self, src: Temp, index: int) -> None:
        self.buf.append(Riscv.SPAdd(-4))
        self.buf.append(Riscv.NativeLoadWord(Riscv.T0, Riscv.SP, self.offsets[src.index] + 4 * index + 4))
        self.buf.append(Riscv.NativeStoreWord(Riscv.T0, Riscv.SP, 0))

    # load some param from stack
    def emitLoadParamFromStack(self, dst: Reg, index: int) -> None:
        self.buf.append(Riscv.NativeLoadWord(dst, Riscv.FP, 4 * (index - 8)))
    
    # store some temp to stack
    # usually happen when reaching the end of a basicblock
    # in step9, you need to think about the fuction parameters here
    def emitStoreToStack(self, src: Reg) -> None:
        if src.temp.index not in self.offsets:
            self.offsets[src.temp.index] = self.nextLocalOffset
            self.nextLocalOffset += 4
        self.buf.append(
            Riscv.NativeStoreWord(src, Riscv.SP, self.offsets[src.temp.index])
        )

    # load some temp from stack
    # usually happen when using a temp which is stored to stack before
    # in step9, you need to think about the fuction parameters here
    def emitLoadFromStack(self, dst: Reg, src: Temp):
        if src.index not in self.offsets:
            raise IllegalArgumentException()
        else:
            self.buf.append(
                Riscv.NativeLoadWord(dst, Riscv.SP, self.offsets[src.index])
            )

    # add a NativeInstr to buf
    # when calling the fuction emitAsm, all the instr in buf will be transformed to RiscV code
    def emitAsm(self, instr: BackendInstr):
        self.buf.append(instr)

    def emitLabel(self, label: Label):
        self.buf.append(Riscv.RiscvLabel(label))

    
    def emitFunc(self):
        self.printer.printComment("start of prologue")
        self.printer.printInstr(Riscv.SPAdd(-self.nextLocalOffset))
        self.printer.printInstr(Riscv.NativeStoreWord(Riscv.RA, Riscv.SP, 4 * len(Riscv.CalleeSaved) + self.info.size))
        self.printer.printInstr(Riscv.NativeStoreWord(Riscv.FP, Riscv.SP, 4 * len(Riscv.CalleeSaved) + self.info.size + 4))
        self.printer.printInstr(Riscv.FPAdd(self.nextLocalOffset))

        # in step9, you need to think about how to store RA here
        # you can get some ideas from how to save CalleeSaved regs
        for i in range(len(Riscv.CalleeSaved)):
            if Riscv.CalleeSaved[i].isUsed():
                self.printer.printInstr(
                    Riscv.NativeStoreWord(Riscv.CalleeSaved[i], Riscv.SP, 4 * i + self.info.size)
                )

        self.printer.printComment("end of prologue")
        self.printer.println("")

        self.printer.printComment("start of body")

        # in step9, you need to think about how to pass the parameters here
        # you can use the stack or regs

        # using asmcodeprinter to output the RiscV code
        for instr in self.buf:
            # print(instr.__dict__)
            self.printer.printInstr(instr)

        self.printer.printComment("end of body")
        self.printer.println("")

        self.printer.printLabel(
            Label(LabelKind.TEMP, self.info.funcLabel.name + Riscv.EPILOGUE_SUFFIX)
        )
        self.printer.printComment("start of epilogue")
        # self.printer.printInstr(Riscv.SPAdd(-self.nextLocalOffset))
        self.printer.printInstr(Riscv.NativeLoadWord(Riscv.RA, Riscv.SP, 4 * len(Riscv.CalleeSaved) + self.info.size))
        self.printer.printInstr(Riscv.NativeLoadWord(Riscv.FP, Riscv.SP, 4 * len(Riscv.CalleeSaved) + self.info.size + 4))
        # self.printer.printInstr(Riscv.FPAdd(self.nextLocalOffset))

        for i in range(len(Riscv.CalleeSaved)):
            if Riscv.CalleeSaved[i].isUsed():
                self.printer.printInstr(
                    Riscv.NativeLoadWord(Riscv.CalleeSaved[i], Riscv.SP, 4 * i + self.info.size)
                )

        self.printer.printInstr(Riscv.SPAdd(self.nextLocalOffset))
        self.printer.printComment("end of epilogue")
        self.printer.println("")

        self.printer.printInstr(Riscv.NativeReturn())
        self.printer.println("")
