from frontend.ast.node import Optional, NullType
from frontend.ast.tree import Continue, Function, Optional, Parameter, Postfix
from frontend.ast import node
from frontend.ast.tree import *
from frontend.ast.visitor import Visitor
from frontend.symbol.varsymbol import VarSymbol
from frontend.type.array import ArrayType
from utils.label.blocklabel import BlockLabel
from utils.label.funclabel import FuncLabel
from utils.tac import tacop
from utils.tac.temp import Temp
from utils.tac.tacinstr import *
from utils.tac.tacfunc import TACFunc
from utils.tac.tacprog import TACProg
from utils.tac.tacvisitor import TACVisitor


"""
The TAC generation phase: translate the abstract syntax tree into three-address code.
"""


class LabelManager:
    """
    A global label manager (just a counter).
    We use this to create unique (block) labels accross functions.
    """

    def __init__(self):
        self.nextTempLabelId = 0

    def freshLabel(self) -> BlockLabel:
        self.nextTempLabelId += 1
        return BlockLabel(str(self.nextTempLabelId))


class TACFuncEmitter(TACVisitor):
    """
    Translates a minidecaf (AST) function into low-level TAC function.
    """

    def __init__(
        self, entry: FuncLabel, numArgs: int, labelManager: LabelManager
    ) -> None:
        self.labelManager = labelManager
        self.func = TACFunc(entry, numArgs)
        self.visitLabel(entry)
        self.nextTempId = 0

        self.continueLabelStack = []
        self.breakLabelStack = []

    # To get a fresh new temporary variable.
    def freshTemp(self) -> Temp:
        temp = Temp(self.nextTempId)
        self.nextTempId += 1
        return temp

    # To get a fresh new label (for jumping and branching, etc).
    def freshLabel(self) -> Label:
        return self.labelManager.freshLabel()

    # To count how many temporary variables have been used.
    def getUsedTemp(self) -> int:
        return self.nextTempId

    # In fact, the following methods can be named 'appendXXX' rather than 'visitXXX'.
    # E.g., by calling 'visitAssignment', you add an assignment instruction at the end of current function.
    def visitAssignment(self, dst: Temp, src: Temp) -> Temp:
        self.func.add(Assign(dst, src))
        return src

    def visitParam(self, value: Temp) -> None:
        self.func.add(Param(value))
        
    def visitCall(self, label: Label) -> Temp:
        temp = self.freshTemp()
        self.func.add(Call(temp, label))
        return temp

    def visitLoad(self, value: Union[int, str]) -> Temp:
        temp = self.freshTemp()
        self.func.add(LoadImm4(temp, value))
        return temp

    def visitUnary(self, op: UnaryOp, operand: Temp) -> Temp:
        temp = self.freshTemp()
        self.func.add(Unary(op, temp, operand))
        return temp

    def visitUnarySelf(self, op: UnaryOp, operand: Temp) -> None:
        self.func.add(Unary(op, operand, operand))

    def visitBinary(self, op: BinaryOp, lhs: Temp, rhs: Temp) -> Temp:
        temp = self.freshTemp()
        self.func.add(Binary(op, temp, lhs, rhs))
        return temp

    def visitBinarySelf(self, op: BinaryOp, lhs: Temp, rhs: Temp) -> None:
        self.func.add(Binary(op, lhs, lhs, rhs))

    def visitBranch(self, target: Label) -> None:
        self.func.add(Branch(target))

    def visitCondBranch(self, op: CondBranchOp, cond: Temp, target: Label) -> None:
        self.func.add(CondBranch(op, cond, target))

    def visitReturn(self, value: Optional[Temp]) -> None:
        self.func.add(Return(value))

    def visitLabel(self, label: Label) -> None:
        self.func.add(Mark(label))

    def visitMemo(self, content: str) -> None:
        self.func.add(Memo(content))

    def visitRaw(self, instr: TACInstr) -> None:
        self.func.add(instr)

    def visitEnd(self) -> TACFunc:
        if (len(self.func.instrSeq) == 0) or (not self.func.instrSeq[-1].isReturn()):
            self.func.add(Return(None))
        self.func.tempUsed = self.getUsedTemp()
        return self.func

    # To open a new loop (for break/continue statements)
    def openLoop(self, breakLabel: Label, continueLabel: Label) -> None:
        self.breakLabelStack.append(breakLabel)
        self.continueLabelStack.append(continueLabel)

    # To close the current loop.
    def closeLoop(self) -> None:
        self.breakLabelStack.pop()
        self.continueLabelStack.pop()

    # To get the label for 'break' in the current loop.
    def getBreakLabel(self) -> Label:
        return self.breakLabelStack[-1]

    # To get the label for 'continue' in the current loop.
    def getContinueLabel(self) -> Label:
        return self.continueLabelStack[-1]


class TACGen(Visitor[TACFuncEmitter, None]):
    # Entry of this phase
    def transform(self, program: Program) -> TACProg:
        labelManager = LabelManager()
        tacFuncs = []
        tacGlobalVars = program.globalVars()
        for funcName, astFunc in program.functions().items(): #遍历每个函数?
            # in step9, you need to use real parameter count
            emitter = TACFuncEmitter(FuncLabel(funcName), len(astFunc.params.children), labelManager)
            for child in astFunc.params.children:
                child.accept(self, emitter)
            astFunc.body.accept(self, emitter) #调用不同的visit函数
            tacFuncs.append(emitter.visitEnd())
        return TACProg(tacFuncs, tacGlobalVars)

    def visitBlock(self, block: Block, mv: TACFuncEmitter) -> None:
        for child in block:
            child.accept(self, mv)

    def visitPostfix(self, postfix: Postfix, mv: TACFuncEmitter) -> None:
        # print("visitPostfix")
        for expr in postfix.exprlist.children:
            expr.accept(self, mv)
        for expr in postfix.exprlist.children:
            mv.visitParam(expr.getattr("val"))
        postfix.setattr('val', mv.visitCall(FuncLabel(postfix.ident.value)))

    def visitParameter(self, param: Parameter, mv: TACFuncEmitter) -> None:
        # print("visitparameter")
        param.getattr("symbol").temp = mv.freshTemp()

    def visitReturn(self, stmt: Return, mv: TACFuncEmitter) -> None:
        stmt.expr.accept(self, mv)
        mv.visitReturn(stmt.expr.getattr("val"))

    def visitBreak(self, stmt: Break, mv: TACFuncEmitter) -> None:
        mv.visitBranch(mv.getBreakLabel())
        
    def visitContinue(self, stmt: Continue, mv: TACFuncEmitter) -> None:
        mv.visitBranch(mv.getContinueLabel())

    def visitIdentifier(self, ident: Identifier, mv: TACFuncEmitter) -> None:
        """
        1. Set the 'val' attribute of ident as the temp variable of the 'symbol' attribute of ident.
        """
        symbol = ident.getattr("symbol")
        # print(symbol.__dict__)
        # breakpoint()
        ident.setattr("val", symbol.temp)
        # print("symbol.temp:", symbol.temp)
        # raise NotImplementedError

    def visitDeclaration(self, decl: Declaration, mv: TACFuncEmitter) -> None:
        """
        1. Get the 'symbol' attribute of decl.
        2. Use mv.freshTemp to get a new temp variable for this symbol.
        3. If the declaration has an initial value, use mv.visitAssignment to set it.
        """
        symbol = decl.getattr("symbol")
        new_temp = mv.freshTemp()
        symbol.temp = new_temp
        if decl.init_expr is not NULL:
            init_temp = decl.init_expr.accept(self, mv)
            mv.visitAssignment(new_temp, decl.init_expr.getattr("val")) #?
            
        # raise NotImplementedError

    def visitAssignment(self, expr: Assignment, mv: TACFuncEmitter) -> None:
        """
        1. Visit the right hand side of expr, and get the temp variable of left hand side.
        2. Use mv.visitAssignment to emit an assignment instruction.
        3. Set the 'val' attribute of expr as the value of assignment instruction.
        """
        expr.rhs.accept(self, mv)
        # breakpoint()
        lhs_symbol = expr.lhs.getattr("symbol").temp
        # print(lhs_symbol)
        rhs_symbol = expr.rhs.getattr("val")
        # print(lhs_symbol)
        rhs_temp = mv.visitAssignment(lhs_symbol, rhs_symbol)
        expr.setattr("val", rhs_temp)
        # raise NotImplementedError

    def visitIf(self, stmt: If, mv: TACFuncEmitter) -> None:
        stmt.cond.accept(self, mv)

        if stmt.otherwise is NULL: #没有else分支
            skipLabel = mv.freshLabel()
            mv.visitCondBranch(
                tacop.CondBranchOp.BEQ, stmt.cond.getattr("val"), skipLabel
            )
            stmt.then.accept(self, mv)
            mv.visitLabel(skipLabel)
        else:
            skipLabel = mv.freshLabel()
            exitLabel = mv.freshLabel()
            mv.visitCondBranch(
                tacop.CondBranchOp.BEQ, stmt.cond.getattr("val"), skipLabel
            )
            stmt.then.accept(self, mv)
            mv.visitBranch(exitLabel)
            mv.visitLabel(skipLabel)
            stmt.otherwise.accept(self, mv)
            mv.visitLabel(exitLabel)

    def visitWhile(self, stmt: While, mv: TACFuncEmitter) -> None:
        beginLabel = mv.freshLabel()
        loopLabel = mv.freshLabel()
        breakLabel = mv.freshLabel()
        mv.openLoop(breakLabel, loopLabel)

        mv.visitLabel(beginLabel)
        stmt.cond.accept(self, mv)
        mv.visitCondBranch(tacop.CondBranchOp.BEQ, stmt.cond.getattr("val"), breakLabel)

        stmt.body.accept(self, mv)
        mv.visitLabel(loopLabel)
        mv.visitBranch(beginLabel)
        mv.visitLabel(breakLabel)
        mv.closeLoop()

    def visitFor(self, stmt: For, mv: TACFuncEmitter) -> None:
        beginLabel = mv.freshLabel()
        loopLabel = mv.freshLabel()
        breakLabel = mv.freshLabel()
        mv.openLoop(breakLabel, loopLabel)
        
        stmt.init.accept(self, mv)
        mv.visitLabel(beginLabel)
        stmt.cond.accept(self, mv)
        mv.visitCondBranch(tacop.CondBranchOp.BEQ, stmt.cond.getattr("val"), breakLabel)
        stmt.body.accept(self, mv)
        
        mv.visitLabel(loopLabel)
        stmt.update.accept(self, mv)
        
        mv.visitBranch(beginLabel)
        mv.visitLabel(breakLabel)
        mv.closeLoop()
        
        

    def visitUnary(self, expr: Unary, mv: TACFuncEmitter) -> None:
        expr.operand.accept(self, mv)

        op = {
            node.UnaryOp.Neg: tacop.TacUnaryOp.NEG,
            node.UnaryOp.BitNot: tacop.TacUnaryOp.BITNOT,
            node.UnaryOp.LogicNot: tacop.TacUnaryOp.LOGICNOT,
            # You can add unary operations here.
        }[expr.op]
        expr.setattr("val", mv.visitUnary(op, expr.operand.getattr("val")))

    def visitBinary(self, expr: Binary, mv: TACFuncEmitter) -> None:
        expr.lhs.accept(self, mv)
        expr.rhs.accept(self, mv)

        op = {
            node.BinaryOp.Add: tacop.TacBinaryOp.ADD,
            node.BinaryOp.LogicOr: tacop.TacBinaryOp.LOR,
            node.BinaryOp.Sub: tacop.TacBinaryOp.SUB,
            node.BinaryOp.Mul: tacop.TacBinaryOp.MUL,
            node.BinaryOp.Div: tacop.TacBinaryOp.DIV,
            node.BinaryOp.Mod: tacop.TacBinaryOp.MOD,
            node.BinaryOp.LT: tacop.TacBinaryOp.SLT,
            node.BinaryOp.GT: tacop.TacBinaryOp.SGT,
            node.BinaryOp.LE: tacop.TacBinaryOp.LEQ,
            node.BinaryOp.GE: tacop.TacBinaryOp.GEQ,
            node.BinaryOp.EQ: tacop.TacBinaryOp.EQU,
            node.BinaryOp.NE: tacop.TacBinaryOp.NEQ,
            node.BinaryOp.LogicAnd: tacop.TacBinaryOp.AND,
            node.BinaryOp.Assign: tacop.TacBinaryOp.ASSIGN,
            # You can add binary operations here.
        }[expr.op]
        expr.setattr(
            "val", mv.visitBinary(op, expr.lhs.getattr("val"), expr.rhs.getattr("val"))
        )

    def visitCondExpr(self, expr: ConditionExpression, mv: TACFuncEmitter) -> None:
        """
        1. Refer to the implementation of visitIf and visitBinary.
        """
        expr.cond.accept(self, mv)
        temp = mv.freshTemp()
        skipLabel = mv.freshLabel()
        exitLabel = mv.freshLabel()
        mv.visitCondBranch(
            tacop.CondBranchOp.BEQ, expr.cond.getattr("val"), skipLabel
        )
        expr.then.accept(self, mv)
        mv.visitAssignment(temp, expr.then.getattr("val"))
        mv.visitBranch(exitLabel) #jump
        mv.visitLabel(skipLabel)
        expr.otherwise.accept(self, mv)
        mv.visitAssignment(temp, expr.otherwise.getattr("val"))
        mv.visitLabel(exitLabel)
        expr.setattr("val", temp)
        # raise NotImplementedError

    def visitIntLiteral(self, expr: IntLiteral, mv: TACFuncEmitter) -> None:
        expr.setattr("val", mv.visitLoad(expr.value))
