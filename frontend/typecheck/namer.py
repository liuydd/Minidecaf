from typing import Protocol, TypeVar, cast

from frontend.ast.node import T, Node, NullType
from frontend.ast.tree import *
from frontend.ast.tree import T, ExpressionList, Parameter, Postfix
from frontend.ast.visitor import T, RecursiveVisitor, Visitor
from frontend.scope.globalscope import GlobalScope
from frontend.scope.scope import Scope, ScopeKind
from frontend.scope.scopestack import ScopeStack
from frontend.symbol.funcsymbol import FuncSymbol
from frontend.symbol.symbol import Symbol
from frontend.symbol.varsymbol import VarSymbol
from frontend.type.array import ArrayType
from frontend.type.type import DecafType
from utils.error import *
from utils.riscv import MAX_INT

"""
The namer phase: resolve all symbols defined in the abstract 
syntax tree and store them in symbol tables (i.e. scopes).
"""


class Namer(Visitor[ScopeStack, None]):
    def __init__(self) -> None:
        pass

    # Entry of this phase
    def transform(self, program: Program) -> Program:
        # Global scope. You don't have to consider it until Step 6.
        program.globalScope = GlobalScope
        ctx = ScopeStack(program.globalScope)
        # breakpoint()
        program.accept(self, ctx)
        return program

    def visitProgram(self, program: Program, ctx: ScopeStack) -> None:
        # Check if the 'main' function is missing
        if not program.hasMainFunc():
            raise DecafNoMainFuncError
        # print(program.functions())
        # for func in program.functions().values():
        for func in program.children:
            func.accept(self, ctx)

    def visitFunction(self, func: Function, ctx: ScopeStack) -> None:
        # func.body.accept(self, ctx)
        # print(GlobalScope.symbols)
        # print(func.__dict__)
        if ctx.isConflict(func.ident.value) or GlobalScope.lookup(func.ident.value):
            raise DecafDeclConflictError(func.ident.value)
        # breakpoint()
        newSymbol = FuncSymbol(func.ident.value, func.ret_t.type, ctx.currentScope())
        if func.params is not None:
            for param in func.params.children:
                newSymbol.addParaType(param.var_t.type)
        GlobalScope.declare(newSymbol)
        func.setattr("symbol", newSymbol)
        # print(func.__dict__)
        func_scope = Scope(ScopeKind.LOCAL)
        ctx.addScope(func_scope)
        # ctx.declare(newSymbol)
        func.params.accept(self, ctx)
        # func.body.accept(self, ctx, func.params)
        for child in func.body.children:
            child.accept(self, ctx)
        ctx.popScope()
            
    def visitParameter(self, param: Parameter, ctx: ScopeStack) -> None:
        if ctx.lookup(param.ident.value):
            raise DecafDeclConflictError(param.ident.value)
        newSymbol = VarSymbol(param.ident.value, param.var_t.type)
        ctx.declare(newSymbol)
        param.setattr("symbol", newSymbol)
        param.ident.setattr('type', newSymbol.type)
            
    def visitParameterList(self, params: ParameterList, ctx: ScopeStack) -> None:
        for child in params.children:
            child.accept(self, ctx)
            
    def visitExpressionList(self, exprlist: ExpressionList, ctx: ScopeStack) -> None:
        for child in exprlist.children:
            child.accept(self, ctx)
            
    def visitPostfix(self, postfix: Postfix, ctx: ScopeStack) -> None:
        # breakpoint()
        # if not ctx.lookup(postfix.ident.value): raise DecafUndefinedVarError(postfix.ident.value)
        if ctx.lookup_top(postfix.ident.value):
            raise DecafBadFuncCallError(postfix.ident.value)
        func = GlobalScope.lookup(postfix.ident.value)
        if not func or not func.isFunc: raise DecafUndefinedVarError(postfix.ident.value)
        if func.parameterNum != len(postfix.exprlist):
            raise DecafBadFuncCallError(postfix.ident.value)
        postfix.ident.setattr('symbol', func)
        postfix.setattr('type', func.type)
        for expr in postfix.exprlist:
            expr.accept(self, ctx)
        

    def visitBlock(self, block: Block, ctx: ScopeStack) -> None:
        # for child in block:
        #     child.accept(self, ctx)
        block_scope = Scope(ScopeKind.LOCAL)
        ctx.addScope(block_scope)
        for child in block:
            child.accept(self, ctx)
        ctx.popScope()

    def visitReturn(self, stmt: Return, ctx: ScopeStack) -> None:
        stmt.expr.accept(self, ctx)

    """
    def visitFor(self, stmt: For, ctx: Scope) -> None:

    1. Open a local scope for stmt.init.
    2. Visit stmt.init, stmt.cond, stmt.update.
    3. Open a loop in ctx (for validity checking of break/continue)
    4. Visit body of the loop.
    5. Close the loop and the local scope.
    """
    def visitFor(self, stmt: For, ctx: ScopeStack) -> None:
        for_scope = Scope(ScopeKind.LOCAL)
        ctx.addScope(for_scope)
        if not stmt.init is NULL: stmt.init.accept(self, ctx)
        if not stmt.cond is NULL: stmt.cond.accept(self, ctx)
        if not stmt.update is NULL: stmt.update.accept(self, ctx)
        ctx.openLoop()
        stmt.body.accept(self, ctx)
        ctx.closeLoop()
        ctx.popScope()

    def visitIf(self, stmt: If, ctx: ScopeStack) -> None:
        stmt.cond.accept(self, ctx)
        stmt.then.accept(self, ctx)

        # check if the else branch exists
        if not stmt.otherwise is NULL:
            stmt.otherwise.accept(self, ctx)

    def visitWhile(self, stmt: While, ctx: ScopeStack) -> None:
        stmt.cond.accept(self, ctx)
        ctx.openLoop()
        stmt.body.accept(self, ctx)
        ctx.closeLoop()

    def visitBreak(self, stmt: Break, ctx: ScopeStack) -> None:
        """
        You need to check if it is currently within the loop.
        To do this, you may need to check 'visitWhile'.

        if not in a loop:
            raise DecafBreakOutsideLoopError()
        """
        if not ctx.inLoop():
            raise DecafBreakOutsideLoopError()
        # raise NotImplementedError

    """
    def visitContinue(self, stmt: Continue, ctx: Scope) -> None:
    
    1. Refer to the implementation of visitBreak.
    """
    def visitContinue(self, stmt: Continue, ctx: ScopeStack) -> None:
        if not ctx.inLoop():
            raise DecafBreakOutsideLoopError()

    def visitDeclaration(self, decl: Declaration, ctx: ScopeStack) -> None:
        """
        1. Use ctx.lookup to find if a variable with the same name has been declared.
        2. If not, build a new VarSymbol, and put it into the current scope using ctx.declare.
        3. Set the 'symbol' attribute of decl.
        4. If there is an initial value, visit it.
        """
        if ctx.isConflict(decl.ident.value): raise DecafUndefinedVarError(f"Variable {decl.ident.value} already declared")
        # if ctx.lookup(decl.ident.value) is not None: raise DecafUndefinedVarError(f"Variable {decl.ident.value} already declared")
        else:
            new_symbol = VarSymbol(decl.ident.value, decl.var_t)
            if ctx.isGlobalScope():
                new_symbol.isGlobal = True
                if decl.init_expr:
                    new_symbol.initValue = decl.init_expr.value
                elif new_symbol.type == INT:
                    decl.init_expr = IntLiteral(0)
            ctx.declare(new_symbol)
            decl.setattr("symbol", new_symbol)
            if decl.init_expr is not None:
                decl.init_expr.accept(self, ctx)

    def visitAssignment(self, expr: Assignment, ctx: ScopeStack) -> None:
        """
        1. Refer to the implementation of visitBinary.
        """
        expr.lhs.accept(self, ctx)
        expr.rhs.accept(self, ctx)
        # raise NotImplementedError


    def visitUnary(self, expr: Unary, ctx: ScopeStack) -> None:
        expr.operand.accept(self, ctx)

    def visitBinary(self, expr: Binary, ctx: ScopeStack) -> None:
        expr.lhs.accept(self, ctx)
        expr.rhs.accept(self, ctx)

    def visitCondExpr(self, expr: ConditionExpression, ctx: ScopeStack) -> None:
        """
        1. Refer to the implementation of visitBinary.
        """
        expr.cond.accept(self, ctx)
        expr.then.accept(self, ctx)
        expr.otherwise.accept(self, ctx)
        # raise NotImplementedError

    def visitIdentifier(self, ident: Identifier, ctx: ScopeStack) -> None:
        """
        1. Use ctx.lookup to find the symbol corresponding to ident.
        2. If it has not been declared, raise a DecafUndefinedVarError.
        3. Set the 'symbol' attribute of ident.
        """
        symbol = ctx.lookup(ident.value)
        if symbol is None:
            raise DecafUndefinedVarError(ident.value)
        ident.setattr("symbol", symbol)
        # raise NotImplementedError

    def visitIntLiteral(self, expr: IntLiteral, ctx: ScopeStack) -> None:
        value = expr.value
        expr.setattr('type', INT)
        if value > MAX_INT:
            raise DecafBadIntValueError(value)
