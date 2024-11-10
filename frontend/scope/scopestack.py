from typing import Optional

from frontend.symbol.symbol import Symbol

from .scope import Scope

class stackOverflow(Exception):
    ...
    

class ScopeStack:
    defaultstackdepth = 256
    def __init__(self, globalscope: Scope, stackdepth: int=defaultstackdepth):
        self.globalscope = globalscope
        self.stack = [globalscope]
        self.stackdepth = stackdepth
        self.loopdepth = 0
    
    #得到当前的Scope, 即栈顶的Scope
    def currentScope(self):
        if not self.stack: return self.globalscope
        return self.stack[-1]
    
    #向栈内加入一个Scope
    def addScope(self, scope: Scope) -> None:
        if len(self.stack) < self.stackdepth:
            self.stack.append(scope)
        else:
            raise stackOverflow
    
    #弹出栈顶的Scope
    def popScope(self):
        self.stack.pop()
    
    #看当前的Scope中是否有重复的名称
    def isConflict(self, name: str) -> Optional[Symbol]:
        if self.currentScope().containsKey(name):
            return self.currentScope().get(name)
        return None
    
    # To declare a symbol.
    def declare(self, symbol: Symbol) -> None:
        self.currentScope().declare(symbol)

    # To check if this is a global scope.
    def isGlobalScope(self) -> bool:
        return self.currentScope().isGlobalScope()
    
    # To get a symbol if declared in the scope
    def lookup(self, name: str) -> Optional[Symbol]:
        s = len(self.stack)
        for d in range(s-1, -1, -1):
            if self.stack[d].containsKey(name):
                return self.stack[d].get(name)
        return None
    
    def lookup_top(self, name: str) -> Optional[Symbol]:
        return self.currentScope().lookup(name)
    
    #检查 break/continue 语句是否在一个循环内
    def openLoop(self) -> None:
        self.loopdepth += 1
    
    def closeLoop(self) -> None:
        self.loopdepth -= 1
    
    def inLoop(self) -> bool:
        return self.loopdepth > 0
        