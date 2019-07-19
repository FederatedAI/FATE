import sys
from os import path
sys.path.append(path.dirname( path.dirname( path.abspath(__file__)))+"/compiler/runtime/Python3/src")
from io import StringIO
from antlr4 import Token
from antlr4.Token import CommonToken

from antlr4.tree.Trees import Trees


class StringBuilder(object):
    def __init__(self):
        self.buf = []

    def __add__(self, other):
        if other is None:
            return self
        if isinstance(other, StringBuilder):
            self.buf.append(other.toString())
        elif isinstance(other, str):
            self.buf.append(other)
        else:
            raise TypeError(str(other) + ":" + str(type(other)))
        return self

    def __iadd__(self, other):
        return self.__add__(other)

    def toString(self):
        return "".join(self.buf)


class StringBuilderVisitor:
    def __init__(self):
        self.indentLevel = 0

    def indent(self, n=None):
        return StringBuilder() + "    " * (self.indentLevel + n if n is not None else self.indentLevel)

    def visitIndent(self, tree, level=1):
        ret = StringBuilder()
        self.indentLevel += level
        ret += self.visit(tree)
        self.indentLevel -= level
        return ret

    def visit(self, ctx):
        if ctx is None:
            raise ValueError
        if isinstance(ctx, CommonToken):
            return ctx.text
        if isinstance(ctx, list):
            ret = StringBuilder()
            for c in ctx:
                ret += self.visit(c)
            return ret
        return ctx.accept(self)

    def defaultResult(self):
        return StringBuilder()

    def visitTerminal(self, node):
        if node.symbol.type == Token.EOF:
            return ""
        return node.getText()

    def aggregateResult(self, aggregate, nextResult):
        ret = aggregate if aggregate is not None else StringBuilder()
        if nextResult is not None:
            ret += nextResult
        return ret


class CompilerPrinter(object):
    def __init__(self):
        self.printDepth = 0

    def escapeWhitespace(self, s, escape_spaces):
        with StringIO() as buf:
            for c in s:
                if c == ' ' and escape_spaces:
                    buf.write('\u00B7')
                elif c == '\t':
                    buf.write("\\t")
                elif c == '\n':
                    buf.write("\\n")
                elif c == '\r':
                    buf.write("\\r")
                else:
                    buf.write(c)
            return buf.getvalue()

    def getTreeString(self, t, ruleNames=None, parser=None):
        if parser is not None:
            ruleNames = parser.ruleNames
        s = self.escapeWhitespace(Trees.getNodeText(t, ruleNames), False)
        c = t.__class__.__name__
        if c.endswith("Context"):
            s = c[:-7]
        if t.getChildCount() == 0:
            return s

        with StringIO() as buf:
            buf.write(u"\n")
            buf.write(u"  " * self.printDepth)
            buf.write(u"")
            buf.write(s)
            buf.write(u' ')
            self.printDepth += 1
            for i in range(0, t.getChildCount()):
                if i > 0:
                    buf.write(u' ')
                buf.write(self.getTreeString(t.getChild(i), ruleNames, parser))
            self.printDepth -= 1
            buf.write(u"")
            return buf.getvalue()

    def getCollectionDefault(self, k, values):
        return values[k] if len(values) > k else k

    def getTokenString(self, tok, lexer):
        with StringIO() as buf:
            buf.write(u"[@")
            buf.write(str(tok.tokenIndex))
            buf.write(u",")
            buf.write(str(tok.start))
            buf.write(u":")
            buf.write(str(tok.stop))
            buf.write(u"='")
            txt = tok.text
            if txt is not None:
                txt = txt.replace(u"\n", u"\\n")
                txt = txt.replace(u"\r", u"\\r")
                txt = txt.replace(u"\t", u"\\t")
            else:
                txt = u"<no text>"
            buf.write(txt)
            buf.write(u"',<")
            buf.write(self.getCollectionDefault(tok.type, lexer.symbolicNames))
            buf.write(u">")
            if tok.channel > 0:
                buf.write(u",channel=")
                buf.write(self.getCollectionDefault(tok.channel, lexer.channelNames))
            buf.write(u",")
            buf.write(str(tok.line))
            buf.write(u":")
            buf.write(str(tok.column))
            buf.write(u"]")
            return buf.getvalue()
