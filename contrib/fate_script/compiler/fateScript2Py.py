import copy
import sys
import json
import os
import re
from os import path
from contrib.fate_script.compiler.Compiler import StringBuilder as sb
from contrib.fate_script.compiler.Compiler import *
from contrib.fate_script.compiler.parser.FmlParser import *
from contrib.fate_script.compiler.parser.FmlLexer import *
from contrib.fate_script.compiler.parser.FmlVisitor import *
from collections import defaultdict, namedtuple
from arch.api.utils import file_utils


class PyNode:
    def __init__(self, *args):
        self.elems = args
        self.children = []
        self.parent = None

    def add_child(self, *child):
        for c in child:
            c.parent = self
            self.children.append(c)

    #for a tree, this is preorder traversal
    def to_py(self):
        return "".join(self.elems) + "".join([c.to_py() for c in self.children])

class PyStmt(PyNode):
    def __init__(self, *args):
        super().__init__(*args)
        self.indent = 0

    def to_py(self):
        return "  " * self.indent + super().to_py()


class FmlVisitorBase(FmlVisitor):
    def _get_stmt(self, node):
        stmt = node
        while stmt.parentCtx is not None and not type(stmt) == FmlParser.StmtContext:
            stmt = stmt.parentCtx
        return stmt
    def _get_left_sibling(self, node):
        cnt = len(node.parentCtx.children)
        for i in range(cnt - 1):
            if node.parentCtx.children[i + 1] == node:
                return node.parentCtx.children[i]
        return None

    @staticmethod
    def _get_right_sibling(node):
        cnt = len(node.parentCtx.children)
        for i in range(1, cnt):
            if node.parentCtx.children[i - 1] == node:
                return node.parentCtx.children[i]
        return None

    def _get_site_name(self, node):
        text = node if isinstance(node, str) else node.getText()
        return text[2:-2].strip() if len(text) > 2 and text[:2] == "<<" else text

    def _get_encrypt_name(self, node):
        text = node if isinstance(node, str) else node.getText()
        #print("get enc name:{}".format(node))
        return "_enc_" + text[2:-2].strip() if len(text) > 2 and text[:2] == "[[" else  text

    def _visitSiteStmt(self, node):
        return self.visitChildren(node)

    def visitSite_assign_stmt(self, ctx: FmlParser.Site_assign_stmtContext):
        return self._visitSiteStmt(ctx)

    def visitSite_atom_stmt(self, ctx: FmlParser.Site_atom_stmtContext):
        return self._visitSiteStmt(ctx)

    def visitSite_for_stmt(self, ctx: FmlParser.Site_for_stmtContext):
        return self._visitSiteStmt(ctx)

    def visitSite_if_stmt(self, ctx: FmlParser.Site_if_stmtContext):
        return self._visitSiteStmt(ctx)

    def visitSite_while_stmt(self, ctx: FmlParser.Site_while_stmtContext):
        return self._visitSiteStmt(ctx)

SiteFlow = namedtuple("SiteFlow", ["action", "stmt", "from_site", "to_site", "tag"])
# class SiteFlow:
#     def __init__(self, action, stmt, from_site, to_site, tag):
#         self.action = action
#         self.stmt = stmt
#         self.from_site = from_site
#         self.to_site = to_site
#         self.tag = tag
#
#     def __hash__(self):
#         return hash(self.action) ^ hash(self.from_site) ^ hash(self.to_site) ^ hash(self.tag)

class FmlVisitorAnalytic(FmlVisitorBase):
    def __init__(self):
        super().__init__()
        self.cur_site_scope = None
        self.site_flow = []
        self.all_site = []



    def visitSite_init_stmt(self, ctx: FmlParser.Site_init_stmtContext):
        site_name = self._get_site_name(ctx.SITE_NAME())
        self.all_site.append(site_name)
        return self._visitSiteStmt(ctx)

    def visitSite_assign_stmt(self, ctx: FmlParser.Site_assign_stmtContext):
        ret = super().visitSite_assign_stmt(ctx)
        stmt = self._get_stmt(ctx)
        site_name = self._get_site_name(ctx.SITE_NAME(0))
        for i in range(len(ctx.atom())):
            if self._get_site_name(ctx.SITE_NAME(i)) != site_name:
                raise SyntaxError("only support assign in one site:" + str(ctx.atom(i).start))
            self.site_flow.append(SiteFlow("put", stmt, site_name, None, self._get_encrypt_name(ctx.atom(i).getText())))
        return ret

    def visitTerminal(self, node):
        left = self._get_left_sibling(node)
        # site定义，忽略
        if left is None : return
        right = self._get_right_sibling(node)
        # 函数调用，忽略
        if right is not None and isinstance(right, FmlParser.TrailerContext): return
        if node.symbol.type == FmlParser.SITE_NAME:
            stmt = self._get_stmt(node)
            site_name = self._get_site_name(node)
            if self.cur_site_scope is None:
                for s in self.all_site:
                    if s == site_name:
                        continue
                    self.site_flow.append(SiteFlow("get",stmt, site_name, s,self._get_encrypt_name(left.getText())))
            else:
                if site_name != self.cur_site_scope:
                    self.site_flow.append(SiteFlow("get", stmt, site_name, self.cur_site_scope, self._get_encrypt_name(left.getText())))

    def _visitSiteStmt(self, node):
        site_name = node.SITE_NAME()[0] if isinstance(node.SITE_NAME(), list) else node.SITE_NAME()
        self.cur_site_scope = self._get_site_name(site_name)
        super().visitChildren(node)
        self.cur_site_scope = None


class FmlVisitorPyTree(FmlVisitorBase):
    def __init__(self, tokenStream: CommonTokenStream, site_flow):
        super().__init__()
        self.init_tag = False
        self.cur_stmt_site = None
        self.cur_expr_encrypt = None
        self.indent = 0
        self.tokenStream = tokenStream
        # 消除连续的put
        self.site_flow = defaultdict(set)
        skipped = set()
        for i in range(len(site_flow)):
            if i in skipped: continue
            final = site_flow[i]
            if final.action != "put": continue
            for j in range(i+1, len(site_flow)):
                cur = site_flow[j]
                if cur.tag == final.tag and cur.from_site == final.from_site:
                    if cur.action == "get":
                        skipped.add(j)
                        break
                    else:
                        final = cur
            # 没被引用就不put了, 引用了多次也有可能
            for get in site_flow:
                if get.action == "get" and get.tag == final.tag and get.from_site == final.from_site:
                    s = final._replace(to_site = get.to_site)
                    self.site_flow[final.stmt].add(s)

        # 消除连续的get
        for i in range(len(site_flow)):
            final = site_flow[i]
            if final.action != "get": continue
            for j in range(i+1, len(site_flow)):
                cur = site_flow[j]

                #allow to get more than once, but only allow different site get from one site, not allow one site get from one site more than once
                if cur.tag == final.tag and cur.from_site == final.from_site and cur.to_site == final.to_site:
                    if cur.action == "get":
                        final = None
                    break
            if final is not None:
                self.site_flow[final.stmt].add(final)
    '''
    def visitSite_init_stmt(self, ctx: FmlParser.Site_init_stmtContext):
        ret = PyNode()
        if self._get_site_name((ctx.SITE_NAME())) == 'INIT':
            ret.add_child(PyNode("import sys"))
            ret.add_child(PyNode("\nfrom sklearn import metrics"))
            ret.add_child(PyNode("\nfrom contrib.fate_script import fate_script"))
            ret.add_child(PyNode("\nfrom contrib.fate_script.blas.blas import *"))
            ret.add_child(PyNode("\nfrom contrib.fate_script.utils.fate_script_transfer_variable import *"))
            ret.add_child(PyNode("\n__site__ = sys.argv[1]"))
            ret.add_child(PyNode("\n__job_id__ = sys.argv[2]"))
            ret.add_child(PyNode("\n__conf_path = sys.argv[3]"))
            ret.add_child(PyNode("\n__work_mode = sys.argv[4]"))

            #ret.add_child(PyNode("\n__site_conf__"))
            #ret.add_child(PyNode(" = "))
            #ret.add_child(self.visit(ctx.test()))
            ret.add_child(PyNode("\nfate_script.init(__job_id__, __conf_path, int(__work_mode))"))
            ret.add_child(PyNode("\ntransfer_variable = HeteroLRTransferVariable()"))
            ret.add_child(PyNode("\nfate_script.init_encrypt_operator()"))
        return ret
    '''
    def visitSite_assign_stmt(self, ctx: FmlParser.Site_assign_stmtContext):
        node = ctx
        site_name = node.SITE_NAME()[0] if isinstance(node.SITE_NAME(),list) else node.SITE_NAME()
        site_name = self._get_site_name(site_name)
        self.cur_stmt_site = site_name
        ret = PyNode()
        ret.add_child(PyNode("if __site__ == \"%s\":\n" % site_name))
        self.indent += 2
        stmt = PyStmt()
        stmt.indent = self.indent

        left = PyNode()
        self.cur_expr_encrypt = False
        # left value
        for i in range(3 * len(ctx.atom()) - 1):
            left.add_child(self.visit(ctx.children[i]))
        left_encrypt = self.cur_expr_encrypt
        self.cur_expr_encrypt = False
        right = PyNode()
        # right value
        for i in range(3 * len(ctx.atom()), len(ctx.children)):
            right.add_child(self.visit(ctx.children[i]))
        right_encrypt = self.cur_expr_encrypt

        stmt.add_child(left)
        if right_encrypt == left_encrypt:
            stmt.add_child(PyNode("="))
            stmt.add_child(right)
        elif left_encrypt:
            stmt.add_child(PyNode("= fate_script.tensor_encrypt("))
            stmt.add_child(right)
            stmt.add_child(PyNode(")"))
        else:
            stmt.add_child(PyNode("= fate_script.tensor_decrypt("))
            stmt.add_child(right)
            stmt.add_child(PyNode(")"))

        self.indent -= 2
        ret.add_child(stmt)
        return ret
    
    def _visitSiteStmt(self, node):
        # return self._get_site_stmt(node).children[-1].add_child(self.visitChildren(node))
        site_name = node.SITE_NAME()[0] if isinstance(node.SITE_NAME(),list) else node.SITE_NAME()
        site_name = self._get_site_name(site_name)
        self.cur_stmt_site = site_name
        ret = PyNode()
        ret.add_child(PyNode("if __site__ == \"%s\":\n" % site_name))
        self.indent += 2
        stmt = PyStmt()
        stmt.indent = self.indent
        stmt.add_child(self.visitChildren(node))
        self.indent -= 2
        ret.add_child(stmt)
        return ret

    def visitTerminal(self, node):
        if node.symbol.type == FmlParser.INDENT:
            self.indent += 2
            raw = ""
        elif node.symbol.type == FmlParser.DEDENT:
            self.indent -= 2
            raw = ""
        elif node.symbol.type == FmlParser.ENCRYPT_NAME:
            raw = self._get_encrypt_name(node)
            self.cur_expr_encrypt = True
        elif node.symbol.type == FmlParser.SITE_NAME:
            raw = ""
            # raw = self._get_site_name(node)
        elif node.symbol.type == Token.EOF:
            raw = ""
        else:
            raw = node.getText()

        hidden = self.tokenStream.getHiddenTokensToRight(node.symbol.tokenIndex, Token.HIDDEN_CHANNEL)
        if hidden is None:
            hidden = []
        return PyNode(raw + "".join([n.text for n in hidden]))

    def defaultResult(self):
        return PyNode()

    def aggregateResult(self, aggregate, next_result):
        if next_result is None:
            return aggregate
        if aggregate.parent is None:
            aggregate.add_child(next_result)
        else:
            aggregate.parent.add_child(next_result)
        return aggregate

    def visitStmt(self, ctx: FmlParser.StmtContext):
        stmt = PyStmt()
        stmt.indent = self.indent
        stmt.add_child(self.visitChildren(ctx))
        ret = PyNode()

        if self.init_tag == False:
            ret.add_child(PyNode("import sys"))
            ret.add_child(PyNode("\nfrom sklearn import metrics"))
            ret.add_child(PyNode("\nfrom contrib.fate_script import fate_script"))
            ret.add_child(PyNode("\nfrom contrib.fate_script.blas.blas import *"))
            ret.add_child(PyNode("\nfrom contrib.fate_script.utils.fate_script_transfer_variable import *"))
            ret.add_child(PyNode("\n__site__ = sys.argv[1]"))
            ret.add_child(PyNode("\n__job_id__ = sys.argv[2]"))
            ret.add_child(PyNode("\n__conf_path = sys.argv[3]"))
            ret.add_child(PyNode("\n__work_mode = sys.argv[4]"))
            ret.add_child(PyNode("\nfate_script.init(__job_id__, __conf_path, int(__work_mode))"))
            ret.add_child(PyNode("\ntransfer_variable = HeteroLRTransferVariable()"))
            ret.add_child(PyNode("\nfate_script.init_encrypt_operator()\n"))
            self.init_tag = True
    

        if self.site_flow[ctx]:
            gets = list(filter(lambda x:x.action=="get",self.site_flow[ctx]))
            if self.cur_stmt_site is not None and gets:
                if_stmt = PyStmt("if __site__ == \"%s\":\n" % self.cur_stmt_site)
                if_stmt.indent = self.indent
                ret.add_child(if_stmt)
            
            
            for put in gets:
                iter_suffix = ''
                s = PyStmt("%s=fate_script.get(transfer_variable.%s.name, (transfer_variable.%s.name + '.' + str(__job_id__)[-14:]%s), idx = 0) # to %s\n" % (put.tag, put.tag, put.tag, iter_suffix, put.to_site))
                s.indent = self.indent + (2 if self.cur_stmt_site is not None else 0)
                tmp = s.indent
                ret.add_child(s)
                s = PyStmt("print(\"%s finish getting %s from %s\")\n" % (put.to_site, put.tag, put.from_site))
                s.indent = tmp
                ret.add_child(s)
                if put.tag == 'paillier_pubkey':
                    s = PyStmt("fate_script.get_public_key(%s)\n" % (put.tag))
                    s.indent = tmp
                    ret.add_child(s)
        ret.add_child(stmt)
        for put in self.site_flow[ctx]:
            #judge if need to remote
            if put.action == "put":
                suffix = put.tag
                #TODO:need to recognize loop, and add suffix for loop variables, otherwise a variable of party G at loop 1 might transfer to party H at loop 2(under normal circumstances variable of party G at loop 1 shall only transfer to party H at loop 1)
                iter_suffix = ''
                ret.add_child(PyNode("\n"))
                s = PyStmt("fate_script.remote(%s, name =transfer_variable.%s.name, tag = (transfer_variable.%s.name + '.' + str(__job_id__)[-14:]%s), role = '%s', idx = 0) # from %s\n"
                           % (suffix, suffix, suffix, iter_suffix, put.to_site, put.from_site))
                s.indent = self.indent + (2 if self.cur_stmt_site is not None else 0)
                tmp = s.indent
                ret.add_child(s)
                s = PyStmt("print(\"%s finish remote %s(name:{}, tag:{}) to %s\".format(transfer_variable.%s.name, transfer_variable.%s.name + '.' + str(__job_id__)[-14:]%s))\n" %\
                           (put.from_site, put.tag, put.to_site, put.tag, put.tag, iter_suffix))
                s.indent = tmp
                ret.add_child(s)
        self.cur_stmt_site = None
        return ret


class FmlVisitor2Py(StringBuilderVisitor, FmlVisitorBase):
    def __init__(self, tokenStream: CommonTokenStream, site_flow):
        super().__init__()
        self.tokenStream = tokenStream
        self.lastToken = None
        self.site_flow_remote = defaultdict(set)
        for (var, target, src,a1,a2) in site_flow:
            self.site_flow_remote[(var,src)].add(target)

    def _visitSiteStmt(self, node):
        site_name = node.SITE_NAME()[0] if isinstance(node.SITE_NAME(),list) else node.SITE_NAME()
        ret = StringBuilder() + "if __site__ == \"%s\":\n" % site_name.getText()[2:-2].strip()
        for n in node.children:
           if hasattr(n, "symbol") and n.symbol.type == FmlParser.SITE_NAME:
               n.symbol.text = ""
        self.indentLevel += 2
        ret += self.indent() + super().visitChildren(node)
        self.indentLevel -= 2
        return ret

    def visitSite_assign_stmt(self, ctx: FmlParser.Site_assign_stmtContext):
        ret = sb()
        for i in range(len(ctx.atom())):
            var_name = ctx.atom(i).getText()
            site_name = self._get_site_name(ctx.SITE_NAME(i))
            for target in self.site_flow_remote[(var_name,site_name)]:
                ret += sb() + ";" + "federation.remote('%s','%s','%s')" % (self._get_encrypt_name(var_name), site_name, target)
        ret = super().visitSite_assign_stmt(ctx) + ret
        return ret

    def visitSite_init_stmt(self, ctx: FmlParser.Site_init_stmtContext):
        return StringBuilder() + "__site_conf__['%s']" % self._get_site_name(ctx.SITE_NAME()) + "=" + self.visit(ctx.test())

    def visitStmt(self, ctx: FmlParser.StmtContext):
        return self.indent() + self.visitChildren(ctx)

    def visitTerminal(self, node):
        if node.symbol.type == FmlParser.INDENT:
            self.indentLevel += 2
            raw = ""
        elif node.symbol.type == FmlParser.DEDENT:
            self.indentLevel -= 2
            raw = ""
        elif node.symbol.type == FmlParser.ENCRYPT_NAME:
            raw = self._get_encrypt_name(node)
        elif node.symbol.type == FmlParser.SITE_NAME:
            raw = ""
        else:
            raw = super().visitTerminal(node)

        self.lastToken = node
        hidden = self.tokenStream.getHiddenTokensToRight(node.symbol.tokenIndex, Token.HIDDEN_CHANNEL)
        if hidden is None:
            hidden = []
        return raw + "".join([n.text for n in hidden])


class TransferDict(dict):
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


def transfer_dict_gen(site_flow, algo_name):
    if site_flow is None:
        raise Exception("site_flow NULL error")
    action_idx = 0
    src_idx = 2
    dst_idx = 3
    var_idx = 4
    transfer_dict = defaultdict(defaultdict)

    for i in range(len(site_flow)):
        site_var = analy.site_flow[i]
        if site_var[src_idx] is None \
                or site_var[dst_idx] is None \
                or site_var[action_idx] != 'get':
            continue

        #当前只处理src为单个的情况
        if str(site_var[var_idx]) in transfer_dict.keys() \
                and 'dst' in transfer_dict[str(site_var[var_idx])].keys():
            if site_var[dst_idx] not in transfer_dict[str(site_var[var_idx])]['dst']:
                transfer_dict[str(site_var[var_idx])]['dst'].append(site_var[dst_idx])
        else:
            inner_dict = defaultdict()
            inner_dict['src'] = site_var[src_idx]
            inner_dict['dst'] = []
            inner_dict['dst'].append(site_var[dst_idx])
            transfer_dict[str(site_var[var_idx])] = inner_dict

    transfer_dict = {algo_name + 'TransferVariable' : transfer_dict}
    #json_path = (os.getcwd()) + "contrib/fate_script/conf/FateScriptTransferVar.json"
    json_path = file_utils.get_project_base_directory() + "/contrib/fate_script/conf/FateScriptTransferVar.json"
    with open(json_path, 'wt') as f:
        f.write(json.dumps(transfer_dict, sort_keys=False, indent=4, separators=(',', ':')))


def runtime_json_generator(site_flow, runtime_path):
    if site_flow is None:
        raise Exception("site_flow is None")

    with open(runtime_path) as runtime_conf:
        runtime_json = json.load(runtime_conf)
    with open(file_utils.get_project_base_directory() + "/contrib/fate_script/conf/route.json") as route_conf:
        route_json = json.load(route_conf)
    

    site_set = set()
    inner_dict = defaultdict()
    for i in range(len(site_flow)):
        if site_flow[i][2] is None or site_flow[i][3] is None:
            continue
        site_set.add(site_flow[i][2])
        site_set.add(site_flow[i][3])

    for i in range(len(site_set)):
        if list(site_set)[i] in route_json["cluster_a"]["role"]:
            cluster_type = "cluster_a"
        elif list(site_set)[i] in route_json["cluster_b"]["role"]:
            cluster_type = "cluster_b"
        else:
            raise Exception("role:{} is not in route.json".format(list(site_set)[i]))

        inner_dict[list(site_set)[i]] = [route_json[cluster_type]["party_id"]]

    for i in range(len(site_set)):
        if list(site_set)[i] in route_json["cluster_a"]["role"]:
            cluster_type = "cluster_a"
        elif list(site_set)[i] in route_json["cluster_b"]["role"]:
            cluster_type = "cluster_b"
        else:
            raise Exception("role:{} is not in route.json".format(list(site_set)[i]))
        
        runtime_json["local"]["role"] = list(site_set)[i]
        runtime_json["local"]["party_id"] = route_json[cluster_type]["party_id"]        
        runtime_json["role"] = inner_dict
        json_path = file_utils.get_project_base_directory() + "/contrib/fate_script/conf/" + list(site_set)[i] + "_runtime_conf.json"
        with open(json_path, "wt") as fout:
            fout.write(json.dumps(runtime_json, indent=4))


if __name__ == '__main__':
    script_path = sys.argv[1]
    #path = file_utils.get_project_base_directory() + "/contrib/fate_script/script/HeteroLR.fml"
    #path = file_utils.get_project_base_directory() + script_path
    path = sys.argv[1]
    algo_start_idx = 0 if '/' not in path else path.rindex('/') + 1
    algo_name = path[algo_start_idx : path.rindex('.')]
    inputStream = FileStream(path, encoding="utf8")
    lexer = FmlLexer(inputStream)   #create a lexer for inputStream
    stream = CommonTokenStream(lexer)   #create a token buffer for token storage
    parser = FmlParser(stream)     #create a parser to process the token in buffer storage
    tree = parser.file_input()      #start parsing the token stream based on the rule "file_input"

    # 打印语法信息
    printer = CompilerPrinter()
    lexer.symbolicNames = parser.symbolicNames

    #进行树状结构的转换
    analy = FmlVisitorAnalytic()
    analy.visit(tree)
    visitor_py_tree = FmlVisitorPyTree(stream, analy.site_flow)

    filename = file_utils.get_project_base_directory() + "/contrib/fate_script/fateScript.py"
    with open(filename, 'w') as f:
        f.write(visitor_py_tree.visit(tree).to_py())
    transfer_dict_gen(analy.site_flow, algo_name)
    runtime_json_generator(site_flow=analy.site_flow, runtime_path = file_utils.get_project_base_directory() + "/contrib/fate_script/conf/role_runtime_conf.json")



