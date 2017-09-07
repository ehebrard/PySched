
from PySched import *
from tools.branch_and_bound import *

                
if __name__ == '__main__':
    
    s = Schedule()
    
    A = Task(s,duration=3,demand=1,label='A')
    B = Task(s,duration=1,demand=1,label='B')
    C = Task(s,duration=5,demand=1,label='C')
    D = Task(s,duration=2,demand=1,label='D')
    E = Task(s,duration=2,demand=1,label='E')
    F = Task(s,duration=5,demand=1,label='F')
    G = Task(s,duration=3,demand=1,label='G')
    H = Task(s,duration=5,demand=1,label='H')
    I = Task(s,duration=1,demand=1,label='I')
    
    A < D
    D < G
    B < E
    E < H
    C < F
    F < I
    
    resA = Resource(s, 'A', [A,B,C])
    resB = Resource(s, 'B', [D,E,F])
    resC = Resource(s, 'C', [G,H,I])
    

    span = 25

    s.setMakespanUB(span)
    
    rows = s.rowsFromPaths()
    
    outfile = open('tex/ex/branchandbound_trace.tex', 'w')
    
    random.seed(12345)
    bnb = BranchAndBound(s)
    
    printsched = lambda x : s.latex(outfile=outfile, ub=True, lb=True, animated=True, windows=False, rows=rows, width=span, horizon=span, decisions=[(d.tasks[int(d.value())], d.tasks[1-int(d.value())]) for d in x.decisions])
    bnb.search(limit=None, executeOnNode=printsched)

    import examples
    examples.writeFile('branchandbound_example', 'branchandbound_trace')

