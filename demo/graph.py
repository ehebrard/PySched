
from PySched import *
import examples
                
if __name__ == '__main__':
    
    s = Schedule()
    
    A = Task(s,duration=(4,6),demand=1,label='A')
    B = Task(s,duration=3,demand=1,label='B')
    C = Task(s,duration=5,demand=1,label='C')
    D = Task(s,duration=(4,6),demand=1,label='D')
    E = Task(s,duration=(3,7),demand=1,label='E')
    F = Task(s,duration=5,demand=1,label='F')
            
    A << B
    A << C
    B << D
    B << E
    C << F

    span = 20
    s.setMakespanUB(span)
    
    rA = Resource(s,'A',[A,B,C])
    rB = Resource(s,'B',[D,E,F])
    rmap = dict([(start(t),t.resourceName()) for t in s.tasks] + [(end(t),t.resourceName()) for t in s.tasks])
    
    s.graph.latex(outfile=open('tex/ex/d_precedence_graph.tex', 'w'), width=9, sep=2.5, resources=rmap)
    s.graph.latex(outfile=open('tex/ex/d_precedence_graph_lb.tex', 'w'), width=9, sep=2.5, lb=[start(D)], resources=rmap)
    s.graph.latex(outfile=open('tex/ex/d_precedence_graph_ub.tex', 'w'), width=9, sep=2.5, ub=[end(A)], resources=rmap)
    s.latex(outfile=open('tex/ex/d_scheduling_instance.tex', 'w'), precedences=True, windows=False)
    
    examples.writeFile('graph_demo', [['d_scheduling_instance'], ['d_precedence_graph']])
    examples.writeFile('graph_bound_demo', [['d_precedence_graph', 'd_precedence_graph_lb', 'd_precedence_graph_ub']])
#




