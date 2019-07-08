
from PySched import *
import PySched.examples as ex
                
if __name__ == '__main__':
    
    s = Schedule()
    
    s.closure = s.graph.FloydWarshall
    
    A = Task(s,duration=4,demand=1,label='A')
    B = Task(s,duration=3,demand=1,label='B')
    C = Task(s,duration=5,demand=1,label='C')
    D = Task(s,duration=4,demand=1,label='D')
            
    s.startBeforeEnd(A,B)
    s.startBeforeEnd(B,A)
    s.startBeforeEnd(C,B)
    s.startBeforeEnd(B,C)
    s.startBeforeEnd(C,D)
    s.startBeforeEnd(D,C)
    s.startBeforeEnd(A,D)
    s.startBeforeEnd(D,A)

    span = 100
    s.setMakespanUB(span)
    
    
    s.graph.printMatrix()
    
    s.propagate()
    
    s.graph.printMatrix()
    
    s.graph.latex(outfile=open('tex/ex/d_precedence_graph.tex', 'w'), width=9, sep=3)
    
    ex.writeFile('graph_demo', [['d_precedence_graph']])
#


#
# from PySched import *
#
#
# if __name__ == '__main__':
#
#
#     s = Schedule()
#
#     s.closure = s.graph.FloydWarshall
#
#     A = Task(s,duration=(4,6),demand=1,label='A')
#     B = Task(s,duration=3,demand=1,label='B')
#     C = Task(s,duration=5,demand=1,label='C')
#     D = Task(s,duration=(4,6),demand=1,label='D')
#
#
#     s.startBeforeEnd(A,B)
#     s.startBeforeEnd(B,A)
#     s.startBeforeEnd(C,B)
#     s.startBeforeEnd(B,C)
#     s.startBeforeEnd(C,D)
#     s.startBeforeEnd(D,C)
#     s.startBeforeEnd(A,D)
#     s.startBeforeEnd(D,A)
#
#     span = 100
#     s.setMakespanUB(span)
#
#     s.graph.printMatrix()
#
#     s.propagate()
#
#     s.graph.printMatrix()

