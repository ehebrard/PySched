
from PySched import *
import examples
                
if __name__ == '__main__':
    outfile = open('tex/ex/d_edgefinding.tex', 'w')
    
    s = Schedule()
    
    A = Task(s,duration=4,release=0,duedate=25,label='A')
    B = Task(s,duration=3,release=1,duedate=9,label='B')
    C = Task(s,duration=3,release=1,duedate=9,label='C')
    D = Task(s,duration=5,release=7,duedate=14,label='D')
    
    
    res = Resource(s, 'A', [A,B,C,D])
    ef = EdgeFinding(res)
    
    s.latex(outfile, animated=True)
    
    while True:
        
        s.save()
        
        if not ef.propagate():
            break

        s.latex(outfile, animated=True, precedences=True, windows=True, rows=[[A],[B],[C],[D]], pruning=True, decisions=False)

    examples.writeFile('edgefinding_demo', 'd_edgefinding', scale=.45)