
from PySched import *

                
if __name__ == '__main__':
    
    s = Schedule()
    
    A = Task(s,duration=4,release=0,duedate=25,label='A')
    B = Task(s,duration=3,release=1,duedate=9,label='B')
    C = Task(s,duration=3,release=1,duedate=9,label='C')
    D = Task(s,duration=5,release=7,duedate=14,label='D')
    
    
    res = Resource(s, 'A', [A,B,C,D])
    ef = EdgeFinding(res)
    
    outfile = open('tex/ex/edge_finding.tex', 'w')
    
    s.latex(outfile, animated=True, width=25, horizon=25)
    
    while True:
        
        s.save()
        
        if not ef.propagate():
            break

        s.latex(outfile, animated=True, precedences=True, windows=True, width=25, horizon=25, rows=[[A],[B],[C],[D]], pruning=True, decisions=False)
        # s.latex(outfile, animated=True, precedences=False, windows=True, width=25, horizon=25, rows=[[A],[B],[C],[D]], pruning=True)
        
    import examples
    examples.writeFile('edge_finding_example', 'edge_finding')
