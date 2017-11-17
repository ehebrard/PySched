
from PySched import *

                
if __name__ == '__main__':
    
    s = Schedule()
    
    # A = Task(s,duration=11,release=0,duedate=19,label='A')
    # B = Task(s,duration=6,release=0,duedate=10,label='B')
    # C = Task(s,duration=5,release=0,duedate=10,label='C')
    # D = Task(s,duration=5,release=0,duedate=10,label='D')
    # rows = [[A],[B],[C],[D]]
    
    
    A = Task(s,duration=4,release=0,duedate=20,demand=1,label='A')
    B = Task(s,duration=1,release=1,duedate=2,demand=4,label='B')
    C = Task(s,duration=1,release=0,duedate=3,demand=2,label='C')
    D = Task(s,duration=1,release=0,duedate=3,demand=2,label='D')
    E = Task(s,duration=1,release=2,duedate=3,demand=1,label='E')
    rows = [[A],[B],[C],[D],[E]]
    
    # A = Task(s,duration=11,release=6,duedate=25,label='A')
    # B = Task(s,duration=6,release=15,duedate=25,label='B')
    # C = Task(s,duration=5,release=15,duedate=25,label='C')
    # D = Task(s,duration=5,release=15,duedate=25,label='D')
    
    
    res = Resource(s, 'A', [A,B,C,D,E], capacity=4)
    ef = NuijtenCumulativeEdgeFinding(res)
    # ef = TimetableEdgeFinding(res)
    
    outfile = open('tex/ex/nuijten_cumulative_edgefinding.tex', 'w')
    
    s.latex(outfile, animated=True, width=20, horizon=20, rows=rows)
    
    while True:
        if not ef.propagate():
            break

        s.latex(outfile, animated=True, precedences=False, windows=True, width=20, horizon=20, rows=rows)
        

