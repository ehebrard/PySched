
from PySched import *

                
if __name__ == '__main__':
    
    s = Schedule()
    
    A = Task(s,duration=11,release=2,duedate=27,label='A')
    B = Task(s,duration=10,release=0,duedate=26,label='B')
    C = Task(s,duration=2,release=7,duedate=23,label='C')
    
    
    res = Resource(s, 'A', [A,B,C])
    nl = NotFirstNotLast(res)
    
    outfile = open('tex/ex/not_first.tex', 'w')
    

    s.latex(outfile, animated=True, precedences=False, windows=True, width=27, horizon=27, rows=[[A],[B],[C]])
    
    while True:
        
        s.save()
        
        if not nl.propagate():
            break

        s.latex(outfile, animated=True, precedences=False, windows=True, width=27, horizon=27, rows=[[A],[B],[C]], pruning=True)
        
    import examples
    examples.writeFile('not_first_example', 'not_first', scale=.4)
