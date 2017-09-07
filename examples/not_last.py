
from PySched import *

                
if __name__ == '__main__':
    
    s = Schedule()
    
    A = Task(s,duration=11,release=0,duedate=25,label='A')
    B = Task(s,duration=10,release=1,duedate=27,label='B')
    C = Task(s,duration=2,release=4,duedate=20,label='C')
    
    
    res = Resource(s, 'A', [A,B,C])
    nl = NotFirstNotLast(res)
    
    outfile = open('tex/ex/not_last.tex', 'w')
    

    s.latex(outfile, animated=True, precedences=False, windows=True, width=27, horizon=27, rows=[[A],[B],[C]])
    
    while True:
        
        s.save()
        
        if not nl.propagate():
            break

        s.latex(outfile, animated=True, precedences=False, windows=True, width=27, horizon=27, rows=[[A],[B],[C]], pruning=True)
        
    import examples
    examples.writeFile('not_last_example', 'not_last', scale=.4)
