
from PySched import *

                
if __name__ == '__main__':
    
    s = Schedule()
    
    A = Task(s,duration=2,release=0,duedate=8,demand=1,label='i')
    B = Task(s,duration=2,release=1,duedate=10,demand=1,label='j')
            
    res = Resource(s, 'A', [A,B])
    
    outfile = open('tex/ex/disjunctive_example.tex', 'w')
    
    s.latex(outfile, animated=True, rows=[[A],[B]])
    
    s.save()
    
    A << B
    s.latex(outfile, animated=True, rows=[[A],[B]])
    
    s.restore()
    
    s.save()
    
    B << A
    s.latex(outfile, animated=True, rows=[[A],[B]])
    
    s.restore()
    
    
    
    
    s.setLatestCompletion(A,4)
    s.setLatestCompletion(B,5)
    
    
    outfile = open('tex/ex/disjunctive.tex', 'w')
    
    s.latex(outfile, animated=True, width=20, horizon=5)
    s.latex(outfile, animated=True, mandatory=True, precedences=False, windows=False, width=20, horizon=5)
    
    tt = Timetabling(res)
    
    while True:
        if not tt.propagate():
            break

        s.latex(outfile, animated=True, precedences=False, width=20, horizon=5, pruning=True)
        s.latex(outfile, animated=True, mandatory=True, precedences=False, width=20, horizon=5)
        
        
    d = Disjunct(A,B)
    
    while True:
        
        s.save()
        
        if not d.propagate():
            break

        s.latex(outfile, animated=True, precedences=True, width=20, horizon=5, rows=[[A],[B]], pruning=True)
        
    s.latex(outfile, animated=True, precedences=True, width=20, horizon=5, rows=[[A],[B]], pruning=True, stop='-')
    
    

    
    


    
        
#
# 45/93                                            xxx
# 21/45                   xxx                                             xxx
# 9/21         xxx                     xxx                     xxx
# 3/9    xxx         xxx         xxx         xxx         xxx         xxx
# 0/3 xxx   xxx   xxx   xxx   xxx   xxx   xxx   xxx   xxx   xxx   xxx   xxx
#




