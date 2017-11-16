
from PySched import *
import examples
                
if __name__ == '__main__':
    outfile = open('tex/ex/d_timetabling.tex', 'w')
    
    s = Schedule()
    
    A = Task(s,duration=8,release=0,duedate=26,demand=1,label='A')
    B = Task(s,duration=6,release=1,duedate=10,demand=4,label='B')
    C = Task(s,duration=6,release=6,duedate=15,demand=4,label='C')
    D = Task(s,duration=8,release=7,duedate=20,demand=3,label='D')
    E = Task(s,duration=7,release=5,duedate=17,demand=2,label='E')
    F = Task(s,duration=9,release=5,duedate=25,demand=4,label='F')
            
    res = Resource(s, 'A', [A,B,C,D,E,F], capacity=7)
    
    A << F
    
    s.latex(outfile, animated=True, precedences=False, rows=[[A,F],[E],[B],[C],[D]])
    s.latex(outfile, animated=True, mandatory=True, profile=[res], precedences=False, profp=-.000001, rows=[[A,F],[E],[B],[C],[D]])
    
    tt = Timetabling(res)
    
    s.save()
    while True:
        if not tt.propagate():
            break

        s.latex(outfile, animated=True, precedences=False, pruning=True, rows=[[A,F],[E],[B],[C],[D]])
        s.latex(outfile, animated=True, mandatory=True, profile=[res], precedences=False, profp=-.000001, rows=[[A,F],[E],[B],[C],[D]])
        
        s.save()

    examples.writeFile('timetabling_demo', 'd_timetabling', scale=.3)



