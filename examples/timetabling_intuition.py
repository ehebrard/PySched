
from PySched import *

def intervals():
    
    s = Schedule()
    
    A = Task(s,duration=10,release=0,duedate=15,demand=2,label='A')
    
    res = Resource(s, 'A', [A], 2)
    
    outfile = open('tex/ex/timetabling_intuition.tex', 'w')
    
    s.latex(outfile, animated=True, width=15, horizon=15, windows=True, shift='Left')
    s.latex(outfile, animated=True, width=15, horizon=15, windows=True, shift='Right')
    s.latex(outfile, animated=True, width=15, horizon=15, windows=True, mandatory=True)
    s.latex(outfile, animated=True, width=15, horizon=15, windows=True, mandatory=True, profile=[res], profp=-.2, stop='-')
    
    
    
def cumulative_example():
    s = Schedule()
    
    A = Task(s,duration=2,release=0,duedate=3,demand=2,label='A')
    B = Task(s,duration=2,release=0,duedate=4,demand=1,label='B')
    C = Task(s,duration=2,release=0,duedate=4,demand=3,label='C')
            
    res = Resource(s, 'A', [A,B,C], capacity=3)
    
    outfile = open('tex/ex/cumulative_timetabling.tex', 'w')
    
    s.latex(outfile, animated=True, width=20, horizon=5)
    s.latex(outfile, animated=True, mandatory=True, precedences=False, windows=False, width=20, horizon=5, profile=[res])

    tt = Timetabling(res)

    while True:

        s.save()

        if not tt.propagate():
            break

        s.latex(outfile, animated=True, precedences=False, width=20, horizon=5, pruning=True)
        s.latex(outfile, animated=True, mandatory=True, precedences=False, width=20, horizon=5)
        s.latex(outfile, animated=True, mandatory=True, precedences=False, width=20, horizon=5, profile=[res])


def simple_example():
    s = Schedule()
    
    A = Task(s,duration=2,release=0,duedate=3,demand=1,label='A')
    B = Task(s,duration=2,release=0,duedate=4,demand=1,label='B')
            
    res = Resource(s, 'A', [A,B])
    
    outfile = open('tex/ex/unary_timetabling.tex', 'w')
    
    s.latex(outfile, animated=True, width=20, horizon=5)
    s.latex(outfile, animated=True, mandatory=True, precedences=False, windows=False, width=20, horizon=5)
    
    tt = Timetabling(res)
    
    while True:
        
        s.save()
        
        if not tt.propagate():
            break

        s.latex(outfile, animated=True, precedences=False, width=20, horizon=5, pruning=True)
        s.latex(outfile, animated=True, mandatory=True, precedences=False, width=20, horizon=5)
        # s.latex(outfile, animated=True, mandatory=True, precedences=False, width=20, horizon=5, profile=[res])
        
        
def algo_example():
    s = Schedule()
    
    C = Task(s,duration=800,release=0,duedate=1000,demand=1,label='C')
    E = Task(s,duration=300,release=300,duedate=800,demand=1,label='E')
    A = Task(s,duration=500,release=700,duedate=1500,demand=1,label='A')
    D = Task(s,duration=200,release=0,duedate=400,demand=1,label='D')
    F = Task(s,duration=600,release=1000,duedate=2000,demand=1,label='F')
    B = Task(s,duration=400,release=200,duedate=1200,demand=1,label='B')
            
    res = Resource(s, 'A', [A,B,C,D,E,F])
    
    outfile = open('tex/ex/unary_timetabling_algo.tex', 'w')
    
    span = 2000
    
    s.latex(outfile, animated=True, rows=[[A],[B],[C],[D],[E],[F]], width=span/100, horizon=span)
    s.latex(outfile, animated=True, mandatory=True, precedences=False, windows=False, rows=[[A],[B],[C],[D],[E],[F]], width=span/100, horizon=span, stop='-3')
    s.latex(outfile, animated=True, mandatory=True, precedences=False, windows=False, profile=[res], rows=[[A],[B],[C],[D],[E],[F]], width=span/100, horizon=span, stop='-',offset=1,profp=-2)
    
    # tt = Timetabling(res)
    #
    # while True:
    #     if not tt.propagate():
    #         break
    #
    #     s.latex(outfile, animated=True, precedences=False, width=20, horizon=5)
    #     s.latex(outfile, animated=True, mandatory=True, precedences=False, width=20, horizon=5)

                
if __name__ == '__main__':
    
    intervals()
    
    simple_example()
    
    cumulative_example()
    
    algo_example()
    


    
    


    
        
#
# 45/93                                            xxx
# 21/45                   xxx                                             xxx
# 9/21         xxx                     xxx                     xxx
# 3/9    xxx         xxx         xxx         xxx         xxx         xxx
# 0/3 xxx   xxx   xxx   xxx   xxx   xxx   xxx   xxx   xxx   xxx   xxx   xxx
#




