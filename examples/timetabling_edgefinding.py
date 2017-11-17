
from PySched import *

                
if __name__ == '__main__':
    
    s = Schedule()
    
    A = Task(s,duration=10,release=0,duedate=17,demand=2,label='A')
    B = Task(s,duration=4,release=5,duedate=12,demand=1,label='B')
    C = Task(s,duration=4,release=5,duedate=12,demand=1,label='C')
    D = Task(s,duration=5,release=5,duedate=12,demand=2,label='A')
            
    res = Resource(s, 'A', [A,B,C,D], capacity=10)
    resNoD = Resource(s, 'B', [A,B,C], capacity=3)
    
    outfile = open('tex/ex/incomparable.tex', 'w')
    
    s.latex(outfile, width=17, animated=True, precedences=False, rows=[[A],[B],[C]])
    s.latex(outfile, width=17, animated=True, mandatory=True, profile=[resNoD], precedences=False, profp=-1, rows=[[A],[B],[C]])


    s.save()
    s.setLatestCompletion(A,12)

    outfile = open('tex/ex/efview.tex', 'w')
    
    s.latex(outfile, width=17, animated=True, precedences=False, rows=[[A],[B],[C]], pruning=True, offset=2, horizon=17)
    
    # s.setEarliestStart(A,12)

    outfile = open('tex/ex/eefview.tex', 'w')
    
    s.latex(outfile, width=17, animated=True, precedences=False, rows=[[D],[C],[B]], pruning=True, offset=3, horizon=17)


    s = Schedule()
    
    A = Task(s,duration=7,release=0,duedate=8,demand=1,label='A')
    B = Task(s,duration=7,release=9,duedate=17,demand=1,label='B')
    C = Task(s,duration=4,release=4,duedate=12,demand=2,label='C')
    D = Task(s,duration=4,release=4,duedate=12,demand=2,label='D')
    E = Task(s,duration=4,release=4,duedate=12,demand=2,label='E')
    F = Task(s,duration=4,release=4,duedate=12,demand=1,label='F')

    
    resA = Resource(s, 'A', [A,B,C,D,E,F], capacity=4)
    # resB = Resource(s, 'B', [A,B], capacity=4)
    
    outfile = open('tex/ex/ttef.tex', 'w')
    
    s.latex(outfile, width=17, animated=True, precedences=False, rows=[[A],[B],[C],[D],[E],[F]], offset=4)
    # s.latex(outfile, width=17, animated=True, mandatory=True, profile=[resA], precedences=False, profp=-.00001, rows=[[A],[B],[C],[D],[E],[F]], offset=4)
    s.latex(outfile, width=17, animated=True, mandatory=True, precedences=False, rows=[[A],[B],[C],[D],[E],[F]], offset=4)
    
    ef = NuijtenCumulativeEdgeFinding(resA)
    # ef = TimetableEdgeFinding(res)
    
    outfile = open('tex/ex/test.tex', 'w')
    
    s.latex(outfile, animated=True, width=17, horizon=17)
    
    while True:
        if not ef.propagate():
            break

        s.latex(outfile, animated=True, precedences=False, windows=True, width=17, horizon=17)
        
        
    s = Schedule()
    
    A = Task(s,duration=5,release=0,duedate=8,demand=2,label='A')
    B = Task(s,duration=7,release=6,duedate=17,demand=1,label='B')
    C = Task(s,duration=4,release=0,duedate=14,demand=2,label='C')
    D = Task(s,duration=4,release=10,duedate=17,demand=3,label='D')
    E = Task(s,duration=7,release=4,duedate=15,demand=2,label='E')
    F = Task(s,duration=8,release=1,duedate=14,demand=1,label='F')

    
    resA = Resource(s, 'A', [A,B,C,D,E,F], capacity=4)
    # resB = Resource(s, 'B', [A,B], capacity=4)
    
    outfile = open('tex/ex/ttef_orig.tex', 'w')
    
    
    s.latex(outfile, width=17, animated=True, precedences=False, rows=[[A],[B],[C],[D],[E],[F]])
    s.latex(outfile, width=17, animated=True, mandatory=True, profile=[resA], precedences=False, profp=-.00001, rows=[[A],[B],[C],[D],[E],[F]], stop='-3')
    s.latex(outfile, width=17, animated=True, mandatory=False, profile=[resA], precedences=False, profp=-.00001, rows=[[A],[B],[C],[D],[E],[F]], offset=1)
    
    d = Schedule()
    
    A = Task(d,duration=3,release=0,duedate=8,demand=2,label='a')
    B = Task(d,duration=4,release=6,duedate=17,demand=1,label='b')
    C = Task(d,duration=4,release=0,duedate=14,demand=2,label='c')
    D = Task(d,duration=3,release=10,duedate=17,demand=3,label='d')
    E = Task(d,duration=4,release=4,duedate=15,demand=2,label='e')
    F = Task(d,duration=5,release=1,duedate=14,demand=1,label='f')
    
    x1 = Task(d,duration=2,release=3,duedate=5,demand=2,label='1')
    x2 = Task(d,duration=2,release=6,duedate=8,demand=1,label='2')
    x3 = Task(d,duration=1,release=8,duedate=9,demand=3,label='3')
    x4 = Task(d,duration=1,release=9,duedate=10,demand=2,label='4')
    x5 = Task(d,duration=1,release=10,duedate=11,demand=3,label='5')
    x6 = Task(d,duration=2,release=11,duedate=13,demand=1,label='6')
    x7 = Task(d,duration=1,release=13,duedate=14,demand=3,label='7')
    
    resA = Resource(d, 'A', [A,B,C,D,E,F,x1,x2,x3,x4,x5,x6,x7], capacity=4)
    
    outfile = open('tex/ex/ttef_decomposition.tex', 'w')
    
    d.latex(outfile, width=17, animated=True, precedences=False, rows=[[A],[B],[C],[D],[E],[F],[],[x1,x2,x3,x4,x5,x6,x7]],offset=2,stop='-')
    


