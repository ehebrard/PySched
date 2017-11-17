
from PySched import *

                
if __name__ == '__main__':
    
    s = Schedule()
    
    A = Task(s,duration=(5,7),label='Mupus')
    B = Task(s,duration=4,label='Ptolemy')
    C = Task(s,duration=2,label='APXS')
    D = Task(s,duration=6,label='Civa')
    E = Task(s,duration=4,label='Rolis')
    F = Task(s,duration=2,label='Sesame')
            
    H = 18
    
    s.setMakespanUB(H)
    
    outfile = open('tex/ex/notation.tex', 'w')
    
    s.latex(outfile, animated=True, horizon=H, width=H, windows=False)
    
    A << B
    E << C
    E << A
    C << F
    
    s.latex(outfile, animated=True, horizon=H, width=H, windows=False)
    
    s.latex(outfile, animated=True, horizon=H, width=H, windows=True, precedences=False, rows=[[E],[A],[C],[B],[F],[D]], shifts=(['']*5 + ['Left']))
    #rows=[[t] for t in [E,A,C,B,F,D]])
    
    b1 = Resource(s, 'A', capacity=6, init=[B,F])
    b2 = Resource(s, 'B', capacity=6, init=[A,C,D,E])
    
    
    s.latex(outfile, animated=True, horizon=H, width=H, windows=False)
    
    s.setDemand(A, 3)
    s.setDemand(B, 4)
    s.setDemand(C, 3)
    s.setDemand(D, 2)
    s.setDemand(E, 1)
    s.setDemand(F, 1)
    
    s.latex(outfile, animated=True, horizon=H, width=H, windows=False)
    
    import examples
    examples.writeFile('example', 'notation', scale=.5)



