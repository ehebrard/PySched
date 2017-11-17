
from PySched import *

                
if __name__ == '__main__':
    
    # s = Schedule()
    #
    # A = Task(s,duration=1,release=0,duedate=1,demand=3,label='a')
    # B = Task(s,duration=1,release=1,duedate=2,demand=2,label='b')
    # C = Task(s,duration=1,release=2,duedate=3,demand=4,label='c')
    # D = Task(s,duration=1,release=2,duedate=3,demand=2,label='d')
    # E = Task(s,duration=1,release=0,duedate=1,demand=1,label='e')
    #
    # res = Resource(s, 'A', [A])
    # res = Resource(s, 'B', [B])
    # res = Resource(s, 'C', [C])
    # res = Resource(s, 'D', [D])
    # res = Resource(s, 'E', [E])
    #
    outfile = open('tex/ex/binpacking_example.tex', 'w')
    #
    # s.latex(outfile, animated=True, rows=[[D,E],[A,B,C]], horizon=3, width=3)
    
    
    outfile.write('\\uncover<1>{\n\\PrintTics{0,1,...,3}{1.000000}\n\\PrintGrid{3}{6}\n\\GroundTask{2.000000}{1.000000}{2}{0}{D}{d}\n\\GroundTask{0.000000}{1.000000}{1}{2}{E}{e}\n\\GroundTask{0.000000}{1.000000}{3}{3}{A}{a}\n\\GroundTask{1.000000}{1.000000}{2}{4}{B}{b}\n\\GroundTask{2.000000}{1.000000}{4}{2}{C}{c}\n\\GroundTask{1.000000}{1.000000}{2}{2}{F}{f}\n}\n')

    outfile.close()


