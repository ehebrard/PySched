
from PySched import *
                
if __name__ == '__main__':
    
    s = Schedule()
    
    A = Task(s,duration=12,release=0,duedate=25,label='A')
    ALS = Task(s,duration=12,release=0,duedate=12,label='A')
    ARS = Task(s,duration=12,release=13,duedate=25,label='A')
    
    
    outfile = open('tex/ex/shifts.tex', 'w')
    
    s.latex(outfile, animated=True, width=25, horizon=25, tasks=[A], rows=[[A],[ALS],[ARS]], windows=True)
    outfile.write('\\draw[ub] (3,.5) -- (3,-3.5);\n')
    outfile.write('\\node[] (a) at (3,-3.8) {a=3};\n')
    outfile.write('\\draw[ub] (19,.5) -- (19,-3.5);\n')
    outfile.write('\\node[] (b) at (19,-3.8) {b=19};\n')
    
    
    
    s.latex(outfile, animated=True, width=25, horizon=25, tasks=[A,ALS], rows=[[A],[ALS],[ARS]], windows=True)
    outfile.write('\\uncover<2>{\n')
    outfile.write('\\draw[lb] (0,.5) -- (0,-3.5);\n')
    outfile.write('\\node[] at (0,-3.8) {$\est{A}=0$};\n')
    outfile.write('\\draw[<->] (0,-4.5) -- node[inner sep=1pt, fill=white] {3} (3,-4.5);\n')
    outfile.write('\\node[below right of=a] {\memph{$\\leftshift{A}{3}=12-3=9$}};\n')
    outfile.write('}\n')
    
    
    s.latex(outfile, animated=True, width=25, horizon=25, tasks=[A,ARS], rows=[[A],[ARS],[ALS]], windows=True)
    outfile.write('\\uncover<3>{\n')
    outfile.write('\\draw[lb] (25,.5) -- (25,-3.5);\n')
    outfile.write('\\node[] at (25,-3.8) {$\lct{A}=25$};\n')
    outfile.write('\\draw[<->] (25,-4.5) -- node[inner sep=1pt, fill=white] {6} (19,-4.5);\n')
    outfile.write('\\node[below left of=b] {\memph{$\\rightshift{A}{19}=12-6=6$}};\n')
    outfile.write('}\n')
    
    s.latex(outfile, animated=True, width=25, horizon=25, tasks=[A], rows=[[A],[ALS],[ARS]], windows=True, stop='-')


    import examples
    examples.writeFile('energetic_example', 'shifts')

