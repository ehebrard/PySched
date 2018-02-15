
from PySched import *
import PySched.structure.theta_tree as theta
import PySched.examples as ex
                
if __name__ == '__main__':
    
    s = Schedule()
    
    example = []
    example.append(Task(s, duration=5, release=0, duedate=15, label='A'))
    example.append(Task(s, duration=4, release=4, duedate=10, label='B'))
    example.append(Task(s, duration=3, release=2, duedate=12, label='C'))
    example.append(Task(s, duration=4, release=7, duedate=14, label='D')) 
    example.append(Task(s, duration=6, release=3, duedate=20, label='E'))
    example.append(Task(s, duration=3, release=10, duedate=16, label='F'))
    
    sorted_example = sorted(example, key=s.getLatestCompletion)
    
    Resource(s, name='C', init=example)
    
    treefile = open('tex/ex/d_theta_tree.tex', 'w')
    schedfile = open('tex/ex/d_theta.tex', 'w')
    

    s.latex(schedfile,animated=True,rows=[[t] for t in example])
    
    tt = theta.ThetaTree(example) 
        
    s.latex(schedfile,animated=True,rows=[[t] for t in tt.tasks])
    s.latex(schedfile,animated=True,rows=[[t] for t in tt.tasks],tasks=[],stop='-')
    
    for i, t in enumerate(sorted_example):
        t.rank = i
    
    for i,t in enumerate(sorted_example):
        schedfile.write('\\uncover<%i->{'%(i+3))
        schedfile.write(t.strInterval(row=t.rank, mode='Left'))
        schedfile.write('}\n')

    tt.exec2latex(example,output=treefile)

    ex.writeFile('thetatree_demo', ['d_theta', 'd_theta_tree'], headers=['colorschedfigure', 'downthreelvltree'], scale=.45)




