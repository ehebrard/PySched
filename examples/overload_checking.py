
from PySched import *

                
if __name__ == '__main__':
    
    s = Schedule()
    
    A = Task(s,duration=5,release=0,duedate=14,label='A')
    B = Task(s,duration=5,release=1,duedate=15,label='B')
    C = Task(s,duration=6,release=2,duedate=13,label='C')
            
    res = Resource(s, 'A', [A,B,C])
    
    outfile = open('tex/ex/overload_checking.tex', 'w')
    
    s.latex(outfile, animated=True, width=15, horizon=15)
    s.latex(outfile, animated=True, mandatory=True, precedences=False, windows=False, width=15, horizon=15)
    
    tt = Timetabling(res)
    
    while True:
        if not tt.propagate():
            break

        s.latex(outfile, animated=True, precedences=False, width=15, horizon=15)
        s.latex(outfile, animated=True, mandatory=True, precedences=False, width=15, horizon=15)
        
        
    disjuncts = res.getDisjuncts()
    
    change = True
    while change:
        change = False
        for d in disjuncts:
            if d.propagate():
                change = True

    s.save()
    s.addPrecedence(B,C)
    s.latex(outfile, animated=True, precedences=True, width=15, horizon=15, rows=[[A],[B],[C]])
    s.restore()
    s.save()
    s.addPrecedence(C,B)
    s.latex(outfile, animated=True, precedences=True, width=15, horizon=15, rows=[[A],[B],[C]])

    outfile.write('\\uncover<5->{')
    outfile.write('\\PrintTics{0,3,...,16}{1.000000}')
    outfile.write('\\PrintGrid{15}{3}')
    outfile.write('\\LeftPhantomExtensibleVariableTask{0.000000}{14.000000}{5.000000}{5.000000}{1}{0}{A}{A}')
    outfile.write('\\LeftPhantomExtensibleVariableTask{1.000000}{15.000000}{5.000000}{5.000000}{1}{1}{A}{B}')
    outfile.write('\\LeftPhantomExtensibleVariableTask{2.000000}{13.000000}{6.000000}{6.000000}{1}{2}{A}{C}')
    outfile.write('\\ExtensibleTask{0}{5.000000}{5.000000}{1}{3.5}{A}{A}')
    outfile.write('\\ExtensibleTask{5}{5.000000}{5.000000}{1}{3.5}{A}{B}')
    outfile.write('\\ExtensibleTask{10}{6.000000}{6.000000}{1}{3.5}{A}{C}')
    outfile.write('}')
    
    
    
    

    
    


    
        
#
# 45/93                                            xxx
# 21/45                   xxx                                             xxx
# 9/21         xxx                     xxx                     xxx
# 3/9    xxx         xxx         xxx         xxx         xxx         xxx
# 0/3 xxx   xxx   xxx   xxx   xxx   xxx   xxx   xxx   xxx   xxx   xxx   xxx
#




