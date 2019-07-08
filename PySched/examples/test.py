
from PySched import *
import PySched.examples as ex
                
if __name__ == '__main__':
    
    s = Schedule()
    
    s.closure = s.graph.FloydWarshall
    
    A = Task(s,duration=10,demand=1,label='A')
    B = Task(s,duration=10,demand=1,label='B')
    C = Task(s,duration=10,demand=1,label='C')
    D = Task(s,duration=10,demand=1,label='D')
            
    s.startBeforeEnd(A,B)
    s.startBeforeEnd(B,A)
    s.startBeforeEnd(C,B)
    s.startBeforeEnd(B,C)
    s.startBeforeEnd(C,D)
    s.startBeforeEnd(D,C)
    s.startBeforeEnd(A,D)
    s.startBeforeEnd(D,A)
    
    A << C

    span = 100
    s.setMakespanUB(span)
    
    
    s.graph.printMatrix()
    
    s.propagate(trace=True)
    
    s.graph.printMatrix()
    

    
    print 'min distance from start(B) to end(D):', -s.graph.distance[end(D)][start(B)];
    print 'min distance from start(D) to end(B):', -s.graph.distance[end(B)][start(D)];
    
    print 'max distance from start(B) to end(D):', s.graph.distance[start(B)][end(D)];
    print 'max distance from start(D) to end(B):', s.graph.distance[start(D)][end(B)];


