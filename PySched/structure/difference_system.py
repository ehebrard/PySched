
import sys

from sparse_set import *
import _testcapi as limit


DEBUG_DIFF  = False


def task_id(x):
    return x//2-1
    
def var(x):
    v = 'o'
    if x==1:
        v = 'm'
    elif x>1:
        if (x%2)==0:
            v = 's%i'%(task_id(x))
        else:
            v = 'e%i'%(task_id(x))
    return '%s'%v

def pretty(t):
    if t == limit.INT_MAX:
        return 'oo'
    elif t == -limit.INT_MAX:
        return '-oo'
    else:
        return str(t)

class Failure(Exception):
    def __init__(self, value=None):
        self.value = value
    def __str__(self):
        return repr(self.value)

# Reversible directed acyclic graph representing distances between points on a 2-d axis
class DifferenceSystem:
    def __init__(self): #x - y <= k
        self.distance = [] # dist[x][y] = (shortest path from x to y) || y is at least -k before x
        self.succ = [] # j \in succ[i] => xi - xj <= distance[i][j]
        self.pred = [] # i \in prec[j] <=> j \in succ[i]
        self.edges = [] # the list of succ in order of addition
        self.trail = [0] # the sizes of edges after each call to "save"
        self.stack = SparseSet()
        self.lb_reason = array.array('i',[])
        self.ub_reason = array.array('i',[])
        self.label = []
        
    # declare a new time point in the graph
    def newElement(self, label=None):
        n = len(self.succ)
        
        for i in range(n):
            self.distance[i].append(limit.INT_MAX)
            self.succ[i].newElement(n)
            self.pred[i].newElement(n)
        
        self.distance.append(array.array('i',[limit.INT_MAX]*(n+1)))        
        self.distance[n][n] = 0
        self.succ.append(SparseSet(capacity=n+1))
        self.pred.append(SparseSet(capacity=n+1))
        
        self.stack.newElement(n)
        self.lb_reason.append(0)
        self.ub_reason.append(1)
        if label is None:
            label = '$x_{%i}$'%(len(self.label))
        self.label.append(label)
                
    # save the current graph and bounds
    def save(self):
        # store the current size of the edge list (to remove subsequent edges)
        self.trail.append(len(self.edges))
    
    # restore to the last saved state
    def restore(self):        
        previous = self.trail.pop()
        while len(self.edges) > previous:
            self.removeEdge(*self.edges.pop())
            
    # add an edge WITHOUT updating the bounds
    def addNoUpdate(self, xi, xj, k): # add xi - xj <= k    
        if DEBUG_DIFF:
            print '\nadd %s - %s <= %i'%(var(xi),var(xj),k)
        # do something only if it is tighter than the exisiting edge
        if self.distance[xj][xi] > k: 
            self.succ[xi].add(xj)
            self.pred[xj].add(xi)
            self.edges.append((xi,xj,self.distance[xj][xi],k));
            self.distance[xj][xi] = k
            return True
        return False
         
    # add an edge (reversible) and put the event(s) on the stack
    def addEdge(self, xi, xj, k, update=True): # add xi - xj <= k
        if self.addNoUpdate(xi, xj, k):
            if update:
                self.update(xi, xj)
                return True
        return False
         
    # run Bellman-Ford from the potential changes
    def update(self, xi, xj):
        self.start = (xi, xj)
        self.stack.add(xj)
        self.stack.add(xi)
        self.propagateLowerBound()
        self.start = (xj, xi)
        self.stack.add(xi)
        self.stack.add(xj)
        self.propagateUpperBound()
        
    # remove an edge and replace it by the previous version
    def removeEdge(self, xi, xj, old=limit.INT_MAX, k=0):
        self.distance[xj][xi] = old
        if old == limit.INT_MAX:
            self.succ[xi].remove(xj)
            self.pred[xj].remove(xi)
        
    # the earliest time for point x
    def getLB(self,x):
        return -self.distance[x][0]
    
    # the latest time for point x
    def getUB(self,x):
        return self.distance[0][x]
        
    def getMinDistance(self,x,y):
        return -self.distance[x][y];
        
    def getMaxDistance(self,x,y):
        return self.distance[y][x];
        
    def printGraph(self):
        for xi, xj, k in self.edges:
            print xi, '-[%i]->'%self.distance[xj][xi], xj
            if xj not in self.succ[xi] or xi not in self.pred[xj]:
                print 'ERROR!'
                sys.exit(1)
        self.printBounds()
            
    def printBounds(self):
        for x in range(len(self.succ)):
            print '%i: [%i..%i]'%(x, self.getLB(x), self.getUB(x))
                
    def updateLB(self,x,d):
        self.addEdge(0,x,d)
        
    def updateUB(self,x,d):
        self.addEdge(x,0,d)
            
    # The following are what we need to run Bellman-Ford (forward or backward)
    def getForwardDistance(self,x,y):
        return self.distance[y][x]

    def getBackwardDistance(self,x,y):
        return self.distance[x][y]

    def setLB(self,x,d):
        self.addNoUpdate(0,x,d)
        # self.schedule.notify(x,LB_CHANGE)

    def setUB(self,x,d):
        self.addNoUpdate(x,0,d)
        # self.schedule.notify(x,UB_CHANGE)

    def cycle(self,x,y):
        if DEBUG_DIFF:
            print 'check cycle w.r.t. %s: %i + %i'%(var(y), self.distance[0][y], self.distance[y][0])
        
        if self.distance[0][y] + self.distance[y][0] < 0:
            return True
        elif (x,y) == self.start:
            self.cycle_count += 1
            return self.cycle_count == 2
        return False

    def BellmanFord(self, stack, neighborhood, getDistance, setDistance, prev):
        self.cycle_count = 0
        while len(stack) > 0:
            u = stack.pop()
            
            if DEBUG_DIFF:
                print 'pop', var(u)
            for v in neighborhood[u]:
                alt = getDistance(0,u) 
                if alt != limit.INT_MAX :
                    alt += getDistance(u,v)
                if alt < getDistance(0,v):
                    
                    if DEBUG_DIFF:
                        print ' ->', var(v)
                    prev[v] = u
                    setDistance(v,alt)
                    if self.cycle(u,v):
                        stack.clear()
                        
                        # print 'Failure in Bellman-Ford on %3s because of %3s'%(var(v), var(u))
                        
                        raise Failure((u,v,alt))
                    stack.add(v)
                    
    def FloydWarshall(self):
        V = range(len(self.succ))
        dist = self.distance
        change = False
        for k in V:
            for i in V:
                for j in V:
                    change |= self.addNoUpdate(j,i,dist[i][k] + dist[k][j])
        return change
    
    def propagateLowerBound(self):
        return self.BellmanFord(self.stack, self.succ, self.getForwardDistance, self.setLB, self.lb_reason)
    
    def propagateUpperBound(self):
        return self.BellmanFord(self.stack, self.pred, self.getBackwardDistance, self.setUB, self.ub_reason)
        
    def printMatrix(self):
        for i in range(len(self.succ)):
            for j in range(len(self.succ)):
                print '%3s'%(pretty(self.distance[i][j])),
            print
    
    def latex(self, outfile=sys.stdout, schedule=True, width=10, sep=2, lb=[], ub=[], resources={}):
        n = len(self.succ)
        
        highlight = set([])
        # print lb
        for v in lb:
            u = v
            while self.lb_reason[u] != 0:
                highlight.add((self.lb_reason[u],u))
                # print u, '<-',
                u = self.lb_reason[u]
            highlight.add((0,u))
            # print 0
            
        for v in ub:
            u = v
            while self.ub_reason[u] != 1:
                highlight.add((u,self.ub_reason[u]))
                u = self.ub_reason[u]
            highlight.add((u,1))
                
        # sys.exit(1)
        
        
        layers = [[0]]
        layer_of = [0]*len(self.succ)
        visited = set([0,1])
        sources = [v for v in range(2,n) if len([u for u in self.pred[v] if u>1 and u!=(v+1)])==0]
        sinks = [v for v in range(2,n) if len([u for u in self.succ[v] if u>1 and u!=(v-1)])==0]
        
        layers.append(sources)   
        
        while True:
            newlayer = set([])
            for u in layers[-1]:
                layer_of[u] = len(layers)
                visited.add(u)
                for v in self.succ[u]:
                    if v not in visited:
                        newlayer.add(v)
            if len(newlayer):
                layers.append([v for v in newlayer])
            else:
                layers.append([1])
                break
                
        X = 0
        layer_sep = sep
        # width = 8.0
        for layer in layers:
            for i,v in enumerate(layer):
                Y = width*(i+1)/(len(layer)+1)
                if resources.has_key(v):
                    outfile.write('\\node[vertexstyle,extst%s] (x%i) at (%f,%f) {%s};\n'%(resources[v],v,X,Y,self.label[v]))
                else:
                    outfile.write('\\node[vertexstyle] (x%i) at (%f,%f) {%s};\n'%(v,X,Y,self.label[v]))
            X += layer_sep

        for u in range(2,n):
            for v in range(2,n):
                if u != v:
                    if self.distance[v][u] != limit.INT_MAX:
                        if u//2 == v//2 or layer_of[v] > layer_of[u]+1:
                            if (u,v) in highlight:
                                outfile.write('\\path (x%i) edge[edgestyle, bend left, color=bostonuniversityred] node[labelstyle] {\\textcolor{bostonuniversityred}{$%i$}} (x%i);\n'%(u,self.distance[v][u],v))
                            else:
                                outfile.write('\\path (x%i) edge[edgestyle, bend left] node[labelstyle] {\\textcolor{DarkBlue}{$%i$}} (x%i);\n'%(u,self.distance[v][u],v))
                        else:
                            if (u,v) in highlight:
                                outfile.write('\\path (x%i) edge[edgestyle, color=bostonuniversityred] node[labelstyle] {\\textcolor{bostonuniversityred}{$%i$}} (x%i);\n'%(u,self.distance[v][u],v))
                            else:
                                outfile.write('\\path (x%i) edge[edgestyle] node[labelstyle] {\\textcolor{DarkBlue}{$%i$}} (x%i);\n'%(u,self.distance[v][u],v))

        for v in sources:
            if (0,v) in highlight:
                outfile.write('\\path (x0) edge[edgestyle, color=bostonuniversityred] node[labelstyle] {\\textcolor{bostonuniversityred}{$%i$}} (x%i);\n'%(self.distance[v][0],v))
            else:
                outfile.write('\\path (x0) edge[edgestyle] node[labelstyle] {\\textcolor{DarkBlue}{$%i$}} (x%i);\n'%(self.distance[v][0],v))
            
        for v in sinks:
            if layer_of[v] < len(layers)-1:
                if (v,1) in highlight:
                    outfile.write('\\path (x%i) edge[edgestyle, bend right=45, color=bostonuniversityred] node[labelstyle] {\\textcolor{bostonuniversityred}{$%i$}} (x1);\n'%(v,self.distance[1][v]))
                else:
                    outfile.write('\\path (x%i) edge[edgestyle, bend right=45] node[labelstyle] {\\textcolor{DarkBlue}{$%i$}} (x1);\n'%(v,self.distance[1][v]))
            else:
                if (v,1) in highlight:
                    outfile.write('\\path (x%i) edge[edgestyle, color=bostonuniversityred] node[labelstyle] {\\textcolor{bostonuniversityred}{$%i$}} (x1);\n'%(v,self.distance[1][v]))
                else:
                    outfile.write('\\path (x%i) edge[edgestyle] node[labelstyle] {\\textcolor{DarkBlue}{$%i$}} (x1);\n'%(v,self.distance[1][v]))

        if self.distance[0][1] != limit.INT_MAX:
            if len(ub)>0:
                outfile.write('\\path (x1) edge[edgestyle, bend right=60, color=bostonuniversityred] node[labelstyle] {\\textcolor{bostonuniversityred}{$%i$}} (x0);\n'%(self.distance[0][1]))
            else:
                outfile.write('\\path (x1) edge[edgestyle, bend right=60] node[labelstyle] {\\textcolor{DarkBlue}{$%i$}} (x0);\n'%(self.distance[0][1]))

        
        # layers = []
        # prec_edges = []
        # dur_edges = []
        # bent_edges = []
        # visited = set([1])
        #
        #
        # if schedule:
        #     layers.append([0])
        #     layers.append([v for v in self.succ[0] if v%2 == 0 and len(self.pred[v])==2])
        #     for v in layers[-1]:
        #         prec_edges.append((0,v))
        # else:
        #     layers.append([i for i in range(len(self.pred)) if len(self.pred[i])==0])
        #
        # layer_of = [0]*len(self.succ)
        #
        # while True:
        #     print '\n'.join([str(l) for l in layers])
        #     print
        #     newlayer = set([])
        #     for u in layers[-1]:
        #         visited.add(u)
        #         for v in self.succ[u]:
        #             if self.distance[0][u] <= self.distance[0][v] and v not in visited:
        #                 newlayer.add(v)
        #     if len(newlayer):
        #         layers.append([v for v in newlayer])
        #     else:
        #         layers.append([1])
        #         break
        #
        # for i,layer in enumerate(layers):
        #     for v in layer:
        #         layer_of[v] = i
        #
        # print '\n'.join([str(l) for l in layers])
        # print
        #
        # print self.succ[7]
        #
        # layer_sep = 2.0
        # width = 10.0
        #
        # # outfile.write('\\begin{tikzpicture}\n')
        #
        # X = 0
        # # for layer in layers:
        # #     for i,v in enumerate(layer):
        # #         Y = width*(i+1)/(len(layer)+1)
        # #         outfile.write('\\Vertex[x=%f ,y=%f]{%s}\n'%(X,Y,self.label[v]))
        # #     X += layer_sep
        # #
        # # for u,v in prec_edges:
        # #     outfile.write('\\Edge[label=$%i$](%s)(%s)\n'%(self.distance[v][u], self.label[u], self.label[v]))
        #
        # for layer in layers:
        #     for i,v in enumerate(layer):
        #         Y = width*(i+1)/(len(layer)+1)
        #         outfile.write('\\node[vertexstyle] (x%i) at (%f,%f) {%s};\n'%(v,X,Y,self.label[v]))
        #     X += layer_sep
        #
        #
        # for u in range(len(self.succ)):
        #     for v in range(u+1, len(self.succ)):
        #         if self.distance[v][u] != limit.INT_MAX:
        #             if u == 0:
        #                 if v > 1 and self.distance[v][u] == 0:
        #                     outfile.write('\\path (x%i) edge[edgestyle] node[labelstyle] {\\textcolor{DarkBlue}{$%i$}} (x%i);\n'%(u,self.distance[v][u],v))
        #             elif u//2 == v//2:
        #                 outfile.write('\\path (x%i) edge[edgestyle, bend left] node[labelstyle] {\\textcolor{DarkBlue}{$%i$}} (x%i);\n'%(u,self.distance[v][u],v))
        #                 outfile.write('\\path (x%i) edge[edgestyle, bend left] node[labelstyle] {\\textcolor{DarkBlue}{$%i$}} (x%i);\n'%(v,self.distance[u][v],u))
        #             elif layer_of[v] > layer_of[u]+1:
        #                 outfile.write('\\path (x%i) edge[edgestyle, bend right=60] node[labelstyle] {\\textcolor{DarkBlue}{$%i$}} (x%i);\n'%(u,self.distance[v][u],v))
        #             else:
        #                 outfile.write('\\path (x%i) edge[edgestyle] node[labelstyle] {\\textcolor{DarkBlue}{$%i$}} (x%i);\n'%(u,self.distance[v][u],v))
        #
        # if self.distance[0][1] != limit.INT_MAX:
        #     outfile.write('\\path (x1) edge[edgestyle, bend right=60] node[labelstyle] {\\textcolor{DarkBlue}{$%i$}} (x0);\n'%(self.distance[0][1]))
        #
        # for u in range(2, len(self.succ)):
        #     pass
                       
                            
        # for u,v in prec_edges:
        #     outfile.write('\\path (x%i) edge[edgestyle] node[labelstyle] {\\textcolor{DarkBlue}{$%i$}} (x%i);\n'%(u,self.distance[v][u],v))
        #
        # for u,v in bent_edges:
        #     outfile.write('\\path (x%i) edge[edgestyle, bend right=60] node[labelstyle] {\\textcolor{DarkBlue}{$%i$}} (x%i);\n'%(u,self.distance[v][u],v))
        #
        # for u,v in dur_edges:
        #     outfile.write('\\path (x%i) edge[edgestyle, bend left] node[labelstyle] {\\textcolor{DarkBlue}{$%i$}} (x%i);\n'%(u,self.distance[v][u],v))
        #     outfile.write('\\path (x%i) edge[edgestyle, bend left] node[labelstyle] {\\textcolor{DarkBlue}{$%i$}} (x%i);\n'%(v,self.distance[u][v],u))
        #
        # # outfile.write('\\end{tikzpicture}\n')
        
      
        
        
if __name__ == '__main__':
    
    graph = DifferenceSystem()
    
    
    graph.newElement(label='$o$')
    
    for i in range(7):
        graph.newElement()
        
    graph.newElement(label='$m$')
        
    graph.addEdge(0,1,-3,update=False)
    graph.addEdge(0,2,-2,update=False)
    
    graph.addEdge(1,3,-4,update=False)
    graph.addEdge(2,4,-1,update=False)
    
    graph.addEdge(3,5,-1,update=False)
    graph.addEdge(3,6,-3,update=False)
    graph.addEdge(4,7,-7,update=False)
    
    graph.addEdge(5,8,-7,update=False)
    graph.addEdge(6,8,1,update=False)
    graph.addEdge(7,8,-6,update=False)
    
    graph.printMatrix()
    
    
    graph.latex(outfile=open('tex/ex/g.tex', 'w'))
    
    
    
        
