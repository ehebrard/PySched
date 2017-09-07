
"""
Theta tree.

File: theta_tree.py
Author: Emmanuel Hebrard (hebrard@laas.fr)

An implementation of Theta trees from Petr Vilim's thesis, 
or rather of my understanding of it.

This is a static search tree. 
Leaves are preordered (construction is nlog(n)),
but given a null value. 

Insertion of a task: set the corresponding leaf to the task,
then go up to the root while updating the branch

Deletion of a task: set the corresponding leaf to nill,
then go up to the root while updating the branch
"""

import operator
import sys
import _testcapi as limit


DEBUG_THETA = False

    
def parent(x):
    return x//2
    
    
class ThetaNode:
    
    def __init__(self, inf):
        self.clear(inf)
    
    def clear(self, inf):
        self.duration = 0
        self.bound = inf
        self.gray_duration = 0
        self.gray_bound = inf
        self.responsibleDuration = None
        self.responsibleBound = None
        
    def gray(self):
        return self.responsibleDuration is None and self.responsibleBound is None
        
        
class ThetaTree:
    
    def __init__(self, tasks, backward=False):
        if len(tasks)>0:
            
            self.tasks = [t for t in tasks]
            self.schedule = tasks[0].schedule
            self.backward = backward
            
            if backward:
                self.inf = limit.INT_MAX
                self.sorting = self.schedule.getLatestCompletion
                self.t_bound = self.schedule.getLatestStart
                self.best = min
                self.combine = operator.__sub__
            else:
                self.inf = -limit.INT_MAX
                self.sorting = self.schedule.getEarliestStart
                self.t_bound = self.schedule.getEarliestCompletion
                self.best = max
                self.combine = operator.__add__
            
            self.N = 1
            while self.N<len(tasks):
                self.N*=2
            self.node = [ThetaNode(self.inf) for i in range(self.N+len(tasks))]
            # self.index = [0]*len(self.tasks)
            self.index = {}
              
            # sortedTasks = sorted(tasks, key=schedule.getEarliestStart)
            self.reinit()
            
    def reinit(self):
        
            self.tasks.sort(key=self.sorting, reverse=self.backward)
            
            if DEBUG_THETA:
                for t in self.tasks:
                    print ' *',str(t) 
                    
            n = self.N
            for t in self.tasks:
                self.index[t] = n
                n += 1
            
    def leftChild(self,x):
        return 2*x

    def rightChild(self,x):
        r = 2*x+1
        if r>=len(self.node):
            return 0
        return r
        
    def insert(self, task, gray=False):
        if DEBUG_THETA:
            print 'insert %s (%i)'%(task, self.index[task]-self.N)
        i = self.index[task]

        if gray:
            self.node[i].duration = 0
            self.node[i].bound = self.inf
            self.node[i].gray_duration = task.minDuration()
            self.node[i].gray_bound = self.t_bound(task)
            self.node[i].responsibleDuration = task
            self.node[i].responsibleBound = task
        else:
            self.node[i].duration = task.minDuration()
            self.node[i].bound = self.t_bound(task)        
            self.node[i].gray_duration = 0
            self.node[i].gray_bound = self.inf
            self.node[i].responsibleDuration = None
            self.node[i].responsibleBound = None
 
        self.update(i, gray)
           
        if DEBUG_THETA:
            print '-> %i <? %i'%(self.node[1].bound, target(task))
            print self
           
    def delete(self, task, gray=False):
        if DEBUG_THETA:
            print 'delete %s (%i)'%(task, self.index[task]-self.N)
        i = self.index[task]
        self.node[i].clear(self.inf)
        self.update(i, gray)
           
        if DEBUG_THETA:
            print '-> %i <? %i'%(self.node[1].bound, target(task))
            print self
           
    def update(self, i, gray):
        while i>1:
           i = parent(i)
           l = self.leftChild(i)
           r = self.rightChild(i)
           self.node[i].duration = self.node[l].duration + self.node[r].duration
           self.node[i].bound = self.best((self.node[r].bound, self.combine(self.node[l].bound, self.node[r].duration)))
           if gray:
                if not self.node[r].gray() and not self.node[l].gray():
                    self.node[i].gray_duration = 0
                    self.node[i].gray_bound = self.inf
                    self.node[i].responsibleDuration = None
                    self.node[i].responsibleBound = None
                else:
               
                    ldrgd = self.node[l].duration + self.node[r].gray_duration
                    rdlgd = self.node[l].gray_duration + self.node[r].duration

                    rge = self.node[r].gray_bound
                    lergd = self.combine(self.node[l].bound, self.node[r].gray_duration)
                    lgerd = self.combine(self.node[l].gray_bound, self.node[r].duration)
                
                    self.node[i].gray_duration = max(ldrgd, rdlgd)
                    self.node[i].gray_bound = self.best(rge, lgerd, lergd)
                
                    if self.node[i].gray_duration == ldrgd:
                        self.node[i].responsibleDuration = self.node[r].responsibleDuration    
                    elif self.node[i].gray_duration == rdlgd:
                        self.node[i].responsibleDuration = self.node[l].responsibleDuration
                    
                    if self.node[r].responsibleBound is not None and self.node[i].gray_bound == rge:
                        self.node[i].responsibleBound = self.node[r].responsibleBound
                    elif self.node[l].responsibleBound is not None and self.node[i].gray_bound == lgerd:
                        self.node[i].responsibleBound = self.node[l].responsibleBound  
                    elif self.node[r].responsibleDuration is not None and self.node[i].gray_bound == lergd:
                        self.node[i].responsibleBound = self.node[r].responsibleDuration
                        
    def clear(self):
        for n in self.node:
            n.clear(self.inf)    
        
    def getBound(self):
        return self.node[1].bound
        
    def getBoundMinus(self, task):
        
        # print self
        
        i = self.index[task]
        duration = 0
        bound = self.inf
        
        # print i, (i%2 == 0), duration, bound
        
        while i>1:
            left = (i%2 == 0)
            i = parent(i)
            if left:
                r = self.rightChild(i)           
                bound = self.best((self.node[r].bound, self.combine(bound, self.node[r].duration)))
                duration = duration + self.node[r].duration
                
            else:
                l = self.leftChild(i)
                bound = self.best((bound, self.combine(self.node[l].bound, duration)))
                duration = self.node[l].duration + duration

            # left = (i%2 == 0)
            
            # print i, left, duration, bound
            
        return bound
        
    def getGrayBound(self):
        return self.node[1].gray_bound
        
    def getCulprit(self):
        return self.node[1].responsibleBound
    
    def __str__(self):
        rstr = ''
        
        ew = 4
        width = 0
        n = 1
        while n<len(self.node):
            n *= 2
            width *= 2
            width += ew
        
        i = 1
        layer = 1
        while i<len(self.node):
            rstr += ' '*((width-ew)/2)
            
            j = 0
            while i+j<len(self.node):
                if self.node[i+j].duration == 0:
                    rstr += '#'*ew
                else:
                    rstr += '%4i'%(self.node[i+j].duration)
                j += 1
                if j>= layer:
                    break
                rstr += ' '*width
            
            j = 0
            rstr += '\n'
            rstr += ' '*((width-ew)/2)
            while i+j<len(self.node):
                if self.node[i+j].bound == self.inf:
                    rstr += '+'*ew
                else:
                    rstr += '%4i'%(self.node[i+j].bound)
                j += 1
                if j>= layer:
                    break
                rstr += ' '*width
            rstr += '\n'
            j = 0
            rstr += ' '*((width-ew)/2)
            while i+j<len(self.node):
                if self.node[i+j].gray_duration == 0:
                    rstr += '#'*ew
                else:
                    rstr += '%4i'%(self.node[i+j].gray_duration)
                j += 1
                if j>= layer:
                    break
                rstr += ' '*width
            j = 0
            rstr += '\n'
            rstr += ' '*((width-ew)/2)
            while i+j<len(self.node):
                if self.node[i+j].gray_bound == self.inf:
                    rstr += '+'*ew
                elif self.node[i+j].gray_bound == self.inf:
                    rstr += '~%3i'%(limit.INT_MAX+self.node[i+j].gray_bound)
                else:
                    rstr += '%4i'%(self.node[i+j].gray_bound)
                j += 1
                if j>= layer:
                    break
                rstr += ' '*width
            j = 0
            rstr += '\n'
            rstr += ' '*((width-ew)/2)
            while i+j<len(self.node):
                if self.node[i+j].responsibleBound == None:
                    rstr += 'none'
                else:
                    rstr += '%4i'%(self.node[i+j].responsibleBound)
                j += 1
                if j>= layer:
                    break
                rstr += ' '*width
            j = 0
            rstr += '\n'
            rstr += ' '*((width-ew)/2)
            while i+j<len(self.node):
                if self.node[i+j].responsibleDuration == None:
                    rstr += 'none'
                else:
                    rstr += '%4i'%(self.node[i+j].responsibleDuration)
                j += 1
                if j>= layer:
                    break
                rstr += ' '*width
            rstr += '\n'
            i += j
            layer *= 2
            width -= ew
            width /= 2
            
        return rstr
        
    def node2latex(self, ni, dur, ect, first, outfile=sys.stdout):
        
        # print outfile
        # sys.exit(1)
        
        if ni==1 :
            outfile.write('\\thetaroot')
        else:
            outfile.write('\\ttnode')
        
        outfile.write('{')
        
        last = 0
        for i,d in enumerate(dur[ni]):
            if d>0:
                if d == last:
                    outfile.write('\\only<%i>{%i}'%(i+2,d))
                else:
                    outfile.write('\\only<%i>{\\textcolor{red!85!black}{%i}}'%(i+2,d))
                    last = d
                    
        outfile.write('}{')
        
        last = -1
        for i,e in enumerate(ect[ni]):
            if e>0:
                if e == last:
                    outfile.write('\\only<%i>{%i}'%(i+2,e))
                else:
                    outfile.write('\\only<%i>{\\textcolor{red!85!black}{%i}}'%(i+2,e))
                    last = e
                    
        outfile.write('}{%i}\n'%(first[ni]+1))
        
        # outfile.write('{%s}{%s}'%(''.join(['\\only<%i>{%i}'%(i+1,d) for i,d in enumerate(dur[ni]) if d>0]), ''.join(['\\only<%i>{%i}'%(i+1,e) for i,e in enumerate(ect[ni]) if e>=0]))
           
        li = self.leftChild(ni)
        if li < len(self.node) and first[li] is not None:
            outfile.write('child{\n')
        
            self.node2latex(li, dur, ect, first, outfile)
            if li >= self.N:
                outfile.write('edge from parent\n')
                outfile.write('       node[kant, left, yshift=1mm, pos=.6] {%s}\n'%(self.tasks[li-self.N].label))
            
            outfile.write('}\n')
            
            ri = self.rightChild(ni)
            if ri > 0 and first[ri] is not None:
                outfile.write('child{\n')
                self.node2latex(ri, dur, ect, first, outfile)
                if ri >= self.N:
                    outfile.write('edge from parent\n')
                    outfile.write('       node[kant, right, yshift=1mm, pos=.6] {%s}\n'%(self.tasks[ri-self.N].label))
        
                if ni==1 :
                    outfile.write('};\n')
                else:
                    outfile.write('}\n')
        
        
    def exec2latex(self, example, output=sys.stdout):
        s = example[0].schedule
        
        dur = [[n.duration] for n in self.node]
        ect = [[n.bound] for n in self.node]
        first = [None for n in self.node]
        
        sortedTasks = sorted(example, key=s.getLatestCompletion)
        
        for t in sortedTasks:
            self.insert(t)
            for i in range(len(self.node)):
                if self.node[i].duration != 0 and first[i] is None:
                    first[i] = len(dur[i])+1
                
                dur[i].append(self.node[i].duration)
                ect[i].append(self.node[i].bound)
            
        self.node2latex(1, dur, ect, first, outfile=output)

            
                      
if __name__ == '__main__':
    import PySched as sched
    
    s = sched.Schedule()
    
    example = []
    example.append(sched.Task(s, duration=5, release=0, duedate=15, label='A'))
    example.append(sched.Task(s, duration=4, release=4, duedate=10, label='B'))
    example.append(sched.Task(s, duration=3, release=2, duedate=12, label='C'))
    example.append(sched.Task(s, duration=4, release=7, duedate=14, label='D')) 
    example.append(sched.Task(s, duration=6, release=3, duedate=20, label='E'))
    example.append(sched.Task(s, duration=3, release=10, duedate=16, label='F'))
    
    sorted_example = sorted(example, key=s.getLatestCompletion)
    
    sched.Resource(s, name='C', init=example)
    
    treefile = open('tex/ex/theta_tree.tex', 'w')
    schedfile = open('tex/ex/theta_sched.tex', 'w')
    
    # outfile.write('\\begin{colorschedfigure}{.45}\n')

    s.latex(schedfile,animated=True,rows=[[t] for t in example])
    
    tt = ThetaTree(example) 
        
    s.latex(schedfile,animated=True,rows=[[t] for t in tt.tasks])
    s.latex(schedfile,animated=True,rows=[[t] for t in tt.tasks],tasks=[],stop='-')
    
    for i, t in enumerate(sorted_example):
        t.rank = i
    
    for i,t in enumerate(sorted_example):
        schedfile.write('\\uncover<%i->{'%(i+3))
        schedfile.write(t.strInterval(row=t.rank, mode='Left'))
        schedfile.write('}\n')
    # treefile.write('\\end{colorschedfigure}\n')
    # treefile.write('\\medskip\n')
    # treefile.write('\\uncover<2->{\n\\begin{downthreelvltree}\n')
    # tt = ThetaTree(example)
    tt.exec2latex(example,output=treefile)
    # treefile.write('\\end{downthreelvltree}\n}\n')
    



    
        
#
# 45/93                                            xxx
# 21/45                   xxx                                             xxx
# 9/21         xxx                     xxx                     xxx
# 3/9    xxx         xxx         xxx         xxx         xxx         xxx
# 0/3 xxx   xxx   xxx   xxx   xxx   xxx   xxx   xxx   xxx   xxx   xxx   xxx
#




