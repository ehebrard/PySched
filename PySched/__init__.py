#! /usr/bin/env python

import sys
import operator
from collections import deque
# import _testcapi as limit

import structure.avl_tree as avl
import structure.theta_tree as theta
import structure.binary_heap as priority
from structure.difference_system import *

DEBUG_RELAX = False
DEBUG_PROP  = False
DEBUG_DP    = False
DEBUG_EDGE  = False
DEBUG_NLAST = False
DEBUG_JSP   = False
DEBUG_TT    = False
DEBUG_TTE   = False
DEBUG_CTT   = False
DEBUG_NLST  = False
DEBUG_NUIJ  = False


NO_CHANGE = 0
LB_CHANGE = 1
UB_CHANGE = 2
EVR_EVENT = 3

trace_file = open('tex/ex/trace.tex', 'w')





def start(t):
    return 2*(int(t)+1)

def end(t):
    return 2*int(t)+3
    
def event(x,y,k):
    # x - y <= k
    if x == 0:
        return '%s>=%i'%(var(y), -k)
    if y == 0:
        return '%s<=%i'%(var(x),  k)
    else:
        return '%s-%s<=%i'%(var(x), var(y), k)
        
def an_event(x,y):
    # x - y <= k
    if x == 0:
        return '%s>=k'%(var(y))
    if y == 0:
        return '%s<=k'%(var(x))
    else:
        return '%s-%s<=k'%(var(x), var(y))
    
def lit(x,b):
    v = var(x)
    sign = '>='
    if b>0 :
        sign = '<='
    return '%s%s%s'%(v,sign,pretty(abs(b)))

def end(t):
    return 2*int(t)+3
            
def on_lb(x):
    return x,0
    
def on_ub(x):
    return 0,x
    
def on_before((x,y)):
    return y,x

def on_after((x,y)):
    return x,y
    
            
# Task implemented essentially as an ID and a pointer to the schedule it belongs to (containing all the static and dynamic info)
class Task:
    
    def __init__(self, schedule, duration=1, demand=1, release=0, duedate=limit.INT_MAX, label=None, source=True, sink=True, update=True):
        self.schedule = schedule
        self.id = len(schedule.tasks)
        self.label = label
        
        minduration, maxduration = duration,duration
        if type(duration) is tuple:
            minduration, maxduration = duration
        self.schedule.addTask(self, minduration, maxduration, demand, release, duedate, source, sink, update)
        self.resources = []
        
    def __lshift__(self, t):
        self.schedule.addPrecedence(self.id, t.id)

    def __int__(self):
        return self.id
        
    def name(self):
        if self.label is None:
            return '$a_{%i}$'%(self.id+1)
        return self.label
        
    def minDuration(self):
        return self.schedule.getMinDuration(self.id)
        
    def maxDuration(self):
        return self.schedule.getMaxDuration(self.id)
        
    def demand(self):
        return self.schedule.demand[self.id]
        
    def earliestStart(self):
        return self.schedule.getEarliestStart(self.id)

    def latestCompletion(self):
        return self.schedule.getLatestCompletion(self.id)
        
    def earliestCompletion(self):
        return self.schedule.getEarliestCompletion(self.id)

    def latestStart(self):
        return self.schedule.getLatestStart(self.id)
        
    def ground(self):
        return self.latestCompletion() - self.earliestStart() == self.minDuration()
        
    def resourceName(self):
        # return ''.join([r.name for r in self.resources])
        rname = ''
        if len(self.resources) > 0:
            rname = self.resources[0].name
        return rname
        
    def __str__(self):
        if self.minDuration() != self.maxDuration():
            return '%s:[%i..(%i-%i)..%i] (%i)'%(self.name(), self.earliestStart(), self.minDuration(), self.maxDuration(), self.latestCompletion(), self.demand())
        return '%s:[%i..(%i)..%i] (%i)'%(self.name(), self.earliestStart(), self.minDuration(), self.latestCompletion(), self.demand())
        
    def strVar(self,start=0,mode='',row=1,factor=1.0):
        if start == 0:
            start = self.earliestStart()
            if mode == 'Right':
                start = self.latestCompletion() - self.maxDuration()
            elif mode != 'Left':
                start += (self.latestCompletion() - self.maxDuration())
                start /= 2
        return '\\ExtensibleTask{%f}{%f}{%f}{%i}{%i}{%s}{%s}'%(float(start)*factor, float(self.minDuration())*factor, float(self.maxDuration())*factor, self.demand(), row, self.resourceName(), self.name())
        
    def strInterval(self,mode='',row=1,factor=1.0):
        return '\\%sExtensibleVariableTask{%f}{%f}{%f}{%f}{%i}{%i}{%s}{%s}'%(mode, float(self.earliestStart())*factor, float(self.latestCompletion())*factor, float(self.minDuration())*factor, float(self.maxDuration())*factor, self.demand(), row, self.resourceName(), self.name())

    def strIntervalPruning(self,mode='',row=1,factor=1.0):
        rstr = ''
        previous = self.schedule.graph.trail[-1]
        for x,y,o,k in reversed(self.schedule.graph.edges[self.schedule.graph.trail[-1]:]):
            if x==0 and y==start(self) and o != limit.INT_MAX:
                # rstr += '\\PrunedLeftInterval{%i}{%i}{%i}{%i}{%s};'%(-k,-o,row,self.demand(), self.resourceName())
                rstr += '\\PrunedExtensibleTask{%f}{%f}{%f}{%i}{%i}{%s}{%s}%%1\n'%(float(-o)*factor, float(o-k)*factor, float(o-k)*factor, self.demand(), row, self.resourceName(), '')
            if x==end(self) and y==0 and o != limit.INT_MAX:
                # rstr += '\\PrunedRightInterval{%i}{%i}{%i}{%i};'%(k,o,row,self.demand(), self.resourceName())
                rstr += '\\PrunedExtensibleTask{%f}{%f}{%f}{%i}{%i}{%s}{%s}%%2\n'%(float(k)*factor, float(o-k)*factor, float(o-k)*factor, self.demand(), row, self.resourceName(), '')
        if self.ground():
            return rstr + '\\GroundTask{%f}{%f}{%i}{%i}{%s}{%s}'%(float(self.earliestStart())*factor, float(self.minDuration())*factor, self.demand(), row, self.resourceName(), self.name()) 
        return rstr + '\\%sExtensibleVariableTask{%f}{%f}{%f}{%f}{%i}{%i}{%s}{%s}'%(mode, float(self.earliestStart())*factor, float(self.latestCompletion())*factor, float(self.minDuration())*factor, float(self.maxDuration())*factor, self.demand(), row, self.resourceName(), self.name())
  
    def strProfile(self,mode='',row=1,factor=1.0):
        rstr = ''
        if self.earliestCompletion() > self.latestStart():
            dur = self.earliestCompletion() - self.latestStart()
            rstr += '\\MandatoryPartTask{%f}{%f}{%f}{%i}{%i}{%s}{%s}\n'%(float(self.earliestStart())*factor, float(dur)*factor, float(2*self.minDuration()-dur)*factor, self.demand(), row, self.resourceName(), self.name())
        else:
            rstr += '\\%sPhantomExtensibleVariableTask{%f}{%f}{%f}{%f}{%i}{%i}{%s}{%s}'%(mode, float(self.earliestStart())*factor, float(self.latestCompletion())*factor, float(self.maxDuration())*factor, float(self.maxDuration())*factor, self.demand(), row, self.resourceName(), self.name())
        return rstr
        
    def strGround(self,row=1,factor=1.0):
        return '\\GroundTask{%f}{%f}{%i}{%i}{%s}{%s}'%(float(self.earliestStart())*factor, float(self.minDuration())*factor, self.demand(), row, self.resourceName(), self.name()) 
        
        
class Trigger:
    def __init__(self, s, (x,y), p, a):
        self.schedule = s
        self.list = s.triggers[x][y]
        self.index = None
        self.algorithm = p
        self.arguments = a
        self.x = x
        self.y = y
        
    def pull(self,trace=False):
        self.algorithm.wake(*self.arguments,trace=trace)
        
    def post(self):        
        if DEBUG_PROP:
            if self.index is not None:
                print ' trying to post an already active trigger!!'
                sys.exit(1)
            print '  post %s on %s: [%s]'%(self, an_event(self.x,self.y), ', '.join([str(t) for t in self.list]))
                        
        self.index = len(self.list)
        self.list.append(self)
        
    def relax(self):
        if DEBUG_PROP:
            if self.index is None:
                print ' trying to relax an already inactive trigger!!'
                sys.exit(1)
            print '  relax %s from [%s]'%(self, ', '.join([str(t) for t in self.list]))
        
        last = self.list.pop()
        # self.schedule.trail.append(self)
        if last != self:
            self.list[self.index] = last
            last.index = self.index
        self.index = None
        
    def __str__(self):
        return str(self.algorithm)
        
        
class PropagationQueue:
    
    def __init__(self):
        self.queue = deque([])
        self.active = SparseSet()
        self.propagator = []
        
    def declare(self, propag):
        propag.id = len(self.propagator)
        self.propagator.append(propag.id)
        self.active.newElement(propag.id)
        
    def add(self, propag):
        self.active.add(propag.id)
        self.queue.append(propag)
        
    def __contains__(self, propag):
        return propag.id in self.active
               
    def pop(self):
        propag = self.queue.popleft()
        self.active.remove(propag.id)
        return propag
        
    def __len__(self):
        return len(self.queue)
        
    def __str__(self):
        return '[%s]'%(', '.join([str(propag) for propag in self.queue]))


# contains all the infos about tasks, the underlying difference system and the ressource constraints
# and manage backtracking, propagation, etc.
class Schedule:

    def __init__(self):
        self.tasks = []
        self.demand      = array.array('i')
        
        self.resources = []
        self.res_id = []
        
        self.graph = DifferenceSystem()
        self.graph.newElement(label='o') # source
        self.graph.newElement(label='m') # sink
        self.graph.schedule = self
        
        # to perform the transitive closure set this function to s.graph.FloydWarshall
        self.closure = lambda *args : False
        
        self.triggers = []
        self.propagationCounter = 0
        self.trail = []
        self.trailSize = [0]
        
        self.queue = PropagationQueue()
        
        self.jobs = []
        
        self.printCounter = {}
        
    def task(self,x):
        return self.tasks[x//2-1]
        
    def propagate(self, check=lambda *args : None, trace=False):
        if trace:
            print 'propagate %i/%i'%(self.propagationCounter,len(self.graph.edges))
        
        # Use a buffer because the order of the trigger lists may change 
        triggered = []
        
        culprit = None
        if self.propagationCounter < len(self.graph.edges) or len(self.queue)>0 :
            fixpoint = False
            try :
                while not fixpoint :
                    while self.propagationCounter < len(self.graph.edges) :
                        x, y, o, k = self.graph.edges[self.propagationCounter]
                        self.propagationCounter += 1
            
                        if trace:
                            print 'propagate event %s [%s] %s'%(event(x,y,k), ' '.join([event(a,b,c) for a,b,d,c in self.graph.edges[self.propagationCounter:]]), self.queue)
                            print '  -> triggers [%s]'%(' '.join([str(t) for t in self.triggers[x][y]]))
                
                        for trigger in self.triggers[x][y]:
                            triggered.append(trigger)
                    
                        while len(triggered)>0:
                            trigger = triggered.pop()
                            
                            culprit = trigger.algorithm
                            
                            trigger.pull(trace=trace)
                
                            check(culprit)
                        
                    if len(self.queue)>0 :
                        culprit = self.queue.pop()
                        if trace:
                            print '  -> propagate %s'%(culprit)
                        culprit.propagate(trace=trace)
                        
                        check(culprit)
                    
                    else:
                        # fixpoint = True
                        fixpoint = not self.closure()
                        
            except Failure as f:
                # print f.value, '//', culprit
                
                if trace:
                    print f.value
                    print 'FAIL!!'
                    
                check(None)
                return culprit
        return None
        
    def addResource(self, res):
        self.resources.append(res)
        res.schedule = self
        
    def addTask(self, t, minduration, maxduration, demand, release, duedate, source, sink, update):
        tid = t.label
        if tid is None:
            tid = '%i'%(t.id+1)
        self.demand.append(demand)
        self.tasks.append(t)
        self.graph.newElement(label='$s_{%s}$'%(tid)) # start
        self.graph.newElement(label='$e_{%s}$'%(tid)) # end
        self.graph.addEdge(start(t), end(t), -minduration, update=update)
        self.graph.addEdge(end(t), start(t),  maxduration, update=update)

        if duedate<limit.INT_MAX:
            self.graph.addEdge(end(t), 0, duedate, update=update)
        if release>0:
            self.graph.addEdge(0, start(t), -release, update=update)
        elif source:
            self.graph.addEdge(0, start(t), 0, update=update)
        if sink:
            self.graph.addEdge(end(t), 1, 0, update=update)
        for i in range(len(self.triggers)):
            while len(self.triggers[i]) <= end(t):
                self.triggers[i].append([])
        while len(self.triggers) <= end(t):
            self.triggers.append([[] for i in range(end(t)+1)])
            
    def addEdge(self, x, y, k, update=True):
        try:
            return self.graph.addEdge(x, y, k, update=update)
        except Failure as f:
            print 'Failure on %s'%(var(f.value))
            raise f

    def addPrecedence(self, ti, tj, transition=0, update=True):
        return self.graph.addEdge(end(ti), start(tj), -transition, update=update)
        
    def endBeforeStart(self, ti, tj, transition=0, update=True):
        return self.graph.addEdge(end(ti), start(tj), -transition, update=update)
        
    def endBeforeEnd(self, ti, tj, transition=0, update=True):
        return self.graph.addEdge(end(ti), end(tj), -transition, update=update)
        
    def startBeforeEnd(self, ti, tj, transition=0, update=True):
        return self.graph.addEdge(start(ti), end(tj), -transition, update=update)
        
    def startBeforeStart(self, ti, tj, transition=0, update=True):
        return self.graph.addEdge(start(ti), start(tj), -transition, update=update)    
        
    def setEarliestStart(self, t, b):
        return self.graph.addEdge(0, start(t), -b)
        
    def setEarliestCompletion(self, t, b):
        return self.graph.addEdge(0, end(t), -b)

    def setLatestStart(self, t, b):
        return self.graph.addEdge(start(t), 0, b)
    
    def setLatestCompletion(self, t, b):
        return self.graph.addEdge(end(t), 0, b)
     
    def setMakespanUB(self, ub):
        return self.graph.addEdge(1, 0, ub)
        
    def setMakespanLB(self, lb):
        return self.graph.addEdge(0, 1, lb)
        
    def close(self):
        return self.setMakespanUB(max([self.getLatestCompletion(t) for t in self.tasks]))
        
    def store(self, thing):
        self.trail.append(thing)
        
    def save(self):
        self.graph.save()
        self.trailSize.append(len(self.trail))
        
    def restore(self): 
        self.graph.restore()
        self.propagationCounter = len(self.graph.edges)
        previous = self.trailSize.pop()
        while len(self.trail) > previous:
            saved = self.trail.pop()
            saved.restore()
            
    def post(self, propag):
        self.queue.declare(propag)
        
    def activate(self, propag):
        if propag not in self.queue:
            self.queue.add(propag)
    
    def getMakespanLB(self):
        return -self.graph.distance[1][0]
        
    def getMakespanUB(self):
        return self.graph.distance[0][1]
        
    def getEarliestStart(self,t):
        return -self.graph.distance[start(t)][0]
        
    def getLatestCompletion(self,t):
        return self.graph.distance[0][end(t)] 
        
    def getEarliestCompletion(self,t):
        return -self.graph.distance[end(t)][0]
        
    def getLatestStart(self,t):
        return self.graph.distance[0][start(t)]
        
    def getMinDuration(self,t):
        return -self.graph.distance[end(t)][start(t)]
        
    def getMaxDuration(self,t):
        return self.graph.distance[start(t)][end(t)] 
        
    def getDemand(self,t):
        return self.demand[int(t)]
        
    # the minimum "energy" (= duration * demand) of t within the interval [x,y)
    def getEnergyIn(self, t, x, y):
        return self.getDemand(t) * max(0, self.getMinDuration(t) - max(0, x - self.getEarliestStart(t)) - max(0, self.LatestCompletion(t) - y))
               
    # the minimum duration of t within the interval [x,y)
    def getMandatoryIn(self, t, x, y):
        return max(0, min(y, self.getEarliestCompletion(t)) - max(x, self.getLatestStart(t)))
        
    # the minimum duration of t that does not belong to the mandatory part
    def getNonMandatory(self, t):
        return min(self.getMinDuration(t), self.getMinDuration(t) - (self.getEarliestCompletion(t) - self.getLatestStart(t)))
        
    # the minimum "energy" (duration * demand) of t that does not belong to the mandatory part
    def getFree(self, t):
        return self.getDemand(t) * self.getNonMandatory(t) 
    
    def isGround(self, t):
        return self.getMinDuration(t) == (self.getLatestCompletion(t) - self.getEarliestStart(t))
                
    def addJob(self, J):
        for t in J:
            t.job = len(self.jobs)
        self.jobs.append(J)
        for a,b in zip(J,J[1:]):
            self.graph.removeEdge(0, start(b))
            self.graph.removeEdge(end(a), 1)
            self.addPrecedence(a, b)
            
    def __str__(self):
        return '%s\n%s' % ('\n'.join([str(r) for r in self.resources]), '\n'.join([str(t) for t in self.tasks]))
        
    def printSequence(self, job, row, maxmkp):
        if len(job) <= 1:
            if len(job) == 1:
                print job[0].strInterval(row=row)
        else:
            est = job[0].earliestStart()
            let = maxmkp
            if maxmkp > job[-1].latestCompletion():
                let = job[-1].latestCompletion()
            dur = sum([self.getMaxDuration(t) for t in job])
            dem = max([self.demand[int(t)] for t in job])
            sep = float(let-est-dur)/float(len(job)-1)
            print '\\LeftBracket{%f}{%i}{%i}'%(est, row, dem)
    
            cur = float(est)
            print job[0].strVar(start=cur, row=row)
            cur += self.getMaxDuration(job[0])
            
            for t in job[1:]:
                print '\\Intertask{%f}{%f}{%i}{%i}'%(cur, cur+sep, row, dem)
                cur += sep
                print t.strVar(start=cur, row=row)
                cur += float(self.getMaxDuration(t))
                            
            print '\\RightBracket{%f}{%i}{%i}'%(let, row, dem)
            
    def printBounds(self):
        for t in self.tasks:
            print str(t)
            
    def setDemand(self, t, d):
        self.demand[int(t)] = d
                        
    def latexPrecedences(self, outfile, style='', tasks=None, decisions=True):
        if tasks is None:
            tasks = self.tasks
        
        ei = len(self.graph.edges)-1
        lvl = len(self.graph.trail)
        while lvl>0:
            lvl -= 1
            first = self.graph.trail[lvl]
            if lvl == 0 or not decisions:
                first -= 1
            while ei > first:
                x,y,k,o = self.graph.edges[ei]
                tx = self.tasks[task_id(x)]
                ty = self.tasks[task_id(y)]
                if tx != ty:
                    if x == end(tx) and y == start(ty):
                        outfile.write('\\Precedence{%s}{%s}{}\n'%(tx.name(), ty.name()))
                    elif x == start(tx) and y == end(ty):
                        if not start(ty) in self.graph.succ[end(tx)]:
                            outfile.write('\\BackPrecedence{%s}{%s}{back}\n'%(tx.name(), ty.name()))
                    elif k != limit.INT_MAX and x != 0 and y != 0:
                        print 'warning, weird precedence %s'%(event(x,y,k))
                ei -= 1
            if lvl > 0:
                x,y,k,o = self.graph.edges[ei]
                tx = self.tasks[task_id(x)]
                ty = self.tasks[task_id(y)]
                if tx != ty:
                    if x == end(tx) and y == start(ty):
                        outfile.write('\\Precedence{%s}{%s}{decision}\n'%(tx.name(), ty.name()))
                    elif x == start(tx) and y == end(ty):
                        if not start(ty) in self.graph.succ[end(tx)]:
                            outfile.write('\\BackPrecedence{%s}{%s}{backdecision}\n'%(tx.name(), ty.name()))
                    elif k != limit.INT_MAX and x != 0 and y != 0:
                        print 'warning, weird precedence %s'%(event(x,y,k))
                ei -= 1
                     
    def latexProfile(self, outfile, resources, offset=0, factor=1.0):
        self.close()
        for res in resources:
            profile = res.getProfile()
            outfile.write('\\CProfile{{%s}}{%i}{%f}{%s}{%f}\n'%(', '.join(['%f/%i/%f/%i'%(float(t1)*factor,u1,float(t2)*factor,u2) for (t1,u1),(t2,u2) in zip(profile, profile[1:])]), res.capacity, offset, res.name, float(profile[-1][0])*factor))
            
    def rowsFromPaths(self):
        # organise the tasks greedily along dfs-based paths
        T = sorted(self.tasks, key=self.getEarliestStart)
        rows = []
        visited = set([])
    
        curpath = []
        for t in T:
            if len(curpath)>0:
                rows.append(curpath)
                curpath = []
            v = t
            while v not in visited:
                # v.rank = len(rows)
                visited.add(v)
                curpath.append(v)
                for su in self.graph.succ[end(v)]:
                    if su > 1:
                        u = self.task(su)
                        if u not in visited:
                            v = u
        if len(curpath)>0:
            rows.append(curpath)
            curpath = []
            
        return rows
    
            
    def latex(self, outfile=sys.stdout, animated=False, tasks=None, precedences=True, windows=True, profile=None, profp=0, mandatory=False, rows=None, width=None, horizon=None, lb=False, ub=False, tics=None, ghost=[], offset=0, stop='', pruning=False, shifts=[], decisions=True):
        # h = max([self.getLatestCompletion(t) for t in self.tasks])
        self.close()
        if horizon is None:
            horizon = self.getMakespanUB()
        if width is None:
            width = min(horizon, 25.0)
        if mandatory:
            printMethod = Task.strProfile
        elif windows:
            if pruning:
                printMethod = Task.strIntervalPruning
            else:
                printMethod = Task.strInterval
        else:
            printMethod = Task.strVar
        
        if animated :
            if not self.printCounter.has_key(outfile):
                self.printCounter[outfile] = 0
            self.printCounter[outfile] += 1
            outfile.write('\\uncover<%i%s>{\n'%(self.printCounter[outfile]+offset, stop))
            
                
        
        if tics is None:
            tics = width//5
        
        f = float(width) / float(horizon)
        
        # print 'horizon=%i, width=%i, factor=%f'%(horizon,  width, f)
        
        if rows is None:
            rows = self.rowsFromPaths()
                            
                        
        rowsize = [max([1]+[self.getDemand(t) for t in p]) for p in rows]
        
        outfile.write('\\PrintTics{0,%i,...,%i}{%f}\n'%(tics, width, f))
        
        if f>1 :
            outfile.write('\\PrintScaledGrid{%i}{%i}{%f}\n'%(width, sum(rowsize), f))
        else:
            outfile.write('\\PrintGrid{%i}{%i}\n'%(width, sum(rowsize)))
               
        if tasks is None:
            tasks = [t for r in rows for t in r]
            
        while len(shifts)<len(tasks):
            shifts.append('')
            
        dshift = {}.fromkeys(tasks)
        for i, t in enumerate(tasks):
            dshift[t] = shifts[i]
        
        
        phantomed = dict([(t,'') for t in tasks])
        if not mandatory:
            for t in ghost:
                phantomed[t] = 'Phantom'
                
        tset = set(tasks)
        
        
        # selected_resources = set([r for r in t.resources for t in selected_tasks])
        row = 0
        for p,r in zip(rows, rowsize):
            for i, t in enumerate(p):
                if t in tset:
                    if t.ground() and not pruning:
                        outfile.write(t.strGround(row=row, factor=f)+'\n')
                    else:
                        if i==0 and len(p) > 1:
                            outfile.write(printMethod(t, row=row, factor=f, mode='Left%s'%phantomed[t])+'\n')
                        elif len(p) > 1 and i==(len(p)-1):
                            outfile.write(printMethod(t, row=row, factor=f, mode='Right%s'%phantomed[t])+'\n')
                        else:
                            outfile.write(printMethod(t, row=row, factor=f, mode='%s%s'%(dshift[t],phantomed[t]))+'\n')
            row += r


        if profile is not None:
            if profp > 0:
                self.latexProfile(outfile, resources=profile, offset=-(1+profp), factor=f)
            elif profp < 0:
                self.latexProfile(outfile, resources=profile, offset=row+max([r.capacity for r in profile])+.5-profp, factor=f)
            else:  
                self.latexProfile(outfile, resources=profile, offset=row, factor=f)
           
        if precedences :
            self.latexPrecedences(outfile, tasks=tasks, decisions=decisions)
                
        if ub:
            u = self.getMakespanUB()
            outfile.write('\\draw[ub] (%f,0) -- (%f,%i);\n'%(float(u)*f,float(u)*f,-row))
            
        if lb:
            l = self.getMakespanLB()
            outfile.write('\\draw[lb] (%f,0) -- (%f,%i);\n'%(float(l)*f,float(l)*f,-row))
                
        if animated:
            outfile.write('}\n')
            
        return rows

    
    def latexJSP(self, animated=True, maxm=limit.INT_MAX):
        self.close()
        if DEBUG_PROP:
            self.printBounds()
            # self.graph.printMatrix()
            print
        
        else:
        
            if animated :
                self.printCounter += 1
                print '\\uncover<%i>{'%self.printCounter
            for row, job in zip(range(len(self.jobs)), self.jobs):
                J = []
                for t in job:
                    if t.ground():
                        self.printSequence(J, row, maxm)
                        print t.strGround(row=row)
                    else:
                        J.append(t)
                self.printSequence(J, row, maxm)
        
            for res in self.resources:
                for t1 in res:
                    for t2 in res:
                        if t1.id != t2.id and t1.job != t2.job:
                            if start(t2) in self.graph.succ[end(t1)]:
                                print '\\Precedence{%s}{%s}'%(t1.label,t2.label)
            if animated:
                print '}'
                
    def latexOSP(self, animated=True, maxm=limit.INT_MAX, precedences=True):
        self.close()
    
        T = sorted(self.tasks, key=self.getLatestCompletion)
        f = 25.0 / float(T[-1].latestCompletion())
        
        row = []
        
        for t in T:
            t.rank = None
            for i,r in enumerate(row):
                if r<=t.earliestStart():
                    t.rank = i
            if t.rank is None:
                t.rank = len(row)
                row.append(t.latestCompletion)
                
        for t in T:
            if t.ground():
                print t.strGround(row=t.rank, factor=f)
            else:
                print t.strInterval(row=t.rank, factor=f)
                
        if precedences:
            for res in self.resources:
                # if res.name == 'I':
                for t1 in res:
                    for t2 in res:
                        if t1.id != t2.id:
                            if start(t2) in self.graph.succ[end(t1)]:
                                print '\\Precedence{%s}{%s}'%(t1.label,t2.label)

    
    def printTasks(self): 
        for t in self.tasks:
            print '%2i (%1i-%1i): [%2i..%2i]'%(int(t), self.getMinDuration(t), self.getMaxDuration(t), self.getEarliestStart(t), self.getLatestCompletion(t))
        print
        
        

class Resource(list):
    
    def __init__(self, schedule, name, init=[], capacity=1):
        self.name = name
        self.capacity = capacity
        self.extend(init)
        schedule.addResource(self)
    
    def append(self, t):
        if not isinstance(t, Task):
            raise TypeError, 'item is not of type %s' % Task
        t.resources.append(self)
        super(Resource, self).append(t) 
        
    def extend(self, tlist):
        for t in tlist:
            self.append(t)
        
    def getDisjuncts(self):
        if self.capacity == 1:
            return [Disjunct(t1,t2) for k in range(1, len(self)) for t1,t2 in zip(self, self[k:])]
        else:
            return self.getDisjunctVariables()
        
    def getDisjunctVariables(self):
        vars = []
        for k in range(1, len(self)):
            for t1,t2 in zip(self, self[k:]):
                l1 = Literal(self.schedule, end(t1), start(t2), 0)
                l2 = Literal(self.schedule, end(t2), start(t1), 0)
                if self.schedule.getDemand(t1) + self.schedule.getDemand(t2) > self.capacity:
                    d = Xor(self.schedule, l1, l2)
                else:
                    d = NotAnd(self.schedule, l1, l2)
                d.suscribe()
                l1.suscribe()
                l2.suscribe()
                vars.append(l1)
                vars.append(l2)
        return vars
            
    def getProfile(self):
        events = []
        for t in self:
            if self.schedule.getEarliestCompletion(t) > self.schedule.getLatestStart(t):
                events.append((self.schedule.getLatestStart(t), self.schedule.getDemand(t)))
                events.append((self.schedule.getEarliestCompletion(t), -self.schedule.getDemand(t)))
            
        events.sort(reverse=True)
    
        profile = [(0,0)]
        usage = 0
        while len(events)>0 :
            time = events[-1][0]
            while len(events)>0 and events[-1][0] == time:
                t,u = events.pop()
                usage += u
            profile.append((time, usage))
        profile.append((self.schedule.getMakespanUB(),0))
        
        return profile
        
    def getCumulativeTT(self, profile, bound):
        ttAfter = [0]*len(self)
        bounds = sorted([(bound(t), i) for i,t in enumerate(self)])
        
        ip = len(profile)-2
        ib = len(bounds)-1

        lt,lu = profile[-1]
        t,u = profile[ip]
        b,i = bounds[ib]
        C = 0
        
        if DEBUG_CTT:
            print 'prof: %s'%(' '.join([str(p) for p in profile]))
            
        while ip>=0 and ib>=0:
  
            if DEBUG_CTT:
                print '\nprof: [%i,%i):%i'%(t,lt,u)
                print 'evts: %i (%s)'%(b, self[i].name())       
            
            C += u*(lt-max(t,b))
            lb = b
            
            if t <= b:
                ttAfter[i] = C
                
                if DEBUG_CTT:
                    print 'ttAfter[%s]=%i, move to next event'%(self[i].name(),C)
                
                ib -= 1
                lt = b
                b,i = bounds[ib]
                
                
            if t >= lb:
                lt,lu = t,u
                ip -= 1
                t,u = profile[ip]
                
                if DEBUG_CTT:
                    print 'C=%i, move along profile'%(C)
        
        return ttAfter
        
        
    def JacksonPreemptiveSchedule(self):
        dur = [self.schedule.getMinDuration(t)*self.schedule.getDemand(t) for t in self]
        
        heap = priority.BinHeap(comparator=lambda x,y : self.schedule.getLatestCompletion(self[x]) < self.schedule.getLatestCompletion(self[y]))
        tasksByST = sorted(range(len(self)), key=lambda x : self.schedule.getEarliestStart(self[x]), reverse=True)
        t = self.capacity * self.schedule.getEarliestStart(tasksByST[-1])
        nt = limit.INT_MAX
        
        if DEBUG_JSP:
            for i in tasksByST:
                print self[i]
        
        while t != limit.INT_MAX:
            
            if DEBUG_JSP:
                print 'consider t=%i'%(t)
                
            # put all the available tasks in the heap
            while len(tasksByST)>0 and self.capacity * self.schedule.getEarliestStart(tasksByST[-1]) == t:
                if DEBUG_JSP:
                    print ' -', self[tasksByST[-1]]
                heap.insert(tasksByST.pop())
        
            # get the next event's time
            nt = limit.INT_MAX
            if len(tasksByST)>0:
                nt = self.capacity * self.schedule.getEarliestStart(tasksByST[-1])
                if DEBUG_JSP:
                    print '(next event @ time=%i)'%(nt)
                
            # at this point all the tasks in the heap are s.t. est<=t, and not task starts in [t+1,nt)
            while t<nt and not heap.empty():
                i = heap.min()
                
                if DEBUG_JSP:
                    print 'schedule', self[i], 
                if dur[i] <= nt-t:
                    if DEBUG_JSP:
                        print 'fully'
                    t += dur[i]
                    dur[i] = 0
                    mkp = t
                    heap.delMin()
                else:
                    if DEBUG_JSP:
                        print 'for %i'%(nt-t)
                    dur[i] -= (nt-t)
                    t = nt
                 
            # nothing happens between t and nt   
            if t<nt:
                t = nt
                
        if DEBUG_JSP:       
            print 'bound =', mkp / self.capacity
            
    def __str__(self):
        return 'res %s: {%s}'%(self.name, ','.join([str(int(t)) for t in self]))
        
        
class Propagator:
    
    def suscribe(self):
        
        if DEBUG_PROP:
            print 'suscribe %s'%(self)

        sched = self.resource.schedule
        for t in self.resource:
            Trigger(sched, on_lb(start(t)), self, []).post()
            Trigger(sched, on_ub(end(t)), self, []).post()
        
        sched.post(self)
        self.exp = []
        
        if DEBUG_PROP:
            print 'suscribed %s'%(self)
        
    def wake(self, trace=False):        
        self.resource.schedule.activate(self)
        
    def explanation(self):
        return self.exp
        
        
class NoOverload(Propagator):
    
    def __init__(self,resource):
        self.resource = resource
        self.theta_tree = theta.ThetaTree(self.resource)
        self.tasks = [t for t in self.resource]
        
    def propagate(self, trace=False):
        sched = self.resource.schedule
        self.theta_tree.clear()
        self.theta_tree.reinit()
        self.tasks.sort(key=sched.getLatestCompletion)

        for t in self.tasks:
            self.theta_tree.insert(t)
            if sched.getLatestCompletion(t) < self.theta_tree.getBound():
                # print 'Failure in overload checking'
                raise Failure(self)
        
        return False

    def __str__(self):
        return 'NoOverload(%s)'%self.resource
        
        
class DetectablePrecedences(Propagator): 
    
    def __init__(self,resource, backward=False):
        self.backward = backward
        self.resource = resource
        self.theta_tree = theta.ThetaTree(self.resource)
        self.pred_sorted_tasks = [t for t in self.resource]
        self.succ_sorted_tasks = [t for t in self.resource]
        self.bound = [0]*len(resource)
        
    def is_succ(self, sched, a, b):
        return sched.getEarliestCompletion(a) > sched.getLatestStart(b) or start(a) in sched.graph.succ[end(b)]
        
    def is_pred(self, sched, a, b):
        return self.is_succ(sched, b, a)
        
    def propagate(self, trace=False):  
        sched = self.resource.schedule
        change = self.boundFromNeighbors(sched, sched.getLatestStart, sched.getEarliestCompletion, self.is_succ, sched.setEarliestStart, 0, max, trace=trace)
        if self.backward and self.boundFromNeighbors(sched, sched.getEarliestCompletion, sched.getLatestStart, self.is_pred, sched.setLatestCompletion, limit.INT_MAX, min, trace=trace):
            change = True
        return change
            
    def boundFromNeighbors(self, sched, predSortKey, succSortKey, succCheck, setBound, no_bound, tightest, trace):
        self.theta_tree.clear()
        self.theta_tree.reinit()
        
        self.pred_sorted_tasks.sort(key=predSortKey)
        self.succ_sorted_tasks.sort(key=succSortKey)

        q = 0
        for i,a in enumerate(self.succ_sorted_tasks):
            self.bound[i] = no_bound
            
            if trace:
                print '[DP] precedences w.r.t %s'%(a)
        
            b = self.pred_sorted_tasks[q]
            while a != b and succCheck(sched, a, b):
                self.theta_tree.insert(b)
                q += 1
                
                if trace:
                    print '[DP]   - %s: %i'%(b, self.theta_tree.getBound()),
                
                self.bound[i] = tightest(self.bound[i], self.theta_tree.getBound())
                        
                if trace:
                    print '[DP]   ok: %s -> %i'%(a, self.bound[i])
                        
                b = self.pred_sorted_tasks[q]
                
        change = False
        for i,a in enumerate(self.succ_sorted_tasks):
            if setBound(a, self.bound[i]):
                change = True
                    
        return change

    def __str__(self):
        return 'DetectablePrecedences(%s)'%self.resource
        
        
class EdgeFinding(Propagator):
    
    def __init__(self,resource, backward=False):
        self.backward = backward
        self.resource = resource
        self.theta_tree = theta.ThetaTree(self.resource, backward)
        
    def propagate(self,trace=False):
        
        if DEBUG_EDGE:
            crit = 'ect'
            cmpop = '<'
            if self.backward:
                crit = 'lst'
                cmpop = '>'
            print '\npropagate %s'%(self.resource.name)
        
        # start with overload checking to fill the theta tree with "white nodes"
        sched = self.resource.schedule
        self.theta_tree.clear()
        self.theta_tree.reinit()
        
        
        target = sched.getLatestCompletion
        comp = operator.__lt__
        if self.backward:
            target = sched.getEarliestStart
            comp = operator.__gt__
        
        tasks = sorted(self.resource, key=target, reverse=self.backward)


        if DEBUG_EDGE:
            print 'tasks of %s by %s: [%s]'%(self.resource.name, crit, ' '.join([str(t) for t in tasks]))

        for t in tasks:
            self.theta_tree.insert(t)
            
            if DEBUG_EDGE:
                print ' - insert white %s, %s=%i %s %i'%(str(t), crit, target(t), cmpop, self.theta_tree.getBound())
                print self.theta_tree
            if comp(target(t), self.theta_tree.getBound()):
                # print 'Failure in overload checking'
                raise Failure(self)
        
        if DEBUG_EDGE:     
            print 'OK'
        
        change = False
        j = tasks[-1]
        while len(tasks)>1:
            self.theta_tree.insert(j, gray=True)
            
            if DEBUG_EDGE:
                print ' - paint %s gray, %s='%(str(j), crit),
                
            tasks.pop()
            j = tasks[-1]
            
            if DEBUG_EDGE:
                print '%i %s %i'%(target(j), cmpop, self.theta_tree.getGrayBound())
                print self.theta_tree
                
            
            while comp(target(j), self.theta_tree.getGrayBound()):
                i = self.theta_tree.getCulprit()
                
                if self.backward:
                    if DEBUG_EDGE:
                        print 'edge b/c of %s, must be before [%s]'%(str(i), ' '.join([str(t) for t in tasks]))
                    for k in tasks:
                        if sched.addPrecedence(i, k):
                            change = True      
                else:
                    if DEBUG_EDGE:
                        print 'edge b/c of %s, must be after [%s]'%(str(i), ' '.join([str(t) for t in tasks]))
                    for k in tasks:
                        if sched.addPrecedence(k, i):
                            change = True
                self.theta_tree.delete(i, gray=True)
                
                if DEBUG_EDGE:
                    print ' - turn %s white %i %s %i'%(str(i), target(j), cmpop, self.theta_tree.getGrayBound())
                    print self.theta_tree 
        
        return change

    def __str__(self):
        return 'EdgeFinding(%s)'%self.resource
        
        
class NotFirstNotLast(Propagator):

    def __init__(self,resource):
        self.resource = resource
        self.theta_tree = theta.ThetaTree(self.resource)
        self.ateht_eert = theta.ThetaTree(self.resource, backward=True)

    def propagate(self, trace=False):

        if DEBUG_NLST:
            print '\npropagate %s'%(self.resource.name)

        sched = self.resource.schedule
        change = self.prune_nlnf(tree=self.theta_tree, left=sched.getLatestStart, right=sched.getLatestCompletion, tightest=min, setBound=sched.setLatestCompletion, past=operator.__gt__, nobound=limit.INT_MAX)
        change |= self.prune_nlnf(tree=self.ateht_eert, left=sched.getEarliestCompletion, right=sched.getEarliestStart, tightest=max, setBound=sched.setEarliestStart, past=operator.__lt__, nobound=0)
        return change
        
    def prune_nlnf(self, tree, left, right, tightest, setBound, past, nobound):
        tree.clear()
        tree.reinit()

        if DEBUG_NLST:
            lstr = 'lst'
            rstr = 'lct'
            ttstr = 'ect'
            paststr = '>'
            bstr = '<='
            if past==operator.__lt__:
                paststr = '<'
                lstr = 'ect'
                rstr = 'est'
                ttstr = 'lst'
                bstr = '>='
        
        left_sorted_tasks = sorted(self.resource, key=left, reverse=(nobound==0))
        right_sorted_tasks = sorted(self.resource, key=right, reverse=(nobound==0))
        bounds = [nobound]*len(self.resource)

        q = 0
        for i,t in enumerate(right_sorted_tasks):
            
            if DEBUG_NLST:
                print 'next task by %s: %s'%(rstr,t)
            
            while q<len(left_sorted_tasks) and past(right(t), left(left_sorted_tasks[q])):
                j = left_sorted_tasks[q]
                if DEBUG_NLST:
                    print '  - %s is in NL(%s)'%(j, t.name())
                
                tree.insert(j)
                q += 1
                
            ectNL = tree.getBoundMinus(t)
                          
            if past(ectNL, left(t)):
                
                if DEBUG_NLST:
                    print '  -> %s(NL(%s)) = %i %s %s(%s)=%i -> ub=%i'%(ttstr, t.name(), ectNL, paststr, lstr, t.name(), left(t), left(j))
                
                bounds[i] = tightest(bounds[i], left(j))
            else:
                if DEBUG_NLST:
                    print '  -> %s(NL(%s)) = %i %s %s(%s)=%i'%(ttstr, t.name(), ectNL, bstr, lstr, t.name(), left(t))
                

        change = False
        for i,t in enumerate(right_sorted_tasks):
            change |= setBound(t, bounds[i])
            if DEBUG_NLST:
                print '  -> %s(%s) %s %i'%(rstr, t.name(), bstr, bounds[i]), change
            
        return change


    def __str__(self):
        return 'NotFirstNotLast(%s)'%self.resource
    
    
class Timetabling(Propagator):
    
    def __init__(self,resource):
        self.resource = resource
        self.heap = priority.BinHeap(comparator=operator.__gt__, score=self.resource.schedule.getDemand)
        
    
    def _getLeftForbidden(self, t, best, node):
        if node is None:
            return best
            
        a,b = node.key
        if b>t:
            return self._getLeftForbidden(t, node, node.leftChild)
        elif best is not None:
            return best
        else:
            return self._getLeftForbidden(t, None, node.rightChild)
        
    def getLeftForbidden(self, t, tree): # return the leftmost interval [a,b) in tree such that b>t
        return self._getLeftForbidden(t, None, tree.rootNode)
        
    def _getRightForbidden(self, t, best, node):
        if node is None:
            return best
            
        a,b = node.key
        if a<t:
            return self._getRightForbidden(t, node, node.rightChild)
        elif best is not None:
            return best
        else:
            return self._getRightForbidden(t, None, node.leftChild)
        
    def getRightForbidden(self, t, tree): # return the leftmost interval [a,b) in tree such that b>t
        return self._getRightForbidden(t, None, tree.rootNode)
        
    def propagate(self):
        change = False        
        sched = self.resource.schedule
        profile = self.resource.getProfile()

        lt, lu = profile[0]
        F = []
        
        for t,u in profile[1:]:
            # print "[%i,%i]:%i"%(lt,t,lu)
            if lu > self.resource.capacity:
                raise Failure(self)
            
            F.append((lu,(lt,t)))
            lt,lu = t,u
            
        F.sort(reverse=True)
        
        if DEBUG_TT:
            print F
         
        tasks = sorted(self.resource, key=sched.getDemand)
        
        tree = avl.AVLTree()
        j = 0
        for t in tasks:
            if DEBUG_TT:
                print t
            
            tu = sched.getDemand(t)
            while j < len(F):
                u,(a,b) = F[j]
                if u+tu > self.resource.capacity:
                    tree.myinsert((a,b))
                else:
                    break
                j+=1
                    
            if DEBUG_TT:
                print tree.out()
            
            node = self.getLeftForbidden(sched.getEarliestStart(t), tree)
            if node is not None:
                a,b = node.key
                
                if DEBUG_TT:
                    print '[%i,%i]'%(a,b)
                
                et = min(sched.getEarliestCompletion(t), sched.getLatestStart(t))
                if et > a:
                    if b > sched.getLatestStart(t):
                        b = sched.getLatestStart(t)
                        
                    if DEBUG_TT:
                        print " ==> pruning! %s >= %i"%(t,b)
                        
                    if sched.setEarliestStart(t, b):
                        change = True
                    
                    
            node = self.getRightForbidden(sched.getLatestCompletion(t), tree)
            if node is not None:
                a,b = node.key
                
                if DEBUG_TT:
                    print '[%i,%i]'%(a,b)
                
                et = max(sched.getLatestStart(t), sched.getEarliestCompletion(t))
                if et < b:
                    if a < sched.getEarliestCompletion(t):
                        a = sched.getEarliestCompletion(t)
                        
                    if DEBUG_TT:
                        print " <== pruning! %s <= %i"%(t,a)
                        
                    if sched.setLatestCompletion(t, a):
                        change = True
                    
        return change
                    
    def __str__(self):
        return 'Timetabling(%s)'%self.resource
        
        
class NuijtenCumulativeEdgeFinding(Propagator):
    
    def __init__(self,resource):
        self.resource = resource
            
    def propagate(self,trace=False):
        s = self.resource.schedule

        change = self.prune_ef(release=s.getEarliestStart, duedate=s.getLatestCompletion, setBound=s.setEarliestStart)

        ub = s.getMakespanUB()
     
        return change
    
    def prune_ef(self, release, duedate, setBound):
        
        if DEBUG_NUIJ:
            print 'propagate', self
    
        sched = self.resource.schedule
    
        inf=limit.INT_MAX
        rdiv=lambda x,y: -(-x // y)
        energy=lambda x: sched.getMinDuration(x) * sched.getDemand(x)
        demand=lambda x: sched.getDemand(x)
        
        C = self.resource.capacity
    
        cbound = [-inf]*len(self.resource)
        bound = [-inf]*len(self.resource)
        
        tasks = sorted(self.resource, key=release)
        
        for k,tk in enumerate(tasks):
            
            if DEBUG_NUIJ:
                print tk
            
            W, H, maxW, maxEst = 0, inf, -inf, duedate(tk)
            for i,ti in reversed(zip(range(len(tasks)), tasks)):
                
                if DEBUG_NUIJ:
                    print ' - [maxW=%s,maxEst=%s]'%(pretty(maxW),pretty(maxEst)), ti, 
                
                if duedate(ti) <= duedate(tk):
                    
                    W += energy(ti)
                    
                    if DEBUG_NUIJ:
                        print 'in O< (E(O< = %i))'%W
                    
                    if W > C * (duedate(tk) - release(ti)):
                        
                        print 'FAIL! (W=%i > C=%i * D=%i=|%i-%i|)'%(W,C, (duedate(tk) - release(ti)), duedate(tk), release(ti))
                        raise Failure(int(tk))
                    if W + C * release(ti) > maxW + C * maxEst: 
                        maxW, maxEst = W, release(ti)
                else:
                    restW = maxW - (C - demand(ti)) * (duedate(tk) - maxEst)
                    
                    if DEBUG_NUIJ:
                        print 'in O> rest(O<, %s)=%i [%i - (%i * %i) * (%i - %i)]'%(ti.name(), restW, maxW, C, demand(ti), duedate(tk), maxEst)
                    
                    if restW > 0:
                        cbound[i] = maxEst + rdiv(restW, demand(ti))
            
            for i,ti in enumerate(tasks):
                
                if DEBUG_NUIJ:
                    print ' *', ti,
                
                if duedate(ti) <= duedate(tk):
                    H = min(H, C * (duedate(tk) - release(ti)) - W)
                    W -= energy(ti)
                    
                    if DEBUG_NUIJ:
                        print ' update W=%i, H=%s'%(W,pretty(H))
                else:
                    if DEBUG_NUIJ:
                        print 'W=%i, H=%s'%(W,pretty(H)), 
                    
                    if C * (duedate(tk) - release(ti)) < W + energy(ti):
                        # setBound(ti, bound[i])
                        bound[i] = max(bound[i], cbound[i])
                        
                        if DEBUG_NUIJ:
                            print 'new bound (1) for %s: %i'%(ti.name(), cbound[i])
                        
                    if H < energy(ti):
                        restW = maxW - (C - demand(ti)) * (duedate(tk) - maxEst)
                        if restW > 0:
                            bound[i] = max(bound[i], maxEst + rdiv(restW, demand(ti)))
                            
                            if DEBUG_NUIJ:
                                print 'new bound (2) for %s: %i'%(ti.name(), (maxEst + rdiv(restW, demand(ti))))
                                   
        change = False
        for i,t in enumerate(tasks):
            change |= setBound(t, bound[i])
            
        return change
        

    def __str__(self):
        return 'NuijtenCumulativeEdgeFinding(%s)'%self.resource
                    
            
class Literal:

    def __init__(self, schedule, x, y, k):
        # True iff (x - y <= k) 
        self.tasks = schedule.tasks[task_id(x)], schedule.tasks[task_id(y)]
        self.p = (x, y)
        self.k = (k, -(k+1))
        self.triggers = []
        self.schedule = schedule

    def value(self):
        d = self.schedule.graph.distance
        x = self.p[0]
        y = self.p[1]
        if d[y][x] <= self.k[0]:
            return True
        elif d[x][y] <= self.k[1]:
            return False
        return None
        
    def domainSize(self):
        t1,t2 = self.tasks
        return (t2.latestStart() - t1.earliestCompletion()) + (t1.latestStart() - t2.earliestCompletion()) 
        
    def getTasks(self):
        return self.tasks
            
    def suscribe(self):
        
        if DEBUG_PROP:
            print 'suscribe %s (%i)'%(self, len(self.triggers))
            
        for i in range(2):
            self.triggers.append(Trigger(self.schedule, on_lb(self.p[i]), self, [1-i]))
            self.triggers.append(Trigger(self.schedule, on_ub(self.p[1-i]), self, [i]))
        self.restore()            
        
        if DEBUG_PROP:
            print 'suscribed %s (%i)'%(self, len(self.triggers))
            
    def relax(self):
        for trigger in self.triggers:
            trigger.relax()
            
    def restore(self):
        # self._val_ = None
        for trigger in self.triggers:
            trigger.post()
            
    def isGround(self):
        return self.value() is not None

    def set(self, v):
        
        cur = self.value()
        if cur is None:
            
            if DEBUG_PROP:
                print '  set', self, '= %r'%v
            
            self._val_ = v
            self.schedule.store(self)
            self.relax()
            i = int(v)
            
            
            self.schedule.graph.addEdge(self.p[1-i], self.p[i], self.k[1-i])
        
            if DEBUG_PROP:
                print '  -> ', self
    
        elif cur != v:
    
            if DEBUG_PROP:
                print 'fail!'
            raise Failure(self)

    def wake(self, i, trace=False):
        lb = self.schedule.graph.getLB(self.p[1-i])
        ub = self.schedule.graph.getUB(self.p[i])
        k = self.k[1-i]
        
        if trace:
            print ' - lit %s react to %s & %s'%( self, lit(self.p[1-i], -lb),  lit(self.p[i], ub) ),
                
        if lb - k > ub:
            self.set(i==0)
            
        if trace:
            print self

    def __str__(self):
        return '%s <= %s'%(var(self.p[0]), var(self.p[1]))
            

class Xor:

    def __init__(self, schedule, a, b):
        # a xor b
        self.lits = (a, b)
        self.triggers = []
        self.schedule = schedule
        self.tasks = list(set([schedule.tasks[task_id(t)] for i in range(2) for t in self.lits[i].p]))
            
    def suscribe(self):
        
        if DEBUG_RELAX:
            print 'suscribe %s (%i)'%(self, len(self.triggers))
            
        s = self.schedule
        a = self.lits[0].p
        b = self.lits[1].p
            
        self.triggers.append(Trigger(s, on_before(a), self, [0])) # a becomes true
        self.triggers.append(Trigger(s, on_after(a), self, [0])) # a becomes false
        if on_before(a) != on_after(b):
            self.triggers.append(Trigger(s, on_before(b), self, [1])) # b becomes true
            self.triggers.append(Trigger(s, on_after(b), self, [1])) # b becomes false
            
        self.restore()            
        
        if DEBUG_RELAX:
            print 'suscribed %s (%i)'%(self, len(self.triggers))
            
    def relax(self):
        for trigger in self.triggers:
            trigger.relax()
            
    def restore(self):
        for trigger in self.triggers:
            trigger.post()

    def wake(self, i, trace=False):
        a = self.lits[i] # a has changed
        b = self.lits[1-i] # update b
        
        if trace:
            print ' - xor react to %s'%( a ),
            
        v = a.value()
        
        if v is not None:
            self.schedule.store(self)
            self.relax()
            
            b.set( not v )    
        
        if trace:
            print self
            
    def explanation(self):
        return self.tasks

    def __str__(self):
        return '%s <=> %s'%(self.lits[0], self.lits[1])
        
        
class NotAnd:

    def __init__(self, schedule, a, b):
        # not(a and b)
        self.lits = (a, b)
        self.triggers = []
        self.schedule = schedule
            
    def suscribe(self):
        
        if DEBUG_RELAX:
            print 'suscribe %s (%i)'%(self, len(self.triggers))
            
        s = self.schedule
        a = self.lits[0].p
        b = self.lits[1].p
            
        self.triggers.append(Trigger(s, on_before(a), self, [0])) # a becomes true
        self.triggers.append(Trigger(s, on_after(a), self, [0])) # a becomes false
        if on_before(a) != on_after(b):
            self.triggers.append(Trigger(s, on_before(b), self, [1])) # b becomes true
            self.triggers.append(Trigger(s, on_after(b), self, [1])) # b becomes false
            
        self.restore()            
        
        if DEBUG_RELAX:
            print 'suscribed %s (%i)'%(self, len(self.triggers))
            
    def relax(self):
        for trigger in self.triggers:
            trigger.relax()
            
    def restore(self):
        for trigger in self.triggers:
            trigger.post()

    def wake(self, i, trace=False):
        a = self.lits[i] # a has changed
        b = self.lits[1-i] # update b
        
        if trace:
            print ' - not and react to %s'%( a ),
            
        v = a.value()
        
        if v is not None:
            self.schedule.store(self)
            self.relax()
            
            if v:
                b.set( False )    
        
        if trace:
            print self

    def __str__(self):
        return '%s # %s'%(self.lits[0], self.lits[1])
        

class Disjunct: # so far it seems faster than the literal/xor option

    def __init__(self, t1, t2):
        # (end(t1) - start(t2) <= 0) xor (end(t2) - start(t1) <= 0)
        self.tasks = (t1, t2)
        self.triggers = []
        self.schedule = t1.schedule
        self._val_ = None
        
    def value(self):
        return self._val_
        
    def domainSize(self):
        t1,t2 = self.tasks
        return (t2.latestStart() - t1.earliestCompletion()) + (t1.latestStart() - t2.earliestCompletion()) 
        
    def getTasks(self):
        return self.tasks
            
    def suscribe(self):
        
        if DEBUG_RELAX:
            print 'suscribe %s (%i)'%(self, len(self.triggers))

        t = self.tasks
        for i in range(2):
            self.triggers.append(Trigger(self.schedule, on_lb(start(t[i])), self, [1-i]))
            self.triggers.append(Trigger(self.schedule, on_ub(end(t[i])), self, [i]))
        
        self.restore()            
        
        if DEBUG_RELAX:
            print 'suscribed %s (%i)'%(self, len(self.triggers))
            
    def relax(self):
        for trigger in self.triggers:
            trigger.relax()
            
    def restore(self):
        self._val_ = None
        for trigger in self.triggers:
            trigger.post()
            
    def isGround(self):
        return self._val_ is not None
            
    def flip(self):
        return not self._val_
            
    def set(self, v, trace=False):
        
        if trace:
            print 'set %s (first=%i)'%(self,int(v)), 
        
        self._val_ = v
        
        if trace:
            print '->', self
        
        self.schedule.store(self)
        self.relax()
        return self.schedule.graph.addEdge(end(self.tasks[int(v)]), start(self.tasks[1-int(v)]), 0)


    def wake(self, i, trace=False):        
        t1 = self.tasks[i]
        t2 = self.tasks[1-i]
        
        if trace:
            print ' - react to %s & %s'%( lit(end(t1), -t1.earliestCompletion()),  lit(start(t2), t2.latestStart()))
                
        if t1.earliestCompletion() > t2.latestStart():
            
            if trace:
                print '   * add precedence %s <= %s'%( var(end(t2)), var(start(t1)))
            
            return self.set(i==0)
        
        return False
            
    def propagate(self):
        change = self.wake(0)
        change |= self.wake(1)
        return change
        
    def explanation(self):
        return self.tasks

    def __str__(self):
        if self.value() is None:
            return 't%i <> t%i'%(int(self.tasks[0]), int(self.tasks[1]))
        elif not self.value():
            return 't%i < t%i'%(int(self.tasks[0]), int(self.tasks[1]))
        else:
            return 't%i < t%i'%(int(self.tasks[1]), int(self.tasks[0]))
            
            
class Overlap: # literal with three values: before, after, during

    def __init__(self, t1, t2):
        # Disjunct(t1,t2) xor (start(t1) - end(t2) <= -1  and start(t2) - end(t1) <= -1)
        self.tasks = (t1, t2)
        self.triggers = []
        self.schedule = t1.schedule
        self._val_ = None # True (Disjunct), False (Overlap)
        self.disjunct = Disjunct(t1, t2) # can become disjunct or two precedences
        #self.domain = 7 # 1 = before, 2 = after, 4 = during
         
    def value(self):
        return self._val_
           
    def suscribe(self):
        
        if DEBUG_PROP:
            print 'suscribe %s (%i)'%(self, len(self.triggers))

        t = self.tasks
        for i in range(2):
            self.triggers.append(Trigger(self.schedule, on_lb(start(t[i])), self, [1-i]))
            self.triggers.append(Trigger(self.schedule, on_ub(end(t[i])), self, [i]))
        
        self.restore()            
        
        if DEBUG_PROP:
            print 'suscribed %s (%i)'%(self, len(self.triggers))
         
    def relax(self):
        for trigger in self.triggers:
            trigger.relax()
               
    def restore(self):
        if self._val_:
            self.disjunct.relax()
        for trigger in self.triggers:
            trigger.post()
        self._val_ = None
        
    def isGround(self):
        if self._val_ is not None:
            return not self._val_ or self.disjunct.isGround()
        return False
        
    # def getValue(self):
    #     return (self, self.value)
                
    def set(self, val):
        
        if DEBUG_PROP:
            print 'set %s (state=%r)'%(self,val), 
        
        self._val_ = val
        if self._val_ :
            self.schedule.store(self)
            self.relax()
            self.disjunct.suscribe()
        
        if DEBUG_PROP:
            print '->', self


    def wake(self, i, trace=False):        
        t1 = self.tasks[i]
        t2 = self.tasks[1-i]
        
        if DEBUG_PROP:
            print ' - react to %s & %s'%( lit(end(t1), -t1.earliestCompletion()),  lit(start(t2), t2.latestStart()))
                
        if t1.earliestCompletion() > t2.latestStart():
            
            if DEBUG_PROP:
                print '   * add precedence %s <= %s'%( var(end(t2)), var(start(t1)))
            
            self.set(1-i)

    def __str__(self):
        if self.value() is None:
            return 't%i <> t%i'%(int(self.tasks[0]), int(self.tasks[1]))
        elif not self.value():
            return 't%i < t%i'%(int(self.tasks[0]), int(self.tasks[1]))
        else:
            return 't%i < t%i'%(int(self.tasks[1]), int(self.tasks[0]))

