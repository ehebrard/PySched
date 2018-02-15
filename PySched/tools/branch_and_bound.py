#! /usr/bin/env python

from PySched import *

import random
import time


DEBUG_SRCH = False
PRINT_LATEX = False
DEBUG_SOL = None

# DEBUG_SOL = [True, False, False, True, True, False, True, False, True, False, True, False, False, True, True, False, False, True, True, False, True, False, False, True, True, False, False, True, True, False, False, True, False, True, True, False, False, True, True, False, False, True, False, True, False, True, False, True, True, False, False, True, True, False, True, False, True, False, True, False, False, True, True, False, False, True, True, False, True, False, False, True, True, False, False, True, True, False, False, True, False, True, True, False, False, True, True, False, False, True, False, True, False, True, False, True]
# DEBUG_LB = [0, 422, 355, 659, 490, 0, 910, 188, 355, 659, 0, 490, 910, 188, 422, 0]
# DEBUG_UB = [355, 659, 422, 1055, 910, 205, 1055, 490, 490, 1055, 355, 659, 1055, 422, 910, 205]




class BranchAndBound:
    def __init__(self, s, timetabling=False, overload=False, detectable=False, edgefinding=False, notfirstnotlast=False, new=False):
        self.schedule = s
        
        if new:
            self.disjuncts = [l for r in s.resources for l in r.getDisjunctVariables()]
            for i,d in enumerate(self.disjuncts):
                d.id = i
        else:
            self.disjuncts = [d for r in s.resources for d in r.getDisjuncts()]
            for d in self.disjuncts:
                d.suscribe()
    
        self.variables = SparseSet(len(self.disjuncts))
        self.variables.fill() 
        self.vsize = []
        self.decisions = []

        self.lb = 0
        self.ub = limit.INT_MAX
        
        if edgefinding:
            self.addEdgeFinding()
        elif overload:
            self.addOverloadChecking()
            
        if detectable:
            self.addDetectablePrecedences()
            
        if notfirstnotlast:
            self.addNotFirstNotLast()
            
        if timetabling:
            self.addTimetabling()
            
        self.weight = [1]*len(self.schedule.tasks)
            
    def cmpSol(self, a):
        for d,s in zip(self.disjuncts,DEBUG_SOL):
            if d.isGround() and d.value() != s:
                print 'WRONG PROPAGATE by %s on %s @%i'%(a, d,self.nb_decisions)
                sys.exit(1)
        for t,u in zip(self.schedule.tasks, DEBUG_UB):
            if t.latestCompletion() < u:
                print 'WRONG PROPAGATE by %s on %s @%i'%(a, t,self.nb_decisions)
                sys.exit(1)
        for t,l in zip(self.schedule.tasks, DEBUG_LB):
            if t.earliestStart() > l:
                print 'WRONG PROPAGATE by %s on %s @%i'%(a, t,self.nb_decisions)
                sys.exit(1)
        
    def on_track(self):
        valid = True
        for d in self.decisions:
            if DEBUG_SOL[d.id] != d.value():
                valid = False
                break
        return valid
        
    def init_parameters(self):
        self.nb_decisions = 0
        self.nb_fails = 0
        self.start_time = time.time()
      
    def print_head(self, out=sys.stdout):
        out.write('{0:{fill}{align}42}\n'.format(' ', fill='=', align='<'))
        out.write('    lb..ub     |  choices    fails    time\n')
        out.write('{0:{fill}{align}42}\n'.format(' ', fill='=', align='<'))
        
    def get_parameters(self):
        if self.ub == limit.INT_MAX:
            return '%6i..oo     | %8i %8i %7.2f'%(self.lb, self.nb_decisions, self.nb_fails, (time.time() - self.start_time))
        else:
            return '%6i..%-6i | %8i %8i %7.2f'%(self.lb, self.ub, self.nb_decisions, self.nb_fails, (time.time() - self.start_time))
            
    def print_parameters(self,final=False,out=sys.stdout):
        if final:
            out.write('{0:{fill}{align}42}\n'.format(' ', fill='=', align='<'))
        out.write(self.get_parameters()+'\n')
        
    def addOverloadChecking(self):
        for res in self.schedule.resources:
            cons = NoOverload(res)
            cons.suscribe()
            
    def addEdgeFinding(self):
        for res in self.schedule.resources:
            cons = EdgeFinding(res)
            cons.suscribe()
            cons = EdgeFinding(res, backward=True)
            cons.suscribe()
            
    def addDetectablePrecedences(self):
        for res in self.schedule.resources:
            cons = DetectablePrecedences(res, backward=True)
            cons.suscribe()
            
    def addNotFirstNotLast(self):
        for res in self.schedule.resources:
            cons = NotFirstNotLast(res)
            cons.suscribe()
            
    def addTimetabling(self):
        for res in self.schedule.resources:
            cons = Timetabling(res)
            cons.suscribe()       

    def branch(self):
        # if DEBUG_SOL is not None:
        #     if self.solution_branch():
        #         print 'wrong propagate', self.nb_decisions, self.nb_fails
        #         sys.exit(1)
        #     else:
        #         print 'solution ok'
        
        if len(self.decisions) > 0:
            last = self.decisions.pop()
            deduction = not last.value() #1-last.first
        
            if DEBUG_SRCH:
                print 'backtrack on %s'%last,

            self.variables.setSize(self.vsize.pop())
            self.schedule.restore()
        
            if DEBUG_SRCH:
                print '-> %s'%last,
        
            try:
                last.set(deduction)
                if DEBUG_SRCH:
                    print '-> %s'%last
            except Failure:
                self.branch()
        else:
            self.lb = self.ub+1
            if DEBUG_SRCH:
                print 'STOP [%i..%i]'%(self.lb,self.ub)
        
    def restart(self):
        # print '\nRESTART'
        # print self.get_parameters()
        while len(self.decisions):
            last = self.decisions.pop()
            # print last,
            self.schedule.restore()
            # print last
        self.variables.fill()
        # self.weight = [int(float(w) * .95) for w in self.weight]
        # print max(self.weight)
        
    def selectFirstVariable(self):
        d = None
        while d is None and len(self.variables)>0:
            d = self.disjuncts[self.variables[-1]]
            if d.isGround():
                d = None
                self.variables.pop()
        return d
        
    def getWeight(self, d):
        t1,t2 = d.getTasks()
        return self.weight[int(t1)] + self.weight[int(t2)]
        
    def selectWeightVariable(self):  
        x = None
        i = None
        while x is None and len(self.variables)>0:
            i = self.variables[-1]
            x = self.disjuncts[i]
            if x.isGround():
                x = None
                self.variables.pop()

        if x is not None:
            wx = self.getWeight(x)
            for j in self.variables:
                y = self.disjuncts[j]
                
                if not y.isGround():
                    wy = self.getWeight(y)
                    if wx * wx * y.domainSize() < wy * wy * x.domainSize():
                        i = j
                        x = y
                        wx = wy
                    
            self.variables.remove(i)
        
        if self.nb_decisions > 1 and (self.nb_decisions % 100) == 0:
            self.weight = [1 + int(float(w-1) * .995) for w in self.weight]
        
        return x
    
    def search(self, executeOnNode=lambda *args : None, executeOnSolution=lambda *args : None, executeOnLb=lambda *args : None, limit=None, heuristic=None):
        self.init_parameters()
        
        if heuristic is None:
            heuristic = self.selectFirstVariable
            
        base = 256
        rlimit = base
        
        while self.lb<self.ub and (limit is None or self.nb_decisions<limit):
            
            conflict = None
            if DEBUG_SOL is not None and self.on_track():
                conflict = self.schedule.propagate(check=self.cmpSol)
            else:
                conflict = self.schedule.propagate()
            executeOnNode(self)
            
            if DEBUG_SRCH:
                print '[%i..%i]'%(self.lb,self.ub)
                print ', '.join([str(d) for d in self.decisions]), 'propagate.'
                
            if conflict is None:
                if len(self.decisions)==0:
                    self.lb = self.schedule.getMakespanLB()
                    executeOnLb(self)
                    
                d = heuristic()
                if DEBUG_SRCH:
                    print 'select %s'%d
                if d is None:
                    self.ub = self.schedule.getMakespanLB()
                    if DEBUG_SRCH:
                        print 'Solution: %i'%(self.ub)
                    
                    self.schedule.setMakespanUB(self.ub)
                    executeOnSolution(self)
                    self.restart()
                    self.schedule.setMakespanUB(self.ub-1)
                else :
                    self.nb_decisions += 1
                    self.schedule.save()
                    self.vsize.append(len(self.variables))
                    i = random.randint(0,1)
                    self.decisions.append(d)
                    try:
                        if DEBUG_SRCH:
                            print 'decision on %s'%d
                        d.set(i==1)
                    except Failure as conflict:
                        # print "d conflict:", conflict.value
                        
                        x,y,k = conflict.value
                        self.weight[task_id(x)] += 1
                        self.weight[task_id(y)] += 1
                        
                        # print ' '.join(['%3i'%w for w in self.weight])
                        
                        self.nb_fails += 1
                        self.branch()
            else:
                for t in conflict.explanation():
                    self.weight[int(t)] += 1
                # print ' '.join(['%3i'%w for w in self.weight])
                    
                self.nb_fails += 1
                
                if len(self.decisions) > 1 and self.nb_fails >= rlimit:
                    self.restart()
                    # sys.stdout.write('   restart limit = %-10i\n'%(base))
                    base = int(float(base) * 1.3)
                    rlimit = self.nb_fails + base
                    # sys.stdout.write('   restart limit = %-10i\n'%(rlimit-self.nb_fails))
                    # self.print_head()
                else:
                    self.branch()
                # self.branch()
        
        self.lb = self.ub
        
    def dump_solution(self):
        print 'DEBUG_SOL =', [d.value() for d in self.disjuncts]
        print 'DEBUG_LB =', [t.earliestStart() for t in self.schedule.tasks]
        print 'DEBUG_UB =', [t.latestCompletion() for t in self.schedule.tasks]
        
    def signal_handler(self, signal, frame):
        print '{0:{fill}{align}42}'.format('Interrupted', fill='=', align='^')
        print self.get_parameters()
        sys.exit(1)
                
                
                
if __name__ == '__main__':
    random.seed(12345)
    s = Schedule()

    T = [Task(s, duration=random.randint(1,5)) for i in range(12)]
        
    rA = Resource(s, 'A')
    rA.extend(T[:4])

    rB = Resource(s, 'B')
    rB.extend(T[4:8])

    rC = Resource(s, 'C')
    rC.extend(T[8:12])

    for i in range(4):
        s.addJob([T[i], T[i+4], T[i+8]])
    
    
    bnb = BranchAndBound(s)
        
    bnb.print_head()
    bnb.search(executeOnSolution=lambda *args : bnb.print_parameters(), heuristic=bnb.selectFirstVariable)
    bnb.print_parameters(final=True)


