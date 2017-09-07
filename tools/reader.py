#! /usr/bin/env python

import sys

from PySched import *


def OSPReader(path, naming=True):
    infile = open(path, 'r')
    
    line = '#'
    while line.startswith('#'):
        line = infile.readline()

    bkb = [int(b.strip('*')) for b in line.split()]
    
    line = infile.readline().split()
    
    nb_job = int(line[0])
    nb_mach = int(line[1])
    
    s = Schedule()
    
    resname = map(chr, range(65, 65+nb_job+nb_mach))
    rid = 0
    
    for j in range(nb_job):
        tasks = infile.readline().split()
        
        J = Resource(s, resname[rid], [Task(s,int(d)) for d in tasks])
        # s.addResource(J)
        rid += 1
        
    for M in zip(*s.resources):
        res = Resource(s, resname[rid], M)
        
        if naming:
            for i,t in enumerate(res):
                t.label = '$%s_{%i}$'%(res.name, i)

        rid += 1
        
    return s, bkb[0], bkb[1]

                
                
                
if __name__ == '__main__':
    import tools.branch_and_bound as algo
    import random
    random.seed(12345)
    
    
    filepath = 'data/osp/gueret-prins/GP03-01.txt'
    if len(sys.argv) > 1:
        filepath = sys.argv[1]

    s, lb, ub = OSPReader(filepath)
    print ' known solution = %i'%ub
    
    bnb = algo.BranchAndBound(s, overload=(len(sys.argv)>2))
    
    bnb.search()
    



