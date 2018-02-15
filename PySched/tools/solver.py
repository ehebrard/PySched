#! /usr/bin/env python

import argparse
import signal
import random
import sys


import PySched.tools.reader as reader
import PySched.tools.branch_and_bound as algo


def cmdLineSolver():
    """
    simple usage of the module: read a file and solve it
    """  
    
    parser = argparse.ArgumentParser(description='Minimalistic Scheduling solver')
    parser.add_argument('file',type=str,help='path to instance file')
    parser.add_argument('--new', action='store_true')
    parser.add_argument('--overload', action='store_true')
    parser.add_argument('--detectable', action='store_true')
    parser.add_argument('--edgefinding', action='store_true')
    parser.add_argument('--notfirstnotlast', action='store_true')
    parser.add_argument('--closure', action='store_true')
    parser.add_argument('--solution', action='store_true')
    parser.add_argument('--noname', action='store_true')
    parser.add_argument('--ub',type=int,default=-1,help='Initial upper bound')
    parser.add_argument('--seed',type=int,default=12,help='Random seed')
    parser.add_argument('--latex',type=str,default=None,help='Trace Tex file')
    parser.add_argument('--heuristic',type=str,default='any',help='Disjunct selection heuristic')
    
    

    args = parser.parse_args()
    
    
    
    
    s, lb, ub = reader.OSPReader(args.file, naming=(not args.noname))
    
    if args.closure:
        s.closure = s.graph.FloydWarshall   
     
    if args.ub >= 0:
        s.setMakespanUB(args.ub) 
    
    random.seed(args.seed)
    bnb = algo.BranchAndBound(s, overload=args.overload, detectable=args.detectable, edgefinding=args.edgefinding, notfirstnotlast=args.notfirstnotlast, new=args.new)
    signal.signal(signal.SIGINT, bnb.signal_handler)
    
    strategy = bnb.selectFirstVariable
    if args.heuristic == 'weight':
        strategy = bnb.selectWeightVariable
    
    
    print '\n target = [%i..%i]'%(lb,ub)
    
    
    if args.solution:
        bnb.search(executeOnSolution=lambda *args : bnb.dump_solution(), heuristic=strategy)
    elif args.latex is not None:
        outfile = open(args.latex, 'w')
        span = s.getMakespanUB()
        rows = s.rowsFromPaths()
        printsched = lambda x : s.latex(outfile=outfile, ub=True, lb=True, animated=True, windows=False, rows=rows, width=25, horizon=span, precedences=True)
        printsol = lambda x : s.latex(outfile=outfile, ub=True, lb=True, animated=True, windows=True, rows=rows, width=25, horizon=span, precedences=False)
        bnb.search(limit=None, executeOnNode=printsched, executeOnSolution=printsol, heuristic=strategy)
    else:
        bnb.print_head()
        bnb.search(executeOnSolution=lambda *args : bnb.print_parameters(), heuristic=strategy)
        bnb.print_parameters(final=True)
        
    
    
if __name__ == '__main__':
    cmdLineSolver()
    
    



