import os
import sys
import subprocess

def writeFile(exfilename, datfilenames, headers=None, scale=.45):
    
    if headers is None:
        headers = ['colorschedfigure']
    try:
        exfile = open('tex/src/%s.tex'%exfilename, 'w+')
        exfile.write('\\documentclass[usenames,dvipsnames,svgnames,table]{beamer}\n\\input{header}\n\\input{sched_macros}\n\\input{theta_macros}\n\n')
        exfile.write('\\begin{document}\n\\begin{frame}[fragile]\n')
        if isinstance(datfilenames, basestring):
            datfilenames = [datfilenames]
        else:
            headers = headers*len(datfilenames)
        for datfilename,header in zip(datfilenames,headers):
            if isinstance(datfilename, basestring):
                exfile.write('\\begin{%s}{%f}\n\\input{ex/%s.tex}\n\\end{%s}\n'%(header,scale,datfilename,header))
            else:
                exfile.write('\\begin{%s}{%f}\n'%(header,scale))
                count = 1
                for dat in datfilename:
                    exfile.write('\\uncover<%i>{\\input{ex/%s.tex}}\n'%(count,dat))
                    count += 1
                exfile.write('\\end{%s}\n'%header)
        exfile.write('\\end{frame}\n\\end{document}\n')
        exfile.close()
        
        print '\n\nsuccessfully created %s.tex in tex/src/\n => compile from tex/ with: pdflatex src/%s.tex\n'%(exfilename, exfilename)
    except IOError as e:
        print "I/O error({0}): {1}".format(e.errno, e.strerror)
    except:
        print 'unexpected error!'
    
# def compileLatex(exfilename):
#     try:
#         subprocess.call(['pdflatex', 'src/%s.tex'%exfilename], cwd='./tex', shell=False)
#     except error:
#         print 'error:', error
#
# def openFile(exfilename):
#     cp = subprocess.call(['open', '%s.pdf'%exfilename])

    
    
    