import os
import subprocess

def writeFile(exfilename, datfilenames, scale=.45):
    try:
        exfile = open('tex/src/%s.tex'%exfilename, 'w+')
        exfile.write('\\documentclass[usenames,dvipsnames,svgnames,table]{beamer}\n\\input{header}\n\\input{sched_macros}\n\\input{theta_macros}\n\n')
        exfile.write('\\begin{document}\n\\begin{frame}[fragile]\n')
        if isinstance(datfilenames, basestring):
            datfilenames = [datfilenames]
        for datfilename in datfilenames:
            if isinstance(datfilename, basestring):
                exfile.write('\\begin{colorschedfigure}{%f}\n\\input{ex/%s.tex}\n\\end{colorschedfigure}\n'%(scale,datfilename))
            else:
                exfile.write('\\begin{colorschedfigure}{%f}\n'%scale)
                count = 1
                for dat in datfilename:
                    exfile.write('\\uncover<%i>{\\input{ex/%s.tex}}\n'%(count,dat))
                    count += 1
                exfile.write('\\end{colorschedfigure}\n')
        exfile.write('\\end{frame}\n\\end{document}\n')
        exfile.close()
        
        print '\n\nsuccessfully created %s.tex.tex in tex/src/\n => compile from tex/ with: pdflatex src/%s.tex\n'%(exfilename, exfilename)
    except error:
        print 'error:', error
    
# def compileLatex(exfilename):
#     try:
#         subprocess.call(['pdflatex', 'src/%s.tex'%exfilename], cwd='./tex', shell=False)
#     except error:
#         print 'error:', error
#
# def openFile(exfilename):
#     cp = subprocess.call(['open', '%s.pdf'%exfilename])

    
    
    