
"""
Binary heap.

File: binary_heap.py
Author: Emmanuel Hebrard (hebrard@laas.fr)

"""

import operator


def leftChild(x):
    return 2*x

def rightChild(x):
    return 2*x+1
    
def parent(x):
    return x//2
    


class BinHeap:
    def __init__(self, comparator=operator.lt, score=None, init=None):
        if score is None:
            self.compare = comparator
        else:
            self.compare = lambda x,y : comparator(score(x), score(y))
        
        if init is None:
            self.heapList = [0]
            # self.value = [0]
        else:
            i = len(init) // 2
            self.heapList = [0] + init[:]
            # self.value = [0] + [score(x) for x in init]
            while (i > 0):
                self.percDown(i)
                i = i - 1
                
    def clear(self):
        if not self.empty():
            self.heapList = [0]

    def percUp(self,i):
        while parent(i) > 0:
          if self.compare(self.heapList[i], self.heapList[parent(i)]):
             tmp = self.heapList[parent(i)]
             self.heapList[parent(i)] = self.heapList[i]
             self.heapList[i] = tmp
          i = parent(i)

    def percDown(self,i):
        while (leftChild(i)) < len(self.heapList):
            mc = self.minChild(i)
            if self.compare(self.heapList[mc], self.heapList[i]):
                tmp = self.heapList[i]
                self.heapList[i] = self.heapList[mc]
                self.heapList[mc] = tmp
            i = mc

    def minChild(self,i):
        if rightChild(i) >= len(self.heapList):
            return leftChild(i)
        else:
            if self.compare(self.heapList[leftChild(i)], self.heapList[rightChild(i)]):
                return leftChild(i)
            else:
                return rightChild(i)

    def delMin(self):
        retval = self.heapList[1]
        self.heapList[1] = self.heapList[-1]
        self.heapList.pop()
        self.percDown(1)
        return retval
        
    def insert(self,k):
        self.heapList.append(k)
        self.percUp(len(self.heapList)-1)

    def empty(self):
        return len(self.heapList)==1
        
    def min(self):
        return self.heapList[1]


if __name__ == '__main__':
        
    h1 = BinHeap(init=[7,2,79,12,8,-9,-38,43])

    while not h1.empty():
        print h1.delMin()
        
        
    score = [7,2,79,12,8,-9,-38,43]
    things = range(8)
    
    
    h2 = BinHeap(init=things, comparator=lambda x,y : score[x] < score[y])

    while not h2.empty():
        x = h2.delMin()
        print x, score[x]




