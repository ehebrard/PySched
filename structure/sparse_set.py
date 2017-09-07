import array

# standard implementation of sparse set
class SparseSet:
    def __init__(self, capacity=8):
        self.size = 0
        self.element = array.array('i', range(capacity))
        self.index = array.array('i', range(capacity))

    def __len__(self):
        return self.size
        
    def capacity(self):
        return len(self.index)

    def newElement(self, x):
        if x>=self.capacity():
            self.element.extend(range(self.capacity(), x+1))
            self.index.extend(range(self.capacity(), x+1))

    def add(self, x):
        if x >= self.capacity():
            self.newElement(x)
        
        if self.index[x] >= self.size:
            self.move(x, self.size)
            self.size += 1
        
    def remove(self, x):
        if x >= self.capacity():
            self.newElement(x)
            
        if self.index[x] < self.size:
            self.size -= 1
            self.move(x, self.size)

    def move(self, x, r):
        ix = self.index[x]
        vl = self.element[r]
        self.element[r] = x
        self.element[ix] = vl
        self.index[x] = r
        self.index[vl] = ix
        
    def pop(self):
        self.size -= 1
        return self.element[self.size]
        
    def depop(self):
        self.size += 1
        return self.element[self.size-1]
        
    def setSize(self, s):
        self.size = s
        
    def clear(self):
        self.size = 0
        
    def fill(self):
        self.size = self.capacity()
        
    def __getitem__(self,i):
        if i>=0:
            return self.element[i]
        else:
            return self.element[self.size+i]
        
    def __setitem__(self,r,x):
        self.move(x,r)
        
    def __contains__(self,x):
        return self.index[x]<self.size
        
    def __iter__(self):
        for x in self.element[:self.size]:
            yield x
            
    def __str__(self):
        return '[%s]'%', '.join([str(x) for x in self])
        
        
if __name__ == '__main__':
    import random
    maxval = 50
    
    s = SparseSet()
    
    print s
    
    for i in range(100000):
        if random.randint(0,1)>0:
            s.add(random.randint(0,maxval))
        else:
            s.remove(random.randint(0,maxval))
            
        print s


        
