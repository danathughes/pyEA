from . import *

i1 = InputGene((100,100,1))
c1 = Conv2DGene((10,10), (2,2), 4)
p1 = Pool2DGene((2,2), (2,2))
c2 = Conv2DGene((8,8), (1,1), 5)
p2 = Pool2DGene((3,3), (2,2))
f1 = FullyConnectedGene(6)
o1 = OutputGene(7)

i1.next_gene = c1
c1.next_gene = p1
p1.next_gene = c2
c2.next_gene = p2
p2.next_gene = f1
f1.next_gene = o1

c1.prev_gene = i1
p1.prev_gene = c1
c2.prev_gene = p1
p2.prev_gene = c2
f1.prev_gene = p2
o1.prev_gene = f1