# oligo

K: Everyone knows how to solve PDEs in n-dimensional Euclidean space. Boring... I know. But did you know you can also solve them on discrete structures? 

You: Wait, you mean like on the edges on a graph?

This package enwraps an original computationally efficient algorithm that converts an arbitrary graph structure with lengths assigned to its edges to a second deterivative matrix, encoding the connectivity of numerous graph comparments. This allows you to take a single PDE (or potentially multiple coupled PDEs) and convert them to many coupled ODEs that can be solved numerically.

So far, I've uploaded some code that shows diffusion on a graph, although I'm building this for my mathematical physiology project where I model calcium flow in oligodendrocytes (big branchy type cells important for learning). I'll add my code here when I'm finished my project to show a more complex use for the algorithm and make the framework more generalizable to different kinds of problems.

Potential appplications in neural modelling, network flow, fluid transport, traffic flow and more!

Example of diffusion on a graph:

![graph-diff-1](https://github.com/kushasareen/oligo/blob/main/graph-diff1.PNG)

A depiction of how it works:
![how-it-works1](https://github.com/kushasareen/oligo/blob/main/how-it-works-1.PNG)
![how-it-works2](https://github.com/kushasareen/oligo/blob/main/how-it-works-2.PNG)
![how-it-works3](https://github.com/kushasareen/oligo/blob/main/how-it-works-3.PNG)
