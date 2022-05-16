import numpy as np
import matplotlib.pyplot as plt
from sympy import Lambda, Symbol
import inspect
from collections import defaultdict

class Node():

    left = None
    right = None
    criterion = None
    left_count = 0
    right_count = 0

    def __init__(self, left=None, right=None, criterion=None, output=None):
        self.left = left
        self.right = right
        self.criterion = criterion
        self.output = output

    # Depth-first search returning a dictionary of each layer of the tree.
    def dfs(self, tree, depth):
        # Pre-order dfs
        depth += 1
        # If a node is a leaf, criterion is None
        if self.criterion is None:
            tree[depth].append([(self.left_count, self.right_count), self.output])
        else:
            tree[depth].append([(self.left_count, self.right_count), self.repr_lambda(self.criterion)])
            self.left.dfs(tree, depth)
            self.right.dfs(tree, depth) 
        
        return tree

    # Used for printing the tree structure to the console.
    def display(self):
        levels_dict = self.dfs(defaultdict(lambda: [], {}), 0)


        for level in levels_dict:
            print("Level {}".format(level))
            for node in levels_dict[level]:
                classCount = node[0]
                p_left = classCount[0] / (classCount[0] + classCount[1])
                p_right = classCount[1] / (classCount[0] + classCount[1])

                def divWeird(x, y):

                    try:
                        return x / y
                    except:
                        return 0

                def side(l, r):
                    
                    a = -divWeird(l , (l + r))
                    
                    if a == 0.0:
                        return 0

                    b = np.log2(-a)

                    return a * b

                entropy = side(p_left, p_right) + side(p_right, p_left)



                print(" {}, entropy = {}, p_left: {}, p_right: {}, {} |".format(node[0], round(entropy, 2), round(p_left, 2), round(p_right, 2), node[1]), end='')
            print()

        print()


    def repr_lambda(self, x):
        string_repr = "".join(inspect.getsourcelines(x)[0]).strip()
        return string_repr[string_repr.index(":")+2:]

    def search(self, sample):
        print("\nSearching for {}".format(sample))
        result = self.traverse(sample)
        print("Result: {}\n".format(result))
        return result

    def traverse(self, sample):
        if sample[1] is not None:
            if sample[1] == 1:
                self.right_count += 1
            else:
                self.left_count += 1
        if self.criterion is None:
            print("Found leaf")
            return self.output
        elif self.criterion(sample[0]):
            print("Right")
            return self.right.traverse(sample)
        else:
            print("Left")
            return self.left.traverse(sample)



if __name__ == "__main__":


    # Create a decision tree
    # If true go right else left
    root = Node(
        left=Node(output=-1),
        right=Node(output=1),
        criterion=lambda x: x[0] > 144.5
    )

    # root2 = Node(
    #     left=Node(
    #         output=-1
    #         ),
    #     right=Node(
    #         left=Node(output=-1),
    #         right=Node(output=1),
    #         criterion= lambda x: x[1] > 0.5
    #         ),
    #     criterion=lambda x: x[1] > -0.5
    # )


    # Enter data from the question
    X = [[168, 132, 59],
    [206, 5, 62],
    [138, 123, 267],
    [88, 12, 102],
    [1, 114, 29],
    [154, 42, 101],
    [46, 146, 167],
    [52, 208, 134],
    [40, 22, 250]]
    y = [1,1,-1,-1,1,1,1,1,-1]

    zipped = list(zip(X,y))

    for x in zipped:
        root.search(x)
        # root2.search(x)

    root.display()
    # root2.display()

    # Search with a new sample
    sample = [[0,0]]
    zipped = list(zip(sample, [None]*len(sample)))
    
    # for x in zipped:
    #     root.search(x)
    #     root2.search(x)
    
    



    