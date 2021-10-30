from solver_interface import Instance, ExplainableOutput, DecisionTree
import kmedplusplus

def build_tree(refset) -> DecisionTree:

    tree = 0
    return tree

class IMM:
    def __call__(self,instance: Instance) -> ExplainableOutput:
        # get a reference set of k centers, we now use kmedplusplus for this
        solver = kmedplusplus.KMedPlusPlus(numiter=5)
        refset = solver(instance)
        output = build_tree(refset)
        return output