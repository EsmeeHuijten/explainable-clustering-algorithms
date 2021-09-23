import algorithms.randomized
import data.artificial

instance = data.artificial.toy_input(5, 2)
solver = algorithms.randomized.RandomCenters()
output = solver.solve(instance)
print(output)
