from SIR import SIR
import networkx as nx
import itertools
import random

""" 
  Model parameters:
    epsilon -> Initial value of pheromone for every node
    q0 -> determines the relative importance of exploitation vs exploration
    k ->  size of the expected set of influencers
    m -> population
    alpha -> Importance of pheromones 
    betha -> Importancce of heuristic
    rho -> Factor of pheromones to be deposited
    steps -> Number of repetitions of the process
"""
alpha = 0.3
betha = 0.7
rho = 0.9
epsilon = 0.03
q0 = 0.8

k = 3
m = 100
steps = 50

heuristics = {"degree": nx.degree}

graph = None  # Graph to test influence
cdg = None    # Complete digraph generated based on the graph


def complete_graph_from_list(L):
  G = nx.Graph()
  if len(L)>1:
    edges = itertools.combinations(L,2)
    G.add_edges_from(edges)

  return G


def selectNextNode(route):
  import operator
  global cdg, alpha, betha, q0
  
  q = random.random()
  node_index = None
  
  current = route[-1]
  neighbors = [node for node in cdg.neighbors(current) if node not in route]
  attractiveness = \
    [(cdg.node[x]["ph"] ** alpha) * (cdg.node[x]["h"] ** betha) for x in neighbors]
    
  
  if q <= q0:
    node_index, value = max(enumerate(attractiveness), key=operator.itemgetter(1))
  else:
    random_value = random.random()
    total = sum(attractiveness)
    probabilities = [value/total for value in attractiveness]
    
    total_prob = 0
    for index in range(len(probabilities)):
      total_prob += probabilities[index]
      if total_prob >= random_value:
        node_index = index
        
#  print("attractiveness: ", dict(zip(neighbors, attractiveness)))
#  a = "exploiting "  if q <= q0 else "exploring"
#  print("selected {} {}".format(neighbors[node_index], a))
  return neighbors[node_index]
        

def generateSolutions():
  global cdg, k, q0, m
  solutions = list()
  route = [random.choice(cdg.nodes())]
  
  for ant in range(m):
    while len(route) < k:
      route.append(selectNextNode(route))
    solutions.append(route)
    
  return solutions
  
    
def evaluateSolutions(solutions):
  global graph
  scores = [0.0 for i in range(len(solutions))]
  evaluations = 100
  model = SIR(graph)
  for i in range(len(solutions)):
    
    # Run the model 100 times for each solution and calculate the average
    total = 0
    for j in range(evaluations):
      influenced = 0

      for node in solutions[i]:
        influenced += model.sir_node(graph, node)
      
      total += influenced
    scores[i] = total/evaluations
  return scores


def updatePheromoneValue(nodes, influence):
  global cdg
  
  for node in nodes:
    cdg.node[node]["ph"] = (1-rho) * cdg.node[node]["ph"]  + influence


def ACO_influence_maximization(cdg, heuristic):
  """ Calculates a set of nodes which maximize the influence using the ACO 
      algorithm """ 
  global epsilon
  
  results = heuristic(cdg)
  for node, attr in cdg.nodes(data=True):
    attr["ph"] = epsilon  # pheromone 
    attr["h"] = results[node] # heuristic
    
  for step in range(steps):
    solutions = generateSolutions()
    scores = evaluateSolutions(solutions)
    
    # Update the pheromone values of the best results
    max_index, influence = max(enumerate(scores))
    updatePheromoneValue(solutions[max_index], influence)
    
    print("Maximum influence on step {} was: {} given with the seeds: {}".format(step, influence, solutions[max_index]))
    

def main():
  global cdg, graph, heurisitics
  
  graph = nx.read_gml("tests/test1.gml");
  cdg = complete_graph_from_list(graph.nodes())
  ACO_influence_maximization(cdg, heuristics["degree"])
  
  
if __name__ == "__main__":
  main()