import random as rd


class SIR:
    def __init__(self, graph):
        self.degree = graph.degree()
        self.average_degree = sum(self.degree)/len(self.degree)
        self.recover_probability = 100.0/self.average_degree
        # self.weighted_choices = [(False, int(100 - self.recover_probability)), (True, int(self.recover_probability))]
        # self.population = [val for val, cnt in self.weighted_choices for i in range(cnt)]

    def sir_algorithm(self, graph, infected, immune):
        infected_prime = set()

        for v in infected:
            # susc = graph[v]
            susc = graph.neighbors(v)
            if not susc: continue

            inf = rd.choice(susc)
            if not(inf in immune) and not(inf in infected):
                infected_prime.add(inf)
            if rd.randrange(0, 100) > self.recover_probability:
                infected_prime.add(v)
            else:
                immune.add(v)

        return infected_prime

    def sir_node(self, graph, node):
        infected, immune = set([node]), set()
        while bool(infected):
            infected = self.sir_algorithm(graph, infected, immune)

        return len(immune)

