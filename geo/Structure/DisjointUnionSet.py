from collections import defaultdict


class DUS:
    """Implémentation de la structure de donnée abstraite DisjointUnionSet"""

    def __init__(self, size):
        self.parents = [i for i in range(size)]
        self.ranks = [0 for _ in range(size)]

    def find(self, x):
        """Trouve l'id du set (id de la racine)"""
        while self.parents[x] != x:
            x, self.parents[x] = self.parents[x], self.parents[self.parents[x]]

        return x

    def is_in_same_set(self, x, y):
        """Renvoie True si les id x et y sont dans le même set"""
        return self.find(x) == self.find(y)

    def unite(self, x, y):
        """Fusionne le set de x et de y"""
        set1 = self.find(x)
        set2 = self.find(y)

        if set1 == set2:
            # Rien à faire
            return False

        if self.ranks[set1] > self.ranks[set2]:
            set1, set2 = set2, set1

        if self.ranks[set1] == self.ranks[set2]:
            self.ranks[set2] += 1

        self.parents[set1] = set2
        return True

    def get_sets(self):
        """Renvoie un dictionnaire set_id -> {ids}"""
        sets = defaultdict(set)
        for i in range(len(self.parents)):
            sets[self.find(i)].add(i)

        return sets


if __name__ == "__main__":
    dus = DUS(10)
    dus.unite(0, 1)
    dus.unite(2, 3)
    dus.unite(1, 2)
    dus.unite(4, 5)

    print(dus.parents)
    print(dus.ranks)

    print(dus.get_sets())
