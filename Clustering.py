from geo.Structure.DisjointUnionSet import DUS
from scipy.spatial import Delaunay


class NPoint:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

        self.neighboors = set()

    def copy(self):
        return NPoint(self.x, self.y)

    def squared_distance_to(self, other):
        return (self.x-other.x)**2+(self.y-other.y)**2

    def __add__(self, other):
        return NPoint(self.x+other.x, self.y+other.y)

    def __sub__(self, other):
        return NPoint(self.x-other.x, self.y-other.y)

    def __truediv__(self, c):
        return NPoint(self.x/c, self.y/c)

    def pseudo_angle(self, other):
        """Calcul du pseudo-angle entre self et other, evite des calcul trigonométrique inutile lorsque l'utilisation se restreint à des relation d'ordre"""
        dp = self - other
        p = dp.x / (abs(dp.x) + abs(dp.y))

        if dp.y > 0:
            return (3 - p) / 4
        else:
            return (1 + p) / 4


def cluster(distance, points):
    EMSF = DUS(len(points))

    # Build EMSF
    delaunay_triangulation = Delaunay(
        [(p.x, p.y) for p in points]).simplices.flatten()
    d2 = distance**2

    for i in range(0, len(delaunay_triangulation), 3):
        tris = [(delaunay_triangulation[i], delaunay_triangulation[i+1]), (delaunay_triangulation[i], delaunay_triangulation[i+2]), (delaunay_triangulation[i+1], delaunay_triangulation[i+2])
                ]

        for i1, i2 in tris:
            if points[i1].squared_distance_to(points[i2]) <= d2:
                EMSF.unite(i1, i2)
                points[i1].neighboors.add(i2)
                points[i2].neighboors.add(i1)

    result = list(EMSF.get_sets().values())

    return result
