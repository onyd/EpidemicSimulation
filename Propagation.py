import numpy.random as rd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from Clustering import cluster, NPoint


def trunc(x, y, box):
    if box[0] > x:
        x = box[0]
    elif x > box[1]:
        x = box[1]

    if box[2] > y:
        y = box[2]
    elif y > box[3]:
        y = box[3]

    return x, y


class Disease:
    def __init__(self, letality, toi, s_toi) -> None:
        self.letality = letality
        self.toi = toi  # average time of infection
        self.s_toi = s_toi  # deviation of toi


class Person:
    def __init__(self, pos, infected=False, toi=float("+inf")) -> None:
        # State
        self.infected = infected
        self.toi = toi
        self.alive = True

        # activities orderes by reverse time progression as [{start, activity}, ...]
        self.activities = []

        self.immunity = 0.0

        self.pos = pos
        self.home_pos = pos

    def set_infected(self, disease):
        self.infected = True
        self.toi = rd.normal(disease.toi, disease.s_toi)

    def update(self, disease):
        if not self.infected or not self.alive:
            return 0

        if self.toi < 0:
            self.infected = False
            self.toi = float("+inf")
            self.immunity = 0.95

            if rd.random() < disease.letality:
                self.alive = False
                return 1
            return 2
        else:
            self.toi -= 1
            return 0

    def goto(self, pos):
        self.pos = pos + NPoint(1-2*rd.random(), 1-2*rd.random())

    def go_home(self):
        self.goto(self.home_pos)
        self.activity = None

    def move(self, city_size):
        if self.alive:
            nx = self.pos.x+(rd.random()-0.5)*0.05
            ny = self.pos.y+(rd.random()-0.5)*0.05

            self.pos.x, self.pos.y = trunc(
                nx, ny, [0, city_size[0], 0, city_size[1]])


class Attendance:
    """This class encapsulates the attendance data over Activities for a whole city and allow to simulate it"""

    def __init__(self, interval_maps, ratio) -> None:

        self.ratio = ratio  # real_time in hour = inner_time * ratio
        self.width = self.ratio * 24  # Inner time range

        self.data = [[0 for _ in range(self.width)]
                     for _ in range(len(interval_maps))]
        self.sum = [0 for _ in range(self.width)]
        self.process_attendance(interval_maps)

    def process_attendance(self, interval_maps):
        """Convert a list of attendance dict of the form {(a, b) : density} to a pseudo discrete density in the inner_time_domain"""
        for k, interval_map in enumerate(interval_maps):
            for interval, density in interval_map.items():
                i1 = int(self.ratio * interval[0])
                i2 = int(self.ratio * interval[1])
                for i in range(i1, i2):
                    self.data[k][i] = density
                    self.sum[i] += density

    def simulate(self, mask):
        """mask is a list of 0 or 1 of size self.width where 1 means keep and 0 except the density

        Simulate a sample of the variable which law has the sum density and precise "what part of data has been hit" in inner_time domain"""
        i = rd.randint(0, self.width)
        d = rd.random() * self.sum[i]

        while mask[i] == 0:
            i = rd.randint(0, self.width)
            d = rd.random() * self.sum[i]

        return i, self.locate(i, d)

    def locate(self, i, d):
        """Return the index of the corresponding density for simulation (i, d) on the sum"""
        current = 0
        for k, density in enumerate(self.data):
            current += density[i]
            if d < current:
                return k

        return None  # Impossible activity at time i we return to update the mask


class Activity:
    def __init__(self, name, duration, shops, attendance) -> None:
        self.name = name
        self.shops = shops  # Shops that offer that activity
        self.attendance = attendance  # dict attendance
        self.duration = duration

    def repart(self, p):
        """Simulate where p goes for the activity, the closer the shops, the higher the probability"""
        distance = [p.pos.squared_distance_to(shop.pos) for shop in self.shops]
        L = sum(distance)
        F = [0]
        for d in distance:
            F.append(F[-1] + (1 / (len(self.shops) - 1)) * (1 - d / L))

        U = rd.random()
        i = 1  # skip first 0
        while U > F[i]:
            i += 1

        return self.shops[i-1]  # inverse shift


class Shop:
    def __init__(self, pos) -> None:
        self.pos = pos


class City:
    def __init__(self, city_size, disease, n, activities, barrier_factor, ratio, n_days) -> None:
        """Build a square city of city_size in which the n people are threatened by disease, they can do activities but apply barrier gestures which reduce by barrier_factor the contamination.
        n_days is the number of days in the simulation
        dt is the time between each step in normalized hour (ie 8,5 means 8:30"""
        self.city_size = city_size
        self.disease = disease
        self.population = [Person(NPoint(*trunc(rd.normal(5, 2), rd.normal(
            5, 2), [0, city_size[0], 0, city_size[1]]))) for _ in range(n)]
        self.activities = activities
        self.barrier_factor = barrier_factor
        self.n_days = n_days

        self.days = [0]
        self.deads = [0]
        self.infected = [0]

        self.attendance = Attendance(interval_maps=[
                                     activity.attendance for activity in self.activities],
                                     ratio=ratio)

    def run(self):
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)
                   ) = plt.subplots(nrows=2, ncols=2)
        first = rd.choice(self.population)
        first.set_infected(self.disease)

        self.ani = FuncAnimation(self.fig, self.update, interval=50,
                                 init_func=self.init, blit=True)
        plt.show()

    def init(self):
        # Plot
        shops = []
        for shops in map(lambda x: x.shops, self.activities):
            shops.extend(shops)

        self.people_scat = self.ax1.scatter([p.pos.x for p in self.population],                                [p.pos.y for p in self.population],
                                            cmap='rainbow',
                                            s=10)
        self.shop_scat = self.ax1.scatter([shop.pos.x for shop in shops], [
            shop.pos.y for shop in shops],
            c='k', s=30, marker='s')
        self.ax1.set_title("Simulation")
        self.ax1.set_xlim(0, self.city_size[0])
        self.ax1.set_ylim(0, self.city_size[1])

        self.cases, = self.ax2.plot(self.days, self.infected)
        self.ax2.set_title("Cases")
        self.ax2.set_xlim(0, self.n_days)
        self.ax2.set_ylim(0, len(self.population))

        self.death, = self.ax3.plot(self.days, self.deads)
        self.ax3.set_title("Death")
        self.ax3.set_xlim(0, self.n_days)
        self.ax3.set_ylim(0, 10*self.disease.letality*len(self.population))

        self.line = [self.people_scat, self.shop_scat, self.cases, self.death]
        return self.line

    def fill_activities(self, p):
        """Fill the activities of Person p for one day"""
        mask = [1 for _ in range(self.attendance.width)]
        activities = []
        while mask != [0] * self.attendance.width:
            src_activity_time, i_activity = self.attendance.simulate(mask)

            # No activity at this time
            if i_activity is None:
                mask[src_activity_time] = 0
                continue

            # Feel according to activity duration
            for t in range(src_activity_time, src_activity_time + int(self.activities[i_activity].duration * self.attendance.ratio)):
                mask[t] = 0
            activities.append({'start': src_activity_time,
                               'activity': self.activities[i_activity]})
        activities.sort(key=lambda p: p['start'], reverse=True)
        p.activities = activities

    def fill_shops(self, p, t):
        """Fill the shops for activity at t in inner_time domain"""
        # Pick activity
        if p.activities and p.activities[-1]['start'] == t:
            activity = p.activities.pop()['activity']
            shop = activity.repart(p)
            p.goto(shop.pos)

    def propagation(self, d=0.05):
        points = [p.pos for p in self.population]
        clusters = cluster(d, points)
        colors_val = np.linspace(0, 1, len(clusters))

        data = []
        colors_array = []
        for k, ids in enumerate(clusters):
            colors_array.extend([colors_val[k] for _ in range(len(ids))])

            # Propagation in the cluster
            for i in ids:
                data.append((points[i].x, points[i].y))
                if self.population[i].infected and self.population[i].alive:
                    for i_neighboor in points[i].neighboors:
                        if not self.population[i_neighboor].infected and self.population[i_neighboor].alive and \
                                rd.random() < (1 - points[i].squared_distance_to(points[i_neighboor]) / d**2) * (1 - self.population[i_neighboor].immunity) * self.barrier_factor:
                            self.infected[-1] += 1
                            self.population[i_neighboor].set_infected(
                                self.disease)

        data = np.array(data)
        colors_array = np.array(colors_array)

        return data, colors_array

    def update(self, frame):
        if frame % self.attendance.width == 0:
            # --Statistics update--
            for p in self.population:
                update_code = p.update(self.disease)
                if update_code == 1:
                    self.deads[-1] += 1
                    self.infected[-1] -= 1
                elif update_code == 2:
                    self.infected[-1] -= 1

                self.fill_activities(p)
                self.fill_shops(p, frame % self.attendance.width)

            self.cases.set_data(self.days, self.infected)
            self.death.set_data(self.days, self.deads)

            self.infected.append(self.infected[-1])
            self.deads.append(self.deads[-1])
            self.days.append(len(self.days))
            print("days :", self.days[-1])
        else:
            for p in self.population:
                self.fill_shops(p, frame % self.attendance.width)

        data, colors_array = self.propagation()

        self.people_scat.set_offsets(data)
        self.people_scat.set_array(colors_array)

        return self.line

    def plot(self):
        T = np.arange(0, self.n_days, 1)
        plt.plot(T, self.infected, label="infected")
        plt.plot(T, self.deads, label="deads")
        plt.legend()
        plt.xlabel("time")
        plt.ylabel("Case evolution")
        plt.show()


def main():
    disease = Disease(letality=1/100, toi=7, s_toi=2)

    activities = [Activity("Restaurant", 0.5, [Shop(NPoint(1, 1)),
                                               Shop(NPoint(9, 9))],
                           {(10, 11): 1, (11, 12): 3, (12, 13): 3, (13, 14): 1})]
    city = City(city_size=[10, 10],
                disease=disease,
                n=100,
                activities=activities,
                barrier_factor=0.8,
                ratio=2,
                n_days=30,
                )
    city.run()


if __name__ == "__main__":
    main()
