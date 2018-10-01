import math


def ComputeEuclideanDistance(x1, y1, x2, y2):
    d = math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))
    return d

d_ag = ComputeEuclideanDistance(3, 104, 18, 90)
print(d_ag)
d_ag = ComputeEuclideanDistance(2, 100, 18, 90)
print(d_ag)
d_ag = ComputeEuclideanDistance(1, 81, 18, 90)
print(d_ag)
d_ag = ComputeEuclideanDistance(101, 10, 18, 90)
print(d_ag)
d_ag = ComputeEuclideanDistance(99, 5, 18, 90)
print(d_ag)
d_ag = ComputeEuclideanDistance(98, 2, 18, 90)
print(d_ag)
d_ag =ComputeEuclideanDistance(0, 0, 18, 90)
print(d_ag)