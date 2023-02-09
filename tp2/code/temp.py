repartition = {
    2: 0.,
    3: 1.,
    4: 0.,
    5: 0.,
    "pass": 0.,
    "fail": 0.
}

n = 10000
for i in range(n):
    repartition = {
        2: 1/3 * repartition[3],
        3: 3/4 * repartition[2] + 3/8 * repartition[4],
        4: 2/3 * repartition[3] + 2/5 * repartition[5],
        5: 5/8 * repartition[4],
        "pass": repartition["pass"] + 3/5 * repartition[5],
        "fail": repartition["fail"] + 1/4 * repartition[2]
    }

print(repartition)