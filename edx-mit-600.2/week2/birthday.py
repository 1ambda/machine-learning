import operator


def atLeastTwoSameBirthday(n):
    xs = range(365 - n, 365)
    ys = map((lambda x: float(x) / 365), xs)
    return 1 - reduce(operator.mul, ys)


def minNumOfPeople99Percent():
    for n in range(1, 365):
        if atLeastTwoSameBirthday(n) >= 0.99:
            return n

    return n


print atLeastTwoSameBirthday(29)
print atLeastTwoSameBirthday(249)
print minNumOfPeople99Percent()
