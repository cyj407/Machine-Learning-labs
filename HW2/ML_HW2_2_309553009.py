from math import factorial

def C(m, n):
    return float(factorial(m) / (factorial(m-n) * factorial(n)))

## input file
file_n = str((input("file path: ") or 'testfile.txt'))
cases = []
with open(file_n) as f:
    for l in f:
        c = l.strip()
        cases.append(c)
f.close()

## input beta prior
prior_a = int((input("a = ") or 0))
prior_b = int((input("b = ") or 0))

## init
posterior_a = prior_a
posterior_b = prior_b

for n, data in enumerate(cases):

    prior_a = posterior_a
    prior_b = posterior_b

    ## count the result == 1
    total_1 = 0
    for i in data:
        if(i == '1'):
            total_1 += 1
    total_0 = len(data) - total_1

    p = float(total_1) / len(data)
    q = 1 - p

    ## binomial : C(total, total_1) * p^(total_1) * q^(total_0)
    bino_li = C(len(data), total_1) * (p ** total_1) * (q ** total_0)    

    print("case {}: {}".format( n+1, data))
    print("Likelihood: {}".format(bino_li))
    print("Beta prior:     a = {:3d}  b = {:3d}".format(prior_a, prior_b))

    posterior_a = prior_a + total_1
    posterior_b = prior_b + total_0

    print("Beta posterior: a = {:3d}  b = {:3d}\n".format(posterior_a, posterior_b))