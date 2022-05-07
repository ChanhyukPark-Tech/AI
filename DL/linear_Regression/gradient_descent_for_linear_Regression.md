Randomly Initialize w,b
lr = 0.1

For e = 1 to n(epoch):
d_w1 = 0; d_w2 = 0; d_b = 0
For i = 1 to m:

        a = w1x1 + w2x2 + b

        d_w1 += 2(a-y)x1
        d_w2 += 2(a-y)x2
        d_b += 2(a-y)

w1 -= lr x d_w1/m
w2 -= lr x d_w2/m
b -= lr x d_b/m
