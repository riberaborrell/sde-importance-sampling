import gradient_descent

soc = gradient_descent.gradient_descent(
    lr=0.004,
    epochs=10,
    m=20,
    M=1,
)
soc.gradient_descent()
a = soc._a
losses = soc._losses

print(a[-1])
print(losses[-1])
