from matplotlib import pyplot as plt


def f(x):
    if (x < -3):
        return 0
    elif (x < 0):
        return 3
    elif (x <= 4):
        return x
    else:
        return 0


x_data = []
y_data = []

for i in range(-6, 10):
    x_data.append(i)
    y_data.append(f(i))


fig, pt = plt.subplots()
pt.set_in_layout(False)
pt.stem(x_data, y_data, markerfmt='ko', basefmt='k', linefmt='k')
pt.spines['bottom'].set_position('zero')
pt.spines['left'].set_position('zero')
pt.spines['right'].set_color('none')
pt.spines['top'].set_color('none')
plt.ylim(-2, 8)
plt.xlim(-8, 11)
plt.xticks([-3, -2, -1, 0, 1, 2, 3, 4])
plt.yticks([4])
pt.set_xlabel('n', loc='right')
pt.set_ylabel('X(n)', loc='top', rotation=0)
plt.show()