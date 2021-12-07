import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg

pred = pd.read_csv('pred.csv')
y_test = pd.read_csv('y_test.csv')

x = range(0, 542)
y = np.array(y_test['0'][2000:2542])
y2 = np.array(pred['0'][2000:2542])

fig, ax = plt.subplots(1, 2)
line, = ax[0].plot(x, y, color='k')
line2, = ax[0].plot(x, y2, color='r')
img = ax[1].imshow(mpimg.imread('sample_2/image_n=2000_t=0.png'))

def update(num, x, y, y2, line, line2, img):
	line.set_data(x[:num], y[:num])
	line2.set_data(x[:num], y2[:num])
	img.set_data(mpimg.imread('sample_2/image_n=' + str(2000 + num) + '_t=0.png'))
    # line.set_data(x[:num], y[:num])
    # line2.set_data(x[:num], y2[:num])
    # line.axes.axis([0, 10, 0, 1])
	return [line, line2, img]

ani = animation.FuncAnimation(fig, update, len(x), fargs=[x, y, y2, line, line2, img],
                              interval=40, blit=True)
ani.save('demo.gif')
plt.show()