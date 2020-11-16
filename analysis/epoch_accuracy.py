import matplotlib.pyplot as plt


# line 1 points
y1 = [
    0.8279122114181519,
 0.8280423283576965,
0.8282157778739929,
0.8284760117530823,
0.8284326195716858,
0.8284326195716858,
0.8283892869949341,
0.9904588460922241,
0.9893312454223633,
0.9891577959060669
      ]
x1 = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
# plotting the line 1 points
plt.plot(x1, y1, label = "Epoch=1")

y2 = [
0.9891577959060669,
    0.9890276789665222,
    0.9903721213340759,
    0.9917598962783813,
 0.9917598962783813,
0.9919334053993225,
 0.9919767379760742,
 0.9917598962783813,
0.9915430545806885,
 0.9916731715202332,
 0.9919334053993225

      ]
x2 = [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
# plotting the line 1 points
plt.plot(x2, y2, label="Epoch=2")

plt.xlabel('Iterations (100/each)')
# Set the y axis label of the current axis.
plt.ylabel('Loss')
# Set a title of the current axes.
plt.title('Validate Accuracy Trends with two Epoches')
# show a legend on the plot
plt.legend(loc='lower left')


plt.savefig("epoch_accuracy.png")

# Display a figure.
plt.show()