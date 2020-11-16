import matplotlib.pyplot as plt


# line 1 points
y1 = [
    0.48211321234703064,
    0.43390369415283203,
    0.45974549651145935,
    0.4574151933193207,
    0.4703247547149658,
    0.39904361963272095,
    0.4873100519180298,
    0.1958690881729126,
    0.05901166796684265,
0.05429263412952423
      ]
x1 = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
# plotting the line 1 points
plt.plot(x1, y1, label = "Epoch=1")

y2 = [
0.05429263412952423,
    0.03732353076338768,
    0.05761750787496567,
    0.03591344505548477,
    0.022256171330809593,
    0.015134639106690884,
    0.016366882249712944,
    0.020947284996509552,
    0.01831391081213951,
    0.020556733012199402,
    0.014590829610824585

      ]
x2 = [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
# plotting the line 1 points
plt.plot(x2, y2, label="Epoch=2")

plt.xlabel('Iterations (100/each)')
# Set the y axis label of the current axis.
plt.ylabel('Loss')
# Set a title of the current axes.
plt.title('Loss Converge Trends with two Epoches')
# show a legend on the plot
plt.legend(loc='lower left')


plt.savefig("epoch.png")

# Display a figure.
plt.show()