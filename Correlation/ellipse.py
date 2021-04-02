import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse as Elp
from correlation import Correlation


class Ellipse:
    def __init__(self, x, y, size, rho):
        self.x = x
        self.y = y
        self.size = size
        self.rho = rho

    def plot(self):
        correlation = Correlation()

        pearson = correlation.pearson_correlation(self.x, self.y)
        mean_x = Correlation.mean(self.x)
        mean_y = Correlation.mean(self.y)
        variance_x = np.sqrt(Correlation.variance(self.x))
        variance_y = np.sqrt(Correlation.variance(self.y))
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        alpha = 1 / 2 * np.arctan((2 * pearson * variance_x * variance_y) /
                                  (variance_x * variance_x - variance_y * variance_y))

        alpha = alpha if alpha > 0 else alpha + np.pi / 2

        scale_x = 3 * variance_x
        scale_y = 3 * variance_y
        ellipse = Elp((mean_x, mean_y), 2 * ell_radius_x * scale_x,
                      2 * ell_radius_y * scale_y, np.degrees(alpha),
                      facecolor='none', edgecolor='red')
        fig, ax = plt.subplots()

        ax.add_artist(ellipse)
        plt.plot(self.x, self.y, 'o')
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Size = ' + str(self.size) + ', rho = ' + str(self.rho))
        plt.savefig('images/' + 'Ellipse' + str(self.size) + 'r' + str(self.rho) + '.png')
        plt.show()
        return
