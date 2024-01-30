from cProfile import label
import os
import numpy as np
import matplotlib.pyplot as plt
import torch


def p5():
    X = 0.1 * np.random.randn(100, 4, 2) + np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    for i in range(4):
        plt.scatter(X[:, i, 0], X[:, i, 1], c='rgby'[i])
    plt.savefig("output/5.png")


def p6():
    x = torch.linspace(0, 2*np.pi, 30)
    plt.figure()
    plt.plot(x, torch.sin(x))
    plt.plot(x, torch.cos(x))
    plt.savefig("output/6.png")


def p7():
    x = torch.linspace(0, 2*np.pi, 10)
    plt.figure(figsize=(12, 12))  # définition d'une figure en précisant sa taille
    # Courbes
    plt.plot(x, torch.sin(x), 'rs-', label='Sinus')
    plt.plot(x, torch.cos(x), 'b*', label='Cosinus')
    # Titre et légende
    plt.title('Comparaison Sinus/Cosinus entre 0 et $2\pi$')
    plt.legend()
    # Texte des axes
    plt.xlabel('Abscisse')
    plt.ylabel('Ordonnée')
    # Définition des limites des axes
    plt.xlim([-1, 7])
    plt.ylim([-2, 2])
    # plt.show()  # Affichage de la figure créée.
    plt.savefig("output/7.png")


def p8():
    x = np.linspace(0, 2*np.pi, 100)
    np.random.shuffle(x)
    plt.figure()
    plt.plot(x, np.sin(x), '.-')
    plt.plot(x, np.cos(x), '+')
    plt.savefig("output/8.png")


def p9():
    x = np.linspace(-50, 50, 250)
    plt.figure()
    plt.plot(x, np.sin(x)/x, '-', label='sin(x)/x')
    plt.plot(x, 1/x, '--', label='1/x')
    plt.xlabel('Abscisse')
    plt.ylabel('Ordonnée')
    plt.xlim([-55, 55])
    plt.ylim([-1, 1])
    plt.savefig("output/9.png")


def p10():
    t = np.linspace(0, 2*np.pi, 10)
    plt.figure()
    plt.scatter(np.cos(t), np.sin(t), c=t, s=100*t)
    plt.axis('equal')  # pour avoir des axes avec les mêmes proportions.
    plt.colorbar()  # Échelle des valeurs associées aux couleurs
    # cbar = plt.colorbar(ticks=[0, np.pi, 2*np.pi])
    # cbar.ax.set_yticklabels(["Low", "Medium", "High"])
    plt.savefig("output/10.png")


def p11():
    n = int(1e3)
    x = np.linspace(0, 1, n)
    degradee = np.tile(x[:, np.newaxis], n)
    im = np.stack((degradee, degradee.T, 1-degradee), axis=2)
    plt.figure()
    plt.imshow(im)
    plt.savefig("output/11.png")


if __name__ == "__main__":

    if not os.path.exists("output"):
        os.makedirs("output")

    p5()
    p6()
    p7()
    p8()
    p9()
    p10()
    p11()

    print("Done")
