# ICLR_Challenge

![MNIST](https://cdn-images-1.medium.com/max/1600/1*yBdJCRwIJGoM7pwU-LNW6Q.png)

J'étudie l'évolution en temps polynomial d'un réseau neuronal convolutif (CNN) à reconnaître les images des données MNIST.

Mes programmes sont basés sur la recherche de la publication lors de la conférence ICLR, qui est disponible à https://openreview.net/forum?id=SkA-IE06W.

## Contexte
Le but de cette publication par Du, Lee et Tian et de savoir si l'apprentissage d'un CNN se fait en temps polynomial. Pour vérifier cette idée, ils ont observé l'apprentissage d'un CNN avec des filtres initialisés différemment.

## Mes observations reproduites
![Mes obervations](https://github.com/TheFloatingString/ICLR_Challenge/blob/master/R%C3%A9sultats/Comparaison.png)
J'ai réussi à reproduire pour 2 des 3 CNN une courbe avec une forme semblable aux observations publiées. Cela soutient leur idée qu'un CNN apprend en temps polynomial.

## Mes programmes (inclus dans ce dossier GitHub)
**ICLR Demo.ipynb:** une démonstration visuelle de mes reproductions
**iclr_conv.py:**    une classe avec un CNN **Keras** que j'ai conçu pour reproduire les travaux de Du, Lee et Tian.
