import numpy as np
from wordcloud import get_single_color_func
import streamlit as st

def portrait_robot(data) :
    sexe_majo = data['sexe'].mode()[0]

    if sexe_majo == "Fille" :
        prenom_majo = data.loc[data['sexe'] == sexe_majo]['prenom_fem'].mode()[0]
    else :
        prenom_majo = data.loc[data['sexe'] == sexe_majo]['prenom_masc'].mode()[0]

    date_de_naissance = data.date.median()
    taille = data.taille.median()
    poids = data.poids.median()
    chevelure = data['longueur_cheveux'].mode()[0]
    couleur = data['couleur_cheveux'].mode()[0]

    return sexe_majo, date_de_naissance, taille, poids, chevelure, couleur, prenom_majo

def calcul_angles(angles, proportion, acc):
    current_angle = proportion * np.pi
    acc.append(current_angle)
    if len(angles) == 0:
        angles.append(current_angle/2)
    else :
        angles.append(angles[-1] + acc[-2]/2 + current_angle/2)

    return angles, acc

def make_hair_bars(df, longueur_cheveux, ordre_cheveux, couleurs_cheveux, ordre_couleur) :

    width = []
    radii = []
    angles = []
    acc = []
    theta = []
    colors = []
    nb_labels = []

    for longueur in ordre_cheveux :
        nb = len(df.loc[(df['longueur_cheveux'] == longueur)])
        angles, acc = calcul_angles(angles, nb/len(df), acc)
        
        for couleur in ordre_couleur:
            nb = len(df.loc[(df['longueur_cheveux'] == longueur) & (df['couleur_cheveux'] == couleur)])
            if nb > 0 :
                nb_labels.append("{} - {}".format(couleur, nb))
                radii.append(longueur_cheveux[ordre_cheveux.index(longueur)])
                colors.append(couleurs_cheveux[ordre_couleur.index(couleur)])
                theta, width = calcul_angles(theta, nb/len(df), width)
    
    # modif label pour cause de texte trop long !
    new_labels = []
    for texte in nb_labels :
        new_label = texte.replace('Pas de cheveux... Pas de cheveux !', 'Chauve !')
        new_labels.append(new_label)

    return width, radii, angles, theta, colors, new_labels

class SimpleGroupedColorFunc(object):
    """Create a color function object which assigns EXACT colors
       to certain words based on the color to words mapping
       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.
       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.word_to_color = {word: color
                              for (color, words) in color_to_words.items()
                              for word in words}

        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)


class GroupedColorFunc(object):
    """Create a color function object which assigns DIFFERENT SHADES of
       specified colors to certain words based on the color to words mapping.
       Uses wordcloud.get_single_color_func
       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.
       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.color_func_to_words = [
            (get_single_color_func(color), set(words))
            for (color, words) in color_to_words.items()]

        self.default_color_func = get_single_color_func(default_color)

    def get_color_func(self, word):
        """Returns a single_color_func associated with the word"""
        try:
            color_func = next(
                color_func for (color_func, words) in self.color_func_to_words
                if word in words)
        except StopIteration:
            color_func = self.default_color_func

        return color_func

    def __call__(self, word, **kwargs):
        return self.get_color_func(word)(word, **kwargs)