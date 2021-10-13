import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import altair as alt

from babel.dates import format_datetime
from textwrap import wrap

from wordcloud import WordCloud, get_single_color_func
from PIL import Image

path_to_df = "processed_data.pkl"
df = pd.read_pickle(path_to_df)
df["date"] = pd.to_datetime(df["date"], dayfirst=True)

source = df
viz_prono = "Pronostics"
robot_baby = "Faire-part de naissance du bébé"
sexes = ['Fille', 'Garçon']

ordre_cheveux = ['Maxi chevelure !', 'Chevelure classique', 'Juste assez pour ne pas être chauve...', 'Pas de cheveux !']
ordre_couleur = ['Noirs', 'Bruns', 'Roux', 'Blonds', 'Pas de cheveux... Pas de cheveux !'] 
couleurs_cheveux = ['black', 'saddlebrown', 'darkorange', 'gold', 'lightgray']
longueur_cheveux = [4, 2, 0.5, 0.1]

## ALTAIR
selector = alt.selection_single(empty='all', fields=['sexe'])
base = alt.Chart(source).add_selection(selector).interactive()

zoom_possible = """ ✨ Zoom et filtre garçon/fille possible + infobulles si vous êtes sur ordinateur. ✨  
"""

def portrait_robot(data, sexe_oppose) :
    sexe_majo = data['sexe'].mode()[0]

    if sexe_oppose :
        sexe_majo = "Garçon" if sexe_majo == "Fille" else "Fille"

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

def formatting_pct(pct, allvals):
    absolute = int(round(pct/100.*np.sum(allvals)))
    return "{:d}\n({:.1f}%)".format(absolute, pct)
    
def sex_predictions(data, sexes, colors_sex):
    fig, ax = plt.subplots(figsize=(4,4))
    _, _, autotexts = ax.pie(data.value_counts(), 
            labels=sexes, labeldistance=1.15,
            wedgeprops = { 'linewidth' : 3, 'edgecolor' : 'white' },
            autopct=lambda pct: formatting_pct(pct, [x for x in data.value_counts()]),
            textprops={'size': 'larger', },
            colors=colors_sex)
    autotexts[0].set_color('white')
    autotexts[1].set_color('white')
    ax.axis('equal')
    return fig

def birth_announcement(source, sexes, sexe_oppose, colors_sex, couleurs_cheveux, ordre_couleur): 

    sexe_majo, birthday, taille, poids, chevelure, couleur, prenom_majo = portrait_robot(source, sexe_oppose)

    sex_color = colors_sex[sexes.index(sexe_majo)]
    winning_color_hair = couleurs_cheveux[ordre_couleur.index(couleur)]

    jour = format_datetime(birthday,"EEEE d MMMM yyyy 'à' H'h'mm ", locale='fr')
    fille = 'e' if sexe_majo == "Fille" else ''
    pronom = "elle" if sexe_majo == "Fille" else 'il'
    presque = "(presque) " if sexe_oppose else ""

    size = .4
    outer_colors = [winning_color_hair, 'white']
    inner_color = ['pink']

    fig, ax = plt.subplots()

    ax.pie([0.5, 0.5], radius=1, colors=outer_colors,
        wedgeprops=dict(width=size, edgecolor='w'),
        )

    ax.pie([1], radius=1-size, colors=inner_color,
        wedgeprops=dict(edgecolor='w'))

    rect = plt.Rectangle(
        # (left - lower corner), width, height
        (0.2, -0.1), size+0.22, 1.05, fill=False, color=sex_color, lw=4, 
        zorder=-100, transform=fig.transFigure, figure=fig,
        linestyle='--',
        #capstyle='round',
        sketch_params=1.1
    )
    fig.patches.extend([rect])

    plt.text(0, 1.15, f"C'est un{fille} {sexe_majo} !".upper(), ha='center', fontsize=10, fontfamily="serif")
    plt.text(0, -1, f"~ {prenom_majo} ~".upper(), ha='center', fontsize=25, color=sex_color, fontfamily="serif")
    plt.text(0, -1.3, f"Né{fille} le {jour}".upper(), ha="center", fontweight="ultralight", fontfamily="serif")
    plt.text(0, -1.5, f"{int(taille)} cm - {int(poids):1,} kg", ha="center", fontfamily="serif")
    plt.text(0, -1.75 , f"En plus, {pronom} a les cheveux {couleur.lower()} ({chevelure.lower()})", ha="center", fontsize=8, fontfamily="serif", color="grey")
    plt.text(0, -1.85, f"En tout cas, c'est ce que vous avez {presque}prédit ;)", ha="center", fontsize=8, fontfamily="serif", color="grey")

    if sexe_oppose :
        plt.text(0, 1.3, 'Version "sexe opposé"', ha='center', fontsize=8, fontfamily='serif', color=sex_color)

    return fig

def display_birthdate_pred(base, condition_color):

    d_day = alt.Chart(pd.DataFrame({
        'Date de naissance' : pd.date_range(start="2022-02-11T23:00:00", end="2022-02-12T22:59:59", periods=100)
    })).mark_tick(
        thickness=2, size=80, color='lightgrey', opacity=1
    ).encode(x='Date de naissance:T').interactive()

    pred_ticks = base.mark_tick(thickness=2, size=80, opacity=0.5).encode(
        alt.X('date:T', title="Date prédite"),
        color=condition_color,
        tooltip=[
            alt.Tooltip('date', title="Date prédite "),
            alt.Tooltip('heure', title="à "),
            alt.Tooltip('prenom', title="par "),
            alt.Tooltip('nom', title=" "),
        ],
    ).properties(height=135).interactive()

    return d_day, pred_ticks

def size_charts(base, condition_color, selector):
    taille_scale = alt.Scale(domain=(40, 60))
    poids_scale = alt.Scale(domain=(2000, 5000))
    tick_axis = alt.Axis(labels=False, ticks=False)
    area_args = {'opacity': .3, 'interpolate': 'step'}

    points = base.mark_circle().encode(
        alt.X('taille', scale=taille_scale, title='Taille (cm)'),
        alt.Y('poids', scale=poids_scale, title='Poids (g)'),
        color=condition_color,
        tooltip=[
            alt.Tooltip('poids', title="Poids"), 
            alt.Tooltip('taille', title='Taille'), 
            alt.Tooltip('prenom', title="par "),
            alt.Tooltip('nom', title=" ")
        ]
    )

    points_F = alt.Chart(pd.DataFrame({
        'Prénom': ['Florian'], 'Poids': [3290], 'Taille' : [50]
    })).mark_point(size=40, color='darkred').encode(
        alt.X('Taille', scale=taille_scale, title='Taille (cm)'),
        alt.Y('Poids', scale=poids_scale, title='Poids (g)'),
        tooltip=['Prénom', 'Poids', 'Taille']        
    ).interactive()

    points_H = alt.Chart(pd.DataFrame({
        'Prénom': ['Hélène'], 'Poids': [3170], 'Taille' : [48]
    })).mark_point(size=40, color='blue').encode(
        alt.X('Taille', scale=taille_scale, title='Taille (cm)'),
        alt.Y('Poids', scale=poids_scale, title='Poids (g)'),
        tooltip=['Prénom', 'Poids', 'Taille']        
    ).interactive()

    points = alt.layer(points, points_H, points_F).interactive()

    top_chart = base.mark_area(**area_args).encode(
        alt.X('taille:Q',
            bin=alt.Bin(maxbins=20, extent=taille_scale.domain),
            stack=None,
            title=''
            ),
        alt.Y('count()', stack=None, title=''),
        color=condition_color,
        tooltip=[alt.Tooltip('taille', title='Taille'), alt.Tooltip('count()', title='Nb ')]
    ).properties(height=80).transform_filter(selector)

    right_chart = base.mark_tick().encode(
        alt.X('sexe', axis=tick_axis, title=''),
        alt.Y('poids', axis=tick_axis, scale=poids_scale, title=''),
        tooltip=alt.Tooltip('poids'),
        color=condition_color
    )

    return top_chart, points, right_chart


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

def cool_hair_plot(df, longueur_cheveux, ordre_cheveux, couleurs_cheveux, ordre_couleur):

    width, radii, angles, theta, colors, nb_labels = make_hair_bars(
    df, longueur_cheveux, ordre_cheveux, couleurs_cheveux, ordre_couleur)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_axes([0, 0, 0.8, 0.8], polar=True)
    bottom = 5
    bars = ax.bar(theta, radii, width=(np.array(width)-0.01), bottom=bottom, color=colors)

    # mise en forme du graphique
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_xticks(angles)
    ax.set_xticklabels(["\n".join(wrap(r, 11, break_long_words=False)) for r in ordre_cheveux])
    ax.set_yticks([0, 10])
    ax.set_yticklabels([])
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.spines["start"].set_color("none")
    ax.spines["end"].set_color("none")
    ax.spines["polar"].set_color("none")

    # affichage des labels + rotation pour lisibilité et opacité
    rotations = np.rad2deg(theta)
    for x, bar, rotation, label in zip(theta, bars, rotations, nb_labels):
        new_rotation = 180 + rotation if rotation > 90 else rotation
        ax.text(x,bottom+bar.get_height()+1, label, size=7,
                ha='center', va='center', rotation=new_rotation, rotation_mode="anchor")  
        bar.set_alpha(0.7) 

    return fig

def get_dict_from_freq(freq_masc, freq_fem):
    dict_masc = dict(freq_masc)
    dict_fem = dict(freq_fem)

    dict_both = {
        k: dict_masc.get(k, 0) + dict_fem.get(k, 0) 
        for k in set(dict_masc) | set(dict_fem)
    }

    return dict_masc, dict_fem, dict_both

def most_voted_names(freq_masc, freq_fem):
    _, _, dict_both = get_dict_from_freq(freq_masc, freq_fem)

    max_freq = max(dict_both.values())
    prenoms_max = [name for name, freq in dict_both.items() if freq == max_freq]
    nb_freq = len(prenoms_max)

    return max_freq, prenoms_max, nb_freq

def both_gender_cloud(freq_masc, freq_fem, colors_sex, mask_path):
    mask = np.array(Image.open(mask_path))
    mask[mask == 0] = 255

    _, dict_fem, dict_both = get_dict_from_freq(freq_masc, freq_fem)

    color_to_words = { colors_sex[0] : list(dict_fem.keys()) }
    grouped_color_func = GroupedColorFunc(color_to_words, default_color = colors_sex[1])

    wordcloud = WordCloud(
        random_state=35,
        background_color = "white", 
        max_font_size=40,
        mask=mask,
        relative_scaling=1,
        color_func=grouped_color_func,
    ).generate_from_frequencies(dict_both)

    fig = plt.figure(figsize=(16,9))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    return fig

def one_gender_cloud(freq, color, mask_path):
    mask = np.array(Image.open(mask_path))
    mask[mask == 0] = 255

    wordcloud = WordCloud(
        random_state=35,
        background_color = "white", 
        max_font_size=40,
        mask=mask,
        color_func=get_single_color_func(color),
    ).generate_from_frequencies(dict(freq))

    fig = plt.figure(figsize=(16,9))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    return fig

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