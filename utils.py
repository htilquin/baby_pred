import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import altair as alt

from babel.dates import format_datetime
from datetime import datetime
from textwrap import wrap

from wordcloud import WordCloud, get_single_color_func
from PIL import Image

from Levenshtein import distance as levdist

path_to_df = "processed_data.pkl"
df = pd.read_pickle(path_to_df)
df["date"] = pd.to_datetime(df["date"], dayfirst=True)

viz_prono = "Pronostics"
profile_baby = "Faire-part de naissance du bébé"
results = "Résultats"
sexes = ['Fille', 'Garçon']

dict_cheveux = {'ordre_cheveux' : ['Maxi chevelure !', 'Chevelure classique', 'Juste assez pour ne pas être chauve...', 'Pas de cheveux !'],
'ordre_couleur' : ['Noirs', 'Bruns', 'Roux', 'Blonds', 'Pas de cheveux... Pas de cheveux !'],
'couleurs_cheveux' : ['black', 'saddlebrown', 'darkorange', 'gold', 'lightgray'],
'longueur_cheveux' : [4, 2, 0.5, 0.1]
}

dict_baby = {
    'birthday' : pd.to_datetime('23/01/2022 03:36:00'),
    'prenom' : 'Raphaël',
    'sexe' : 'Garçon',
    'taille': 47,
    'poids' : 2565,
    'chevelure' : 'Maxi chevelure !',
    'couleur_cheveux' : 'Noirs',
    'color' : 'indigo',
}

dict_F = {
    'prenom' : 'Florian',
    'sexe' : 'Garçon',
    'taille': 50,
    'poids' : 3290,
    'chevelure' : 'Maxi chevelure !',
    'couleur_cheveux' : 'Noirs',
    'color' : 'darkred'
}

dict_H = {
    'prenom' : 'Hélène',
    'sexe' : 'Fille',
    'taille': 48,
    'poids' : 3170,
    'chevelure' : 'Maxi chevelure !',
    'couleur_cheveux' : 'Noirs',
    'color' : 'blue'
}

## ALTAIR
selector = alt.selection_single(empty='all', fields=['sexe'])
base = alt.Chart(df).add_selection(selector).interactive()

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

    dict_profile = {
        'birthday' : data.date.median(),
        'prenom' : prenom_majo,
        'sexe' : sexe_majo,
        'sexe_oppose' : sexe_oppose,
        'taille': data.taille.median(),
        'poids' : data.poids.median(),
        'chevelure' : data['longueur_cheveux'].mode()[0],
        'couleur_cheveux' : data['couleur_cheveux'].mode()[0],
    }

    return dict_profile 

def display_birth_announcement(dict_baby, sexes, colors_gender, dict_cheveux): 

    sex_color = colors_gender[sexes.index(dict_baby['sexe'])]
    winning_color_hair = dict_cheveux['couleurs_cheveux'][dict_cheveux['ordre_couleur'].index(dict_baby['couleur_cheveux'])]    

    jour = format_datetime(dict_baby['birthday'],"EEEE d MMMM yyyy 'à' H'h'mm ", locale='fr')
    fille = 'e' if dict_baby['sexe'] == "Fille" else ''
    pronom = "elle" if dict_baby['sexe'] == "Fille" else 'il'

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

    plt.text(0, 1.15, f"C'est un{fille} {dict_baby['sexe']} !".upper(), ha='center', fontsize=10, fontfamily="serif")
    plt.text(0, -1, f"~ {dict_baby['prenom']} ~".upper(), ha='center', fontsize=25, color=sex_color, fontfamily="serif")
    plt.text(0, -1.3, f"Né{fille} le {jour}".upper(), ha="center", fontweight="ultralight", fontfamily="serif")
    plt.text(0, -1.5, f"{int(dict_baby['taille'])} cm - {int(dict_baby['poids']):1,} kg", ha="center", fontfamily="serif")

    if dict_baby['chevelure'] == 'Pas de cheveux !':
        plt.text(0, -1.75 , f"En plus, {pronom} est chauve !", ha="center", fontsize=8, fontfamily="serif", color="grey")
    
    else:
        plt.text(0, -1.75 , f"En plus, {pronom} a les cheveux {dict_baby['couleur_cheveux'].lower()} ({dict_baby['chevelure'].lower()})",
         ha="center", fontsize=8, fontfamily="serif", color="grey")   

    ## Faux faire-part
    if 'sexe_oppose' in dict_baby.keys():
        presque = "(presque) " if dict_baby['sexe_oppose'] else ""
        plt.text(0, -1.85, f"En tout cas, c'est ce que vous avez {presque}prédit ;)",
         ha="center", fontsize=8, fontfamily="serif", color="grey")

        if dict_baby['sexe_oppose'] :
            plt.text(0, 1.3, 'Version "sexe opposé"',
             ha='center', fontsize=8, fontfamily='serif', color=sex_color)

    return fig

def formatting_pct(pct, allvals):
    absolute = int(round(pct/100.*np.sum(allvals)))
    return "{:d}\n({:.1f}%)".format(absolute, pct)
    
def sex_predictions(data, sexes, colors_gender):
    fig, ax = plt.subplots(figsize=(4,4))
    _, _, autotexts = ax.pie(data.value_counts(), 
            labels=sexes, labeldistance=1.15,
            wedgeprops = { 'linewidth' : 3, 'edgecolor' : 'white' },
            autopct=lambda pct: formatting_pct(pct, [x for x in data.value_counts()]),
            textprops={'size': 'larger', },
            colors=colors_gender)
    autotexts[0].set_color('white')
    autotexts[1].set_color('white')
    ax.axis('equal')
    return fig

def display_birthdate_pred(base, condition_color_gender):

    expected_d_day = alt.Chart(pd.DataFrame({
        'Date de naissance' : pd.date_range(start="2022-02-11T23:00:00", end="2022-02-12T22:59:59", periods=100)
    })).mark_tick(
        thickness=2, size=80, color='lightgrey', opacity=1
    ).encode(x='Date de naissance:T').interactive()

    d_day = alt.Chart(pd.DataFrame([dict_baby])).mark_tick(
        thickness=2, size=80, opacity=1, color='indigo'
    ).encode(
        alt.X('birthday:T'),
        tooltip=[
            alt.Tooltip('prenom', title=' '),
            alt.Tooltip('birthday', title='  ', format=format_datetime(dict_baby['birthday'],"EEEE d MMMM yyyy 'à' H'h'mm ", locale='fr'))
        ]
    ).interactive()

    pred_ticks = base.mark_tick(thickness=2, size=80, opacity=0.5).encode(
        alt.X('date:T', title="Date de naissance"),
        color=condition_color_gender,
        tooltip=[
            alt.Tooltip('date', title="Date prédite "), #, format='%d-%m à %H:%M'
            alt.Tooltip('heure', title="à "),
            alt.Tooltip('prenom', title="par "),
            alt.Tooltip('nom', title=" "),
        ],
    ).properties(height=135).interactive()

    return expected_d_day, d_day, pred_ticks

def size_charts(base, condition_color_gender, selector): #colors_scale_family
    taille_scale = alt.Scale(domain=(40, 60))
    poids_scale = alt.Scale(domain=(2000, 5000))
    tick_axis = alt.Axis(labels=False, ticks=False)
    area_args = {'opacity': .3, 'interpolate': 'step'}

    points = base.mark_circle().encode(
        alt.Y('taille', scale=taille_scale, title='Taille (cm)'),
        alt.X('poids', scale=poids_scale, title='Poids (g)'),
        color=condition_color_gender,
        tooltip=[
            alt.Tooltip('poids', title="Poids "), 
            alt.Tooltip('taille', title='Taille '), 
            alt.Tooltip('prenom', title="par "),
            alt.Tooltip('nom', title=" ")
        ]
    )

    points_F = alt.Chart(pd.DataFrame([dict_F])
    ).mark_point(size=50, color=dict_F['color']).encode(
        alt.Y('taille', scale=taille_scale, title='Taille (cm)'),
        alt.X('poids', scale=poids_scale, title='Poids (g)'),
        tooltip=[
            alt.Tooltip('prenom', title=" "),
            alt.Tooltip('poids', title="Poids "), 
            alt.Tooltip('taille', title='Taille '), 
        ]       
    ).interactive().transform_filter(selector)

    points_H = alt.Chart(pd.DataFrame([dict_H])
    ).mark_point(size=50, color=dict_H['color']).encode(
        alt.Y('taille', scale=taille_scale, title='Taille (cm)'),
        alt.X('poids', scale=poids_scale, title='Poids (g)'),
        tooltip=[
            alt.Tooltip('prenom', title=" "),
            alt.Tooltip('poids', title="Poids "), 
            alt.Tooltip('taille', title='Taille '), 
        ]      
    ).interactive().transform_filter(selector)

    points_R = alt.Chart(pd.DataFrame([dict_baby])
    ).mark_point(size=50, color=dict_baby['color']).encode(
        alt.Y('taille', scale=taille_scale, title='Taille (cm)'),
        alt.X('poids', scale=poids_scale, title='Poids (g)'),
        tooltip=[
            alt.Tooltip('prenom', title=" "),
            alt.Tooltip('poids', title="Poids "), 
            alt.Tooltip('taille', title='Taille '), 
        ]       
    ).interactive().transform_filter(selector)

    # points_all = alt.Chart(pd.DataFrame([dict_F, dict_H, dict_baby])
    # ).mark_point(size=50).encode(
    #     alt.Y('taille', title='Taille (cm)'),
    #     alt.X('poids', title='Poids (g)'),
    #     color=alt.Color('prenom', scale=colors_scale_family),
    #     tooltip=[
    #         alt.Tooltip('prenom', title=" "),
    #         alt.Tooltip('poids', title="Poids "), 
    #         alt.Tooltip('taille', title='Taille '), 
    #     ]       
    # ).interactive()

    points = alt.layer(points, points_H, points_F, points_R).interactive()
    #points = alt.layer(points, points_all).interactive()

    right_chart = base.mark_area(**area_args).encode(
        alt.Y('taille:Q',
            bin=alt.Bin(maxbins=20, extent=taille_scale.domain),
            stack=None,
            title='', axis=tick_axis,
            ),
        alt.X('count()', stack=None, title=''),
        color=condition_color_gender,
        tooltip=[alt.Tooltip('taille', title='Taille'), alt.Tooltip('count()', title='Nb ')]
    ).properties(width=100).transform_filter(selector)

    top_chart = base.mark_tick().encode(
        alt.Y('sexe', axis=tick_axis, title=''),
        alt.X('poids', axis=tick_axis, scale=poids_scale, title=''),
        tooltip=alt.Tooltip('poids'),
        color=condition_color_gender
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

def make_hair_bars(df, dict_cheveux) :
    width = []
    radii = []
    angles = []
    acc = []
    theta = []
    colors = []
    nb_labels = []

    for longueur in dict_cheveux['ordre_cheveux'] :
        nb = len(df.loc[(df['longueur_cheveux'] == longueur)])
        angles, acc = calcul_angles(angles, nb/len(df), acc)
        
        for couleur in dict_cheveux['ordre_couleur']:
            nb = len(df.loc[(df['longueur_cheveux'] == longueur) & (df['couleur_cheveux'] == couleur)])
            if nb > 0 :
                nb_labels.append("{} - {}".format(couleur, nb))
                radii.append(dict_cheveux['longueur_cheveux'][dict_cheveux['ordre_cheveux'].index(longueur)])
                colors.append(dict_cheveux['couleurs_cheveux'][dict_cheveux['ordre_couleur'].index(couleur)])
                theta, width = calcul_angles(theta, nb/len(df), width)
    
    # modif label pour cause de texte trop long !
    new_labels = []
    for texte in nb_labels :
        new_label = texte.replace('Pas de cheveux... Pas de cheveux !', 'Chauve !')
        new_labels.append(new_label)

    return width, radii, angles, theta, colors, new_labels

def cool_hair_plot(df, dict_cheveux):

    width, radii, angles, theta, colors, nb_labels = make_hair_bars(df, dict_cheveux)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_axes([0, 0, 0.8, 0.8], polar=True)
    bottom = 5
    bars = ax.bar(theta, radii, width=(np.array(width)-0.01), bottom=bottom, color=colors)

    # mise en forme du graphique
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_xticks(angles)
    ax.set_xticklabels(["\n".join(wrap(r, 11, break_long_words=False)) for r in dict_cheveux['ordre_cheveux']])
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

def both_gender_cloud(freq_masc, freq_fem, colors_gender, mask_path):
    mask = np.array(Image.open(mask_path))
    mask[mask == 0] = 255

    _, dict_fem, dict_both = get_dict_from_freq(freq_masc, freq_fem)

    color_to_words = { 
        colors_gender[0] : list(dict_fem.keys()),
        dict_baby['color'] : [dict_baby['prenom']]
         }
    grouped_color_func = GroupedColorFunc(color_to_words, default_color = colors_gender[1])

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

    color_to_words = { 
        dict_baby['color'] : [dict_baby['prenom']]
         }
    grouped_color_func = GroupedColorFunc(color_to_words, default_color = color)

    wordcloud = WordCloud(
        random_state=35,
        background_color = "white", 
        max_font_size=40,
        mask=mask,
        color_func=grouped_color_func,
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

def set_hair_points(dict_baby, dict_cheveux):

    index_chevelure = dict_cheveux['ordre_cheveux'].index(dict_baby['chevelure'])
    coeff_cheveux = {}
    for chevelure in dict_cheveux['ordre_cheveux'] :
        dist = abs(index_chevelure - dict_cheveux['ordre_cheveux'].index(chevelure))
        coeff_cheveux[chevelure] = 1-dist*1/3

    dist_couleur = {'Pas de cheveux... Pas de cheveux !': 0, 'Roux':0.25, 'Blonds' :0.3, 'Bruns':0.75, 'Noirs': 1}
    coeff_couleur = {}
    for couleur in dict_cheveux['ordre_couleur'] :
        coeff_couleur[couleur] = 1 - (abs(dist_couleur[dict_baby['couleur_cheveux']]-dist_couleur[couleur]))

    return  coeff_cheveux, coeff_couleur

def calculate_scores(df_pred, dict_baby) :
    df = df_pred.copy()
    df = df.replace(np.nan,'',regex=True)
    df.prenom_masc = df.prenom_masc.astype(str)
    df.prenom_fem = df.prenom_fem.astype(str)

    name_dist = lambda x: 1 if levdist(dict_baby['prenom'].lower(), x.lower())<3 else np.maximum(1-0.2*(levdist(dict_baby['prenom'], x.lower())-2)-0.05*np.abs(len(x)-7)**0.5,0)

    delta = np.abs(df.date-dict_baby['birthday']).view(int)*1e-15    
    df['score_date'] = (np.exp(-0.5*delta**2))
    df['score_sexe'] = (df.sexe==dict_baby['sexe'])*(1-df['sexe'].value_counts()[dict_baby['sexe']]/len(df))

    is_boy = dict_baby['sexe']=="Garçon"

    df['score_prenom'] = np.maximum(
        df.prenom_masc.apply(name_dist)*(1 if is_boy else 0.8), 
        df.prenom_fem.apply(name_dist)*(0.8 if is_boy else 1)
    )

    df['score_poids'] = np.exp(-((dict_baby['poids']-df.poids)/500)**2)
    df['score_taille'] = np.maximum(0,(1-0.15*np.abs(dict_baby['taille']-df.taille)))

    coeff_cheveux, coeff_couleur = set_hair_points(dict_baby, dict_cheveux)
    df['score_cheveux'] = df.longueur_cheveux.apply(lambda x : coeff_cheveux[x])
    df['score_couleur'] = df.couleur_cheveux.apply(lambda x : coeff_couleur[x])
    df['score_cheveux'] = 0.5*(df.score_cheveux+df.score_couleur)
    df.drop('score_couleur', axis=1, inplace=True)

    scores = [col for col in df.columns if 'score' in col ]
    df['score'] = df[scores].sum(1)

    place = np.argsort(df.score.to_numpy())[::-1].tolist()
    df['place'] = [place.index(i)+1 for i in range(len(df))]

    df = df.sort_values(by='score', ascending=False)
    df = df.set_index('place')

    return df.round(3)

def beautify_df(df) :   

    mapper = { 'prenom' : 'Prénom',
        'nom' : 'Nom',
        'sexe' : 'Sexe',
        'date' : 'Date de naissance',
        'poids' : 'Poids',
        'taille' : 'Taille',
        'longueur_cheveux' : 'Longueur des cheveux',
        'couleur_cheveux' : 'Couleur des cheveux',
        'prenom_masc' : 'Prénom masculin',
        'prenom_fem' : 'Prénom féminin',
        'heure' : 'Heure',
        'jour' : 'Jour',
        'score_date' : 'Score date',
        'score_sexe' : 'Score sexe',
        'score_prenom' : 'Score prenom',
        'score_poids' : 'Score poids',
        'score_taille' : 'Score taille',
        'score_cheveux' : 'Score cheveux',
        'score' : 'Score total',
        'place' : 'Classement'
    }

    df.rename(columns=mapper, inplace=True)

    cols = df.columns.tolist()
    cols = [cols[0], cols[7], *cols[1:7] , *cols[8:10] , *cols[12:]]

    return df[cols]

def df_styler(df):
    styler = df.style.format({
        "Date de naissance" : lambda t : format_datetime(t,"d MMMM yyyy 'à' H'h'mm ", locale='fr')
    })

    return styler

def function_select(df):
    options = {}
    for _, row in df.iterrows():
        options[f''] = pd.Series(row)
    return options

def scores_participant(serie_participant, len_df) :

    st.markdown(f"<center><b>~ &nbsp; ~ &nbsp; Prédictions de {serie_participant['Prénom']} {serie_participant['Nom']} &nbsp; ~ &nbsp; ~</b></center>",
    unsafe_allow_html=True)

    st.markdown(f"<center><b>Score total</b> : {serie_participant['Score total']} pts</center>", unsafe_allow_html=True)
    st.markdown(f"<center><b>Classement</b> : {serie_participant.place} / {len_df}</center>", unsafe_allow_html=True)

    st.write('&nbsp;')

    col0, col1 = st.columns((2,1))
    with col0 :
        st.markdown(f"**Sexe** : {serie_participant['Sexe']}")
        #fmt = "%-d %B à %-Hh%M"
        date_predite = format_datetime(datetime.strptime(serie_participant['Date de naissance'], '%Y-%m-%d %H:%M:%S'),"d MMMM yyyy 'à' H'h'mm ", locale='fr')
        st.markdown(f"**Date de naissance** : {date_predite}")
        st.markdown(f"**Poids** : {serie_participant['Poids']} g")
        st.markdown(f"**Taille** : {serie_participant['Taille']} cm")
        st.markdown(f"**Cheveux** : {serie_participant['Longueur des cheveux']} - {serie_participant['Couleur des cheveux']}")
        double_prenom = True if len(serie_participant['Prénom masculin'])>1 and len(serie_participant['Prénom féminin'])>1 else False
        st.markdown(f"""**Prénom{'s' if double_prenom else ''}** : {serie_participant['Prénom masculin']} 
        {'&nbsp; & &nbsp;' if double_prenom else "&nbsp;"} 
        {serie_participant['Prénom féminin']}""")

    with col1 :
        st.markdown(f"{serie_participant['Score sexe']} pt")
        st.markdown(f"{serie_participant['Score date']} pt")
        st.markdown(f"{serie_participant['Score poids']} pt")
        st.markdown(f"{serie_participant['Score taille']} pt")
        st.markdown(f"{serie_participant['Score cheveux']} pt")
        st.markdown(f"""{serie_participant['Score prenom']} pt""")

    

footer="""<style>
a:link , a:visited{
color: red;
background-color: transparent;
text-decoration: none;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: gray;
text-align: center;
}
</style>
<div class="footer">
<p>Made with 💖 by Hélène T.</p>
</div>
"""