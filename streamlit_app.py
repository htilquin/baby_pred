import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import altair as alt
from textwrap import wrap
from wordcloud import WordCloud
from PIL import Image

from utils import *

path_to_df = "processed_data.pkl"

st.markdown("# ✨ Team Crevette ✨")
st.write("~~")
st.markdown("Bienvenue sur la page de visualisation de vos pronostics !")

df = pd.read_pickle(path_to_df)
df["date"] = pd.to_datetime(df["date"], dayfirst=True)

### VARIABLES ALTAIR
with st.container():
    source = df
    sexes = ['Fille', 'Garçon']
    colors_sex = ['#1FC3AA', 'darkorange']
    color_scale = alt.Scale(domain=sexes, range=colors_sex)
    selector = alt.selection_single(empty='all', fields=['sexe'])
    condition_color = alt.condition(selector, 'sexe:N',
                        alt.value('lightgray'), scale=color_scale,
                        legend=alt.Legend(title="", labelFontSize=12, orient="right"))
    base = alt.Chart(source).add_selection(selector).interactive()

    zoom_possible = """##### ✨ Zoom et filtre garçon/fille possible, avec infobulles si vous êtes sur ordinateur. ✨  
    """

## INTRO + SEXE
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.write("~~")
        st.markdown("""Vous avez été **{}** personnes à participer.
        Si vous n'avez pas encore joué, il est toujours temps en [cliquant ici](https://form.jotform.com/bubbl3wrap/mini-mimi).        
        """. format(len(df)))
        st.markdown("**Merci ! 🤗**")
        st.write("Vous trouverez sur cette page un petit récap de vos pronostics...")
        st.write("""&nbsp;  
        Pour rappel, la personne avec le plus de points* remportera le poids (réel!) du bébé
        en chocolats et/ou bonbons, selon sa préférence.  
        &nbsp;&nbsp;🍬 &nbsp; 🍫 &nbsp; 🍭

        * barême à suivre... 🧮     
        """)

    ### SEXE 
    with col2 :
        fig, ax = plt.subplots(figsize=(4,4))
        ax.pie(df['sexe'].value_counts(), 
                labels=sexes, labeldistance=1.15,
                wedgeprops = { 'linewidth' : 3, 'edgecolor' : 'white' },
                autopct='%1.1f%%',
                colors=colors_sex)
        ax.axis('equal')
        st.pyplot(fig)

### DATE DE NAISSANCE
with st.container():
    st.markdown("&nbsp;")
    st.markdown("## 📆 &nbsp; Date de naissance")
    st.markdown("&nbsp;")
    st.markdown(zoom_possible)
    st.markdown("&nbsp;")
    st.markdown("🪧 Terme : **12 février 2022** (en gris clair).")

    ticks = base.mark_tick(thickness=2, size=80, opacity=0.5).encode(
        alt.X('date:T', title="Date prédite"),
        color=condition_color,
        tooltip=[
            alt.Tooltip('date', title="Date prédite "),
            alt.Tooltip('heure', title="à "),
            alt.Tooltip('prenom', title="par "),
            alt.Tooltip('nom', title=" "),
        ],
    ).properties(height=135).interactive()

    d_day = alt.Chart(pd.DataFrame({
        'Date de naissance' : pd.date_range(start="2022-02-11T23:00:00", end="2022-02-12T22:59:59", periods=100)
    })).mark_tick(
        thickness=2, size=80, color='lightgrey', opacity=1
    ).encode(x='Date de naissance:T').interactive()

    st.altair_chart(
        alt.layer(d_day, ticks),
        use_container_width=True
    )

### Jour et heure de naissance ?


### TAILLE + POIDS
with st.container():
    st.markdown('## 📏 &nbsp; Mensurations...')
    st.markdown("&nbsp;")
    st.markdown(zoom_possible)
    st.markdown("&nbsp;")

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

    st.altair_chart(top_chart & (points | right_chart), use_container_width=True)
# ajouter les poids/taille de Hélène et Flo ?

### CAPILARITE
with st.container():
    st.markdown("## 💈 &nbsp; Et les cheveux ?")

    ordre_cheveux = ['Maxi chevelure !', 'Chevelure classique', 'Juste assez pour ne pas être chauve...', 'Pas de cheveux !']
    ordre_couleur = ['Noirs', 'Bruns', 'Roux', 'Blonds', 'Pas de cheveux... Pas de cheveux !'] 
    couleurs_cheveux = ['black', 'saddlebrown', 'darkorange', 'gold', 'lightgray']
    longueur_cheveux = [4, 2, 0.5, 0.1]

    width, radii, angles, theta, colors, nb_labels = make_bars_specifics(
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
        lab = ax.text(x,bottom+bar.get_height()+1, label, size=7,
                        ha='center', va='center', rotation=new_rotation, rotation_mode="anchor")  
        bar.set_alpha(0.7) 
        
    st.pyplot(fig)

### PRENOM
with st.container():
    st.markdown("## ✏️ &nbsp; Et enfin... le prénom !")
    st.markdown("&nbsp;")

    dict_masc = dict(df.prenom_masc.value_counts())
    dict_fem = dict(df.prenom_fem.value_counts())
    dict_both = {
        k: dict_masc.get(k, 0) + dict_fem.get(k, 0) 
        for k in set(dict_masc) | set(dict_fem)
    }

    max_freq = max(dict_both.values())
    prenoms_max = [name for name, freq in dict_both.items() if freq == max_freq]
    nb_freq = len(prenoms_max)

    st.markdown("""Plus un prénom est écrit en gros, plus vous avez été nombreux à miser dessus 😉 
    &nbsp;  
    *N.B. : Les prénoms n'ont pas non plus été proposés 50 fois, le maximum étant "seulement" de {} fois pour le{} prénom{} {} !*
    """.format(max_freq,
            's' if nb_freq>1 else "",
            's' if nb_freq>1 else "",           
    ", ".join(prenoms_max[:-1]) + (" & " if nb_freq>1 else "") + prenoms_max[-1]))

    mask = np.array(Image.open("cloud-icon.png"))
    mask[mask == 0] = 255

    color_to_words = { colors_sex[0] : list(dict_fem.keys()) }
    grouped_color_func = GroupedColorFunc(color_to_words, default_color = colors_sex[1])

    wordcloud = WordCloud(
        random_state=35,
        background_color = "white", 
        width=1000, height=600,
        max_font_size=40,
        mask=mask,
        relative_scaling=1,
        color_func=grouped_color_func,
    ).generate_from_frequencies(dict_both)

    fig = plt.figure(figsize=(16,9))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    st.pyplot(fig)

## Fille et garçon séparéS !
with st.container() :
    col1, col2 = st.columns(2)

    mask_thunder = np.array(Image.open("thunder.png"))
    mask_thunder[mask_thunder == 0] = 255

    with col1 :
        wordcloud = WordCloud(
            random_state=35,
            background_color = "white", 
            max_font_size=40,
            mask=mask_thunder,
            color_func=get_single_color_func(colors_sex[0]),
        ).generate_from_frequencies(dict_fem)

        fig = plt.figure(figsize=(16,9))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")

        st.pyplot(fig)

    with col2 :
        wordcloud = WordCloud(
            random_state=35,
            background_color = "white", 
            max_font_size=40,
            mask=mask_thunder,
            color_func=get_single_color_func(colors_sex[1]),
        ).generate_from_frequencies(dict_masc)

        fig = plt.figure(figsize=(16,9))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")

        st.pyplot(fig)

# FOOTER
st.write("### &nbsp;")
st.markdown("##### Fait avec 💖 par Hélène.")

