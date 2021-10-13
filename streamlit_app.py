import streamlit as st
import pandas as pd
import numpy as np

from datetime import date

import matplotlib.pyplot as plt
import altair as alt

from utils import *

st.set_page_config(
    page_title="Team Crevette", 
    page_icon="✨", 
    layout='centered', 
    initial_sidebar_state='auto', 
    menu_items=None)


with st.container():
    today = date.today()
    dpa = date(2022, 2, 12)
    days = (dpa-today).days

    st.markdown(f"# ✨ Team Crevette ✨ J - {days} ! ✨")
    st.write("~~")
    st.markdown("""Bienvenue sur la page de visualisation de vos pronostics !
    \nVous avez été **{}** personnes à participer.
    Si vous n'avez pas encore joué, il est toujours temps en [cliquant ici](https://form.jotform.com/bubbl3wrap/mini-mimi).        
    """. format(len(df)))
    st.markdown("Je me suis bien amusée à faire cette page, alors **merci à vous** ! 🤗")
    st.write("""&nbsp;  
    Pour rappel, la personne avec le plus de points* remportera le poids (réel!) du bébé
    en chocolats et/ou bonbons, selon sa préférence.  
    &nbsp; &nbsp;🍬 &nbsp; 🍫 &nbsp; 🍭
    """)
    st.write("🧮 *&nbsp;barême à suivre...")
    st.write("~~")

   ### SIDEBAR
    st.markdown("**Fun place !**")
    st.markdown("<small>Choisissez votre mode d'affichage ci-dessous. Vous pouvez aussi changer les couleurs (ça ne sert à rien mais c'est rigolo).</small>", unsafe_allow_html=True)
    st.write("Cliquez sur la couleur à modifier, sélectionnez la nouvelle couleur, puis cliquez à côté pour valider.")
    cols = st.columns(3)
    color_girl = cols[0].color_picker(label="Couleur Fille", value='#1FC3AA')
    color_boy = cols[1].color_picker(label="Couleur Garçon", value='#ff8c00')

    display = st.radio("Visualisations à afficher", 
    (viz_prono, robot_baby))
    if display == robot_baby :
        sexe_oppose = st.checkbox("Voir le faire-part du sexe opposé.")

## COLOR-RELATED VARIABLES
colors_sex = [color_girl, color_boy]
color_scale = alt.Scale(domain=sexes, range=colors_sex)
condition_color = alt.condition(selector, 'sexe:N',
                    alt.value('lightgray'), scale=color_scale,
                    legend=alt.Legend(title="", labelFontSize=12, orient="right"))

## INTRO
with st.container():
    col1, col0, col2 = st.columns((10,2,10))

    with col1:
        if display == robot_baby :
            st.markdown("&nbsp;")
            st.markdown("**Et voici le tant attendu faire-part du bébé !!**")
            st.write("Enfin, d'après vos pronostics, bien sûr...")

            with st.expander("Explications..."):
                    st.markdown("""<small>Rempli avec les médianes pour la date de naissance, la taille et le poids du bébé, et le choix de la majorité pour la longueur et la couleur des cheveux.</small>
                    \n<small>Le prénom est celui qui a été le plus donné pour le sexe majoritaire, par ceux qui ont prédit ce sexe.</small>
                    \n<small>*Par exemple : s'il y a une majorité de "garçon", c'est le prénom masculin le plus donné par ceux qui ont prédit "garçon".*</small>
                    """, unsafe_allow_html=True)

## CAMEMBERT SEXE
        else :
            st.markdown("## ⚤ &nbsp; Sexe")
            fig = sex_predictions(df['sexe'], sexes, colors_sex)
            st.pyplot(fig)

### FAIRE-PART
    with col2 :
        if display == robot_baby :
            fig = birth_announcement(source, sexes, sexe_oppose, colors_sex, couleurs_cheveux, ordre_couleur)
            st.pyplot(fig)


if display == viz_prono :
    ### DATE DE NAISSANCE
    with st.container():
        st.markdown("&nbsp;")
        st.markdown("## 📆 &nbsp; Date de naissance")
        st.markdown("&nbsp;")
        st.markdown(zoom_possible)
        st.markdown("🪧 Terme : **12 février 2022** (en gris clair).")

        d_day, pred_ticks = display_birthdate_pred(base, condition_color)
        st.altair_chart(alt.layer(d_day, pred_ticks), use_container_width=True)


    ### TAILLE + POIDS
    with st.container():
        st.markdown('## 📏 &nbsp; Mensurations...')
        st.markdown("&nbsp;")
        st.markdown(zoom_possible)
        st.markdown("&nbsp;")

        st.markdown("En rouge foncé, Florian et en bleu, Hélène !")

        top_chart, points, right_chart = size_charts(base, condition_color, selector)
        st.altair_chart(top_chart & (points | right_chart), use_container_width=True)
        

    # ajouter les poids/taille de Hélène et Flo ?

    ### CAPILARITE
    with st.container():
        st.markdown("## 💈 &nbsp; Et les cheveux ?")      
        fig = cool_hair_plot(df, longueur_cheveux, ordre_cheveux, couleurs_cheveux, ordre_couleur)
        st.pyplot(fig)

    ### PRENOM
    with st.container():
        st.markdown("## ✏️ &nbsp; Et enfin... le prénom !")
        st.markdown("&nbsp;")

        freq_masc = df.prenom_masc.value_counts()
        freq_fem = df.prenom_fem.value_counts()
        max_freq, prenoms_max, nb_freq = most_voted_names(freq_masc, freq_fem)

        st.markdown("""Plus un prénom est écrit en gros, plus vous avez été nombreux à miser dessus 😉 
        &nbsp;  
        *N.B. : Les prénoms n'ont pas non plus été proposés 50 fois, le maximum étant "seulement" de {} fois pour le{} prénom{} {} !*
        """.format(max_freq,
                's' if nb_freq>1 else "",
                's' if nb_freq>1 else "",           
        ", ".join(prenoms_max[:-1]) + (" & " if nb_freq>1 else "") + prenoms_max[-1]))

        fig = both_gender_cloud(freq_masc, freq_fem, colors_sex, "cloud-icon.png")
        st.pyplot(fig)


    ## Fille et garçon séparés !
    with st.container() :

        mask_path_thunder = "thunder.png"    
        fig_female = one_gender_cloud(freq_fem, colors_sex[0], mask_path_thunder)
        fig_male = one_gender_cloud(freq_masc, colors_sex[1], mask_path_thunder)

        col1, col2 = st.columns(2)
        col1.pyplot(fig_female)
        col2.pyplot(fig_male)


# FOOTER
st.write("### &nbsp;")
st.markdown("<small>**Fait avec 💖 par Hélène.**</small>", unsafe_allow_html=True)
