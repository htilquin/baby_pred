import streamlit as st
from datetime import date
from babel.dates import format_datetime
import altair as alt

from utils import *
import pandas as pd

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

    st.markdown(f"# ✨ Team Crevette ✨") #J - {days} ! ✨
    st.write("~~")
    st.markdown("""Bienvenue sur la page de visualisation de vos pronostics !
    \nVous avez été **{}** personnes à participer.
    {} étant là, il est bien entendu trop tard pour participer...        
    """. format(len(df), dict_baby['prenom']))
    st.markdown("Je me suis bien amusée à faire cette page, alors **merci à vous** ! 🤗")
    st.write("""&nbsp;  
    Pour rappel, les 5 personnes avec le plus de points se partageront le poids (réel!) du bébé
    en chocolats et/ou bonbons (selon préférence).  
    &nbsp; &nbsp;🍬 &nbsp; 🍫 &nbsp; 🍭
    """)
    st.write("~~")

   ### SIDEBAR
    st.markdown("**Fun place !**")
    st.markdown("<small>Choisissez votre mode d'affichage ci-dessous. Vous pouvez aussi changer les couleurs (ça ne sert à rien mais c'est rigolo).</small>", unsafe_allow_html=True)
    
    display = st.radio("Visualisations à afficher", 
    (viz_prono, profile_baby, results))
    
    help_color="Cliquez sur la couleur à modifier, sélectionnez la nouvelle couleur, puis cliquez à côté pour valider."
    cols = st.columns(3)
    color_girl = cols[0].color_picker(label="Couleur Fille", value='#1FC3AA', help=help_color)
    color_boy = cols[1].color_picker(label="Couleur Garçon", value='#ff8c00', help=help_color)

    ## COLOR-RELATED VARIABLES
    colors_gender = [color_girl, color_boy]
    color_scale_gender = alt.Scale(domain=sexes, range=colors_gender)
    condition_color_gender = alt.condition(selector, 'sexe:N',
                        alt.value('lightgray'), scale=color_scale_gender,
                        legend=alt.Legend(title="", labelFontSize=12, orient="right"))


## INTRO
with st.container():

    ### FAIRE-PART
    if display == profile_baby :

        col0, col1, col2 = st.columns((1,4,1))
        
        with col1:
            st.markdown("&nbsp;")
            st.markdown("#### Et voici le tant attendu faire-part du bébé !!")
            
            fake_announcement = st.checkbox("Voir le faire-part basé sur les pronostics.")
            if fake_announcement :
                sexe_oppose = st.checkbox("Voir le faire-part du sexe opposé.")

            if fake_announcement :
                st.write("Enfin, d'après vos pronostics...")

                dict_profile = portrait_robot(df, sexe_oppose)
                fig = display_birth_announcement(dict_profile, sexes, colors_gender, dict_cheveux)

                with st.expander("Explications..."):
                        st.markdown("""<small>Rempli avec les valeurs médianes pour la date de naissance, la taille et le poids du bébé, et le choix de la majorité pour la longueur et la couleur des cheveux.</small>
                        \n<small>Le prénom est celui qui a été le plus donné pour le sexe majoritaire, par ceux qui ont prédit ce sexe.</small>
                        \n<small>*Par exemple : s'il y a une majorité de "garçon", c'est le prénom masculin le plus donné par ceux qui ont prédit "garçon".*</small>
                        """, unsafe_allow_html=True)

            else :
                fig = display_birth_announcement(dict_baby, sexes, colors_gender, dict_cheveux)
                
            st.pyplot(fig)

    ### VIZ PRONOS
    if display == viz_prono :

        ## CAMEMBERT SEXE
        st.markdown("&nbsp;")
        col1, col0, col2 = st.columns((2,5,2))

        col0.markdown("## ⚤ &nbsp; Sexe")
        fig = sex_predictions(df['sexe'], sexes, colors_gender)
        col0.pyplot(fig)

        ### DATE DE NAISSANCE
        with st.container():
            st.markdown("&nbsp;")
            st.markdown("## 📆 &nbsp; Date de naissance")
            st.markdown("&nbsp;")
            st.markdown(zoom_possible)
            st.markdown("🪧 Terme : **12 février 2022** (en gris clair).")
            st.markdown("🥳 Naissance de {} : **{}** (en violet).".format(dict_baby['prenom'],format_datetime(dict_baby['birthday'],"d MMMM yyyy 'à' H'h'mm ", locale='fr') ))

            expected_d_day, d_day, pred_ticks = display_birthdate_pred(base, condition_color_gender)
            st.altair_chart(alt.layer(expected_d_day, d_day, pred_ticks), use_container_width=True)


        ### TAILLE + POIDS
        with st.container():
            st.markdown('## 📏 &nbsp; Mensurations...')
            st.markdown("&nbsp;")
            st.markdown(zoom_possible)

            st.markdown("Les mensurations à la naissance de Florian sont en rouge foncé, celles d'Hélène en bleu.")
            st.markdown("Les mensurations de **Raphaël** sont en **violet** !")
            st.markdown("&nbsp;")

            top_chart, points, right_chart = size_charts(base, condition_color_gender, selector)
            st.altair_chart(top_chart & (points | right_chart), use_container_width=True)
            
        ### CAPILARITE
        with st.container():
            st.markdown("## 💈 &nbsp; Et les cheveux ?")      
            fig = cool_hair_plot(df, dict_cheveux)
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

            fig = both_gender_cloud(freq_masc, freq_fem, colors_gender, "cloud-icon.png")
            st.pyplot(fig)


        ## Fille et garçon séparés !
        with st.container() :

            mask_path_thunder = "thunder.png"    
            fig_female = one_gender_cloud(freq_fem, colors_gender[0], mask_path_thunder)
            fig_male = one_gender_cloud(freq_masc, colors_gender[1], mask_path_thunder)

            col1, col2 = st.columns(2)
            col1.pyplot(fig_female)
            col2.pyplot(fig_male)

    if display == results:
        ## Profil bébé
        with st.container() :

            jour = format_datetime(dict_baby['birthday'],"EEEE d MMMM yyyy 'à' H'h'mm ", locale='fr')
            fille = 'e' if dict_baby['sexe'] == "Fille" else ''
            pronom = "elle" if dict_baby['sexe'] == "Fille" else 'il'

            st.markdown("&nbsp;")
            st.markdown(f"<center><h5> 🥳 &nbsp; <b>C'est un{fille} {dict_baby['sexe'].lower()} !</b> &nbsp; 🥳</h5></center>",
                unsafe_allow_html=True)
            st.markdown(f"""<center>Notre petit{fille} {dict_baby['prenom']} est né{fille}</br>le {jour.strip()}, </br>
            {pronom} pesait alors {int(dict_baby['poids']):1,} kg pour {int(dict_baby['taille'])} cm,</center>""",
                unsafe_allow_html=True) 

            if dict_baby['chevelure'] == 'Pas de cheveux !':
                st.markdown(f"<center>et {pronom} était chauve !</center>", unsafe_allow_html=True)
            else :
                st.markdown(f"""<center>ses cheveux sont {dict_baby['couleur_cheveux'].lower()} 
                ({dict_baby['chevelure'].lower()}).</center>""",
                    unsafe_allow_html=True)

            st.markdown("&nbsp;")
            
        ## PROFIL "GAGNANT" DU BÉBÉ
        with st.container() :
            st.markdown("#### Profil du bébé")

            with st.expander("""Par défaut, ce sont les valeurs réelles qui sont renseignées. 
            Vous pouvez les modifier pour voir comment le classement évolue..."""):
                dict_win = {
                    'birthday_day' : st.date_input(
                        "Date de naissance",
                        value=dict_baby['birthday'],
                        min_value=date(2021,12,1),
                        max_value=date(2022, 2, 28)
                    ),
                    'birthday_time' : st.time_input(
                        "Heure de naissance",
                        dict_baby['birthday'],
                        True
                    ),
                    'sexe' : st.radio(
                        "Sexe",
                        ('Garçon', 'Fille'),
                        0 if dict_baby['sexe']== 'Garçon' else 1
                    ),
                    'prenom' : st.text_input('Prénom',dict_baby['prenom']),
                    'taille': st.slider(
                        "Taille (cm)", 40, 60, dict_baby['taille']
                    ),
                    'poids': st.number_input(
                        "Poids (g)", 2000, 5000, dict_baby['poids']
                    ),
                    'chevelure': st.select_slider(
                        "Chevelure",
                        dict_cheveux['ordre_cheveux'][::-1],
                        dict_baby['chevelure']
                    ),
                    'couleur_cheveux' : st.select_slider(
                        "Couleur des cheveux",
                        dict_cheveux['ordre_couleur'][::-1],
                        dict_baby['couleur_cheveux']
                    )
                }

            dict_win['birthday'] = pd.Timestamp.combine(dict_win['birthday_day'], dict_win['birthday_time'])
            dict_win.pop('birthday_day')
            dict_win.pop('birthday_time')


        ## CLASSEMENT
        with st.container() :

            st.markdown("&nbsp;")
            st.markdown("#### Classement des participants")

            st.write("Triez selon une colonne en cliquant sur son titre.")
            with st.expander("Chaque critère rapporte jusqu'à 1 point. Voir plus de détails..." ) :
                st.markdown(f"**Sexe** : si bonne réponse ({dict_baby.sexe}), sinon 0 pt.")
                st.markdown(f"**Date de naissance** : ")
                st.markdown(f"**Poids** : ")
                st.markdown(f"**Taille** : ")
                st.markdown(f"**Cheveux** : ")
                st.markdown(f"**Prénom** : ") 

            df_results = beautify_df(calculate_scores(df, dict_win))
            st.write(df_styler(df_results))

            st.markdown("&nbsp;")
            st.markdown("##### Chercher le score d'un participant")

            participant = st.text_input("Tapez le prénom du participant à afficher...",)
            participant_selected = False

            if len(participant) > 0:

                df_filter = df_results.loc[df_results['Prénom'].str.lower().str.contains(participant.lower())]

                if len(df_filter) == 0 :
                    col0, col1, col2 = st.columns((2,5,2))
                    with col0 :
                        st.markdown("<center>⁉️</center>", unsafe_allow_html=True)
                    with col1 :
                        st.markdown("<center>Pas de résultats... Essayez avec moins de lettres ?</center>", unsafe_allow_html=True)
                    with col2 :
                        st.markdown("<center>⁉️</center>", unsafe_allow_html=True)

                else :
                    if len(df_filter) > 1 :
                        options = {}
                        for index, row in df_filter.iterrows():
                            options[index] = f'{row.Prénom} {row.Nom}'

                        st.write('&nbsp;')
                        selected = st.selectbox(
                            "Plusieurs participants ont été trouvés. Choisissez le participant à afficher :", 
                            options=options.keys(),
                            format_func= lambda x:options[x]
                            )
                        
                        df_filter = df_filter.reset_index()
                        serie_participant = pd.Series(df_filter.loc[df_filter['place'] == selected].iloc[0], dtype='str')
                        participant_selected =  True

                    else : #len(df_filter) ==1 :  
                        serie_participant = pd.Series(df_filter.reset_index(level=0).iloc[0], dtype='str')
                        participant_selected = True

                    st.write('&nbsp;')
                    scores_participant(serie_participant, len(df_results))
            
            st.write('&nbsp;')


        ## FAUX PRONOS
        with st.container():
            st.markdown("#### Faux pronostics")
            st.markdown("Réalisez de faux pronostics pour voir quel classement vous auriez pu avoir !")

            with st.expander(f"""Par défaut, ce sont les valeurs de naissance de {dict_baby['prenom']} qui sont renseignées."""):

                dict_default = dict_baby.copy()

                prenom_masc = dict_default['prenom'] if dict_default['sexe']== 'Garçon' else ""
                prenom_fem = "" if dict_default['sexe']== 'Garçon' else dict_default['prenom']

                dict_fake = {
                    'prenom' : 'Pourde',
                    'nom' : 'Fo.',
                    'birthday_day' : st.date_input(
                        "Date de naissance ",
                        value=dict_default['birthday'],
                        min_value=date(2021,12,1),
                        max_value=date(2022, 2, 28),
                    ),
                    'birthday_time' : st.time_input(
                        "Heure de naissance ",
                        dict_default['birthday'],
                        True,
                    ),
                    'sexe' : st.radio(
                        "Sexe ",
                        ('Garçon', 'Fille'),
                        0 if dict_default['sexe']== 'Garçon' else 1
                    ),
                    'prenom_masc' : st.text_input('Prénom masculin', prenom_masc),
                    'prenom_fem' : st.text_input('Prénom féminin', prenom_fem),
                    'taille': st.slider(
                        "Taille (cm) ", 40, 60, dict_default['taille']
                    ),
                    'poids': st.number_input(
                        "Poids (g) ", 2000, 5000, dict_default['poids']
                    ),
                    'longueur_cheveux': st.select_slider(
                        "Chevelure ",
                        dict_cheveux['ordre_cheveux'][::-1],
                        dict_default['chevelure']
                    ),
                    'couleur_cheveux' : st.select_slider(
                        "Couleur des cheveux ",
                        dict_cheveux['ordre_couleur'][::-1],
                        dict_default['couleur_cheveux']
                    )
                }


            dict_fake['date'] = pd.Timestamp.combine(dict_fake['birthday_day'], dict_fake['birthday_time'])
            df_fake = df.append(dict_fake, ignore_index=True)
            len_df = len(df_fake)

            fake_results = beautify_df(calculate_scores(df_fake, dict_win))

            df_fake_filter = fake_results.loc[fake_results['Prénom'] == dict_fake['prenom']]
            fake_participant = pd.Series(df_fake_filter.reset_index(level=0).iloc[0], dtype='str')

            if participant_selected :
                if fake_participant.drop(['Prénom', 'Nom', 'birthday_day', 'birthday_time', 'place']).equals(
                    other=serie_participant.drop(['Prénom', 'Nom', 'place'])) :
                    fake_participant = serie_participant.copy()
                    len_df = len(df_fake)-1

            scores_participant(fake_participant, len_df)

            classement = int(fake_participant['place'])
            st.write('&nbsp;')
            if classement > 1 :
                st.write(f"Score de la personne juste devant : {fake_results.iloc[classement-2]['Score total']} pts.")
            if classement < len_df :
                st.write(f"Score de la personne juste derrière : {fake_results.iloc[classement]['Score total']} pts.")
            

# FOOTER
st.markdown(footer,unsafe_allow_html=True)
