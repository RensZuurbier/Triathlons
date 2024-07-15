import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time
from modules import *


# Get dataframes of the triathlons
df_hhw = pd.read_csv('triathlons_data/hhw.csv')
df_langedijk = pd.read_csv('triathlons_data/langedijk.csv')
df_dirkshorn = pd.read_csv('triathlons_data/dirkshorn.csv')
df_niedorp = pd.read_csv('triathlons_data/niedorp.csv')
df_schagen = pd.read_csv('triathlons_data/schagen.csv')


#### DH TRIATHLONNERS ####
dirkshorners = ['Rens Zuurbier', 'Mitchell Tijsen', 'Thierry Spaans', 'Martin Bloothoofd',
                'Co van Bueren']

########## CODE ##########
##### ZORG DAT ALLE DATAFRAMES HETZELFDE ZIJN #####
# Omdat er voor de triathlon HHW (Stad v d Zon) de data al gefilterd
# is op mannen (M) moet dit voor iedere dataframe gebeuren

# HERNOEMEN KOLOMMEN HHW
df_hhw = df_hhw.rename(columns={
    'Plaats': '#Tot',
    'bib': 'StNr',
    'Swim': 'Zwem',
    'Bike': 'Fiets',
    'Run': 'Loop',
    'Eindtijd': 'Totaal'
    })

# Maak een lijst met alle dataframes van de triathlons_data
dataframes =[df_hhw, df_langedijk, df_dirkshorn, df_niedorp, df_schagen]

# 1) Voeg de kolom "Triathlon" toe voor de naamgeving per DF
dataframes = add_name(dataframes)

# 2) Filter op Mannen categorie (M)
dataframes = filter_df(dataframes, "MV", "M")

# 3) Drop onnodige kolommen en ranking kolommen om de ranking later weer aan te maken
dataframes = drop_column(dataframes)

# 4) Hernoemt "#MV" kolommen naar "#Tot" of maakt nieuwe "#Tot" kolom aan
dataframes = create_tot(dataframes)

df_hhw, df_langedijk, df_dirkshorn, df_niedorp, df_schagen = dataframes

# 5) Convert alle kolommen met tijd om naar type TimeDelta
dataframes = convert_to_time(dataframes)

# Maak opnieuw de "NaFiets" kolom aan door de tijden op te tellen
dataframes = add_nafiets(dataframes)

# Maakt de ranking  kolommen voor ieder DF opnieuw aan
dataframes = add_ranking(dataframes)

m_hhw, m_langedijk, m_dirkshorn, m_niedorp, m_schagen = dataframes

###### Check hoe de dataframes er stuk voor stuk uit zien. Moet dezelfde opzet zijn ######
# print("\n" + "Resultaten HHW")
# print(m_hhw.head())
#
# print("\n" +"Resultaten Langedijk")
# print(m_langedijk.head())
#
# print("\n" + "Resultaten Dirkshorn")
# print(m_dirkshorn.head())
#
# print("\n" + "Resultaten Niedorp")
# print(m_niedorp.head())
#
# print("\n" + "Resultaten Schagen")
# print(m_schagen.head())
###### Check hoe de dataframes er stuk voor stuk uit zien. Moet dezelfde opzet zijn ######


########################### BAR PLOTS ###########################
# plot_stacked_bar(m_hhw, "Stad v d Zon", dirkshorners)
# plot_stacked_bar(m_ld, "Langedijk", dirkshorners)
# plot_stacked_bar(m_dh, "Dirkshorn", dirkshorners)
# plot_stacked_bar(m_nn, "Nieuwe Niedorp")
# plot_stacked_bar(m_schagen, "Schagen", dirkshorners)
########################### BAR PLOTS ###########################




atleet1 = input("Selecteer atleet 1: ")
df_atleet1, atleet1 = search_and_select(dataframes, atleet1)

# print(f'Dit zijn de resultaten van {atleet1}:')
# print(df_atleet1)

atleet2 = input("\n" "selecteer atleet 2: ")
df_atleet2, atleet2 = search_and_select(dataframes, atleet2)
#print(atleet2)

combined_df = combined_df(df_atleet1, df_atleet2)

# # Controleer welke Triathlons beide atleten hebben gedaan
# gemeenschappelijk = pd.merge(df_atleet1['Triathlon'], df_atleet2['Triathlon'], on='Triathlon')
#
# # print("Gemeenschappelijke Triathlons:")
# # print(gemeenschappelijk)
#
# # Voeg de datarames van atleet1 & atleet2 samen in 1 DF
# combined_df = pd.concat([
# df_atleet1[df_atleet1['Triathlon'].isin(gemeenschappelijk['Triathlon'])].reset_index(drop=True),
# df_atleet2[df_atleet2['Triathlon'].isin(gemeenschappelijk['Triathlon'])].reset_index(drop=True)
# ], ignore_index=True)



visualize_differences(combined_df, atleet1, atleet2)

exit()

########################### lINE PLOTS ###########################
#plot_line(m_hhw, dirkshorners)
#plot_line(m_ld, dirkshorners)
#plot_line(m_dh, dirkshorners)
#plot_line(m_nn, dirkshorners)
#plot_line(m_sch, dirkshorners)
########################### lINE PLOTS ###########################


########################### BAR PLOTS ###########################
# plot_bar(m_nn, "Zwem", "Zwem Niedorp", 27, dirkshorners)
# plot_bar(m_nn, "Wis2", "Wis2 Niedorp", 27, dirkshorners)
# plot_bar(m_nn, "Fiets", "Fiets Niedorp", 27, dirkshorners)
# plot_bar(m_nn, "Wis1", "Wis1 Niedorp", 27, dirkshorners)
# plot_bar(m_nn, "Loop", "Loop Niedorp", 27, dirkshorners)
#plot_bar(m_sch, "Loop", "Schagen", 27, dirkshorners)
########################### BAR PLOTS ###########################




#triathlon_keuze(triathlons)
# atleet1, atleet2, gekozen_df = triathlon_keuze(triathlons)
# if gekozen_df is not None:
#     print("\nHead van het gekozen DataFrame:")
#     print(gekozen_df.head())
# else:
#     print("Er is iets misgegaan bij het kiezen van de triathlon.")
