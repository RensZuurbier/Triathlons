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

########## PROGRAM ##########
##### ZORG DAT ALLE DATAFRAMES HETZELFDE ZIJN #####
# Omdat er voor de triathlon HHW (Stad v d Zon) de data al gefilterd is op mannen (M)
# wordt dit toegepast op ieder dataframe

# HHW heeft ook als enige een andere naamgeving. Deze wordt hier herstelt naar de massa
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

# 5) Convert alle kolommen met tijd om naar type TimeDelta
dataframes = convert_to_time(dataframes)

# 6) Maak opnieuw de "NaFiets" kolom aan door de tijden op te tellen
dataframes = add_nafiets(dataframes)

# 7) Maakt de ranking  kolommen voor ieder DF opnieuw aan
dataframes = add_ranking(dataframes)


# Voegt namen van de Triathlons toe in een extra kolom
for i , df in enumerate(dataframes):
    triathlon = ['Stad van de Zon', 'Langedijk', 'Dirkshorn', 'Niedorp', 'Schagen'][i]
    df['Triathlon'] = triathlon

# Combineert de dataframes tot 1 grote dataframe
combined_df = pd.concat(dataframes, ignore_index=True)

# Exporteer naar .csv bestand
combined_df.to_csv('triathlons_combined.csv', index=False)

###### Check hoe de dataframes er stuk voor stuk uit zien. Moet dezelfde opzet zijn ######
for df in dataframes:
    print(df.head())


# print("\n" + "Resultaten HHW")
# print(df_hhw.head())
#
# print("\n" +"Resultaten Langedijk")
# print(df_langedijk.head())
#
# print("\n" + "Resultaten Dirkshorn")
# print(df_dirkshorn.head())
#
# print("\n" + "Resultaten Niedorp")
# print(df_niedorp.head())
#
# print("\n" + "Resultaten Schagen")
# print(df_schagen.head())
###### Check hoe de dataframes er stuk voor stuk uit zien. Moet dezelfde opzet zijn ######


## Laat de gebruiker atleet 1 en atleet 2 uit de beschikbare data selecteren voor vergelijking ##
# atleet1 = input("Selecteer atleet 1: ")
# df_atleet1, atleet1 = search_and_select(dataframes, atleet1)
#
# atleet2 = input("\n" "selecteer atleet 2: ")
# df_atleet2, atleet2 = search_and_select(dataframes, atleet2)
#
# atleet3 = input("\n" "selecteer atleet 3: ")
# df_atleet3, atleet3 = search_and_select(dataframes, atleet3)
#
# atleet4 = input("\n" "selecteer atleet 4: ")
# df_atleet4, atleet4 = search_and_select(dataframes, atleet4)


# atl_dfs = [df_atleet1, df_atleet2, df_atleet3, df_atleet4]

# Maakt een nieuwe DF van de resultaten van beide atleten op de gemeenschappelijke DFs
# combined_df = combined_df(atl_dfs)


######### PLOT RADAR DIAGRAM TUSSEN 2 ATLETEN ############
#visualize_differences(combined_df)

exit()

#### DH TRIATHLONNERS ####
# dirkshorners = ['Rens Zuurbier', 'Mitchell Tijsen', 'Thierry Spaans', 'Martin Bloothoofd',
#                'Co van Bueren']

########################### STACKED PLOTS ###########################
# plot_stacked_bar(m_hhw, "Stad v d Zon", dirkshorners)
# plot_stacked_bar(m_ld, "Langedijk", dirkshorners)
# plot_stacked_bar(m_dh, "Dirkshorn", dirkshorners)
# plot_stacked_bar(m_nn, "Nieuwe Niedorp")
# plot_stacked_bar(m_schagen, "Schagen", dirkshorners)
########################### BAR PLOTS ###########################


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
