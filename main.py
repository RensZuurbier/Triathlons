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
#dataframes =[df_hhw, df_langedijk, df_dirkshorn, df_niedorp, df_schagen]
dataframes = filter_df(dataframes, "MV", "M")

# 3) Drop onnodige kolommen en ranking kolommen
dataframes = drop_column(dataframes)

# 4) Hernoemd "#MV" kolommen naar "#Tot" of Maakt nieuwe "#Tot" kolom aan
dataframes = create_tot(dataframes)

df_hhw, df_langedijk, df_dirkshorn, df_niedorp, df_schagen = dataframes

print("TYPE FIETS VOOR CONVERSIE")
print(df_dirkshorn['Fiets'].dtype)
print(df_dirkshorn.head())
print("\n")


# print(m_dh.dtypes)
######df_dirkshorn = convert_to_time(df_dirkshorn, "Zwem", "Wis1", "Fiets", "NaFiets", "Wis2", "Loop", "Totaal")

#m_dh = convert_to_time(m_dh, tijd_kolommen)
# print(m_dh.dtypes)

# m_ld = convert_to_time(m_ld, "Zwem", "Wis1", "Fiets", "NaFiets", "Wis2", "Loop", "Totaal")
# m_nn = convert_to_time(m_nn, "Zwem", "Wis1", "Fiets", "NaFiets", "Wis2", "Loop", "Totaal")
# m_hhw = convert_to_time(m_hhw, "Zwem", "Wis1", "Fiets", "NaFiets", "Wis2", "Loop", "Totaal")
# m_sch = convert_to_time(m_sch, "Zwem", "Wis1", "Fiets", "NaFiets", "Wis2", "Loop", "Totaal")

dataframes = convert_to_time(dataframes)

print("TYPE FIETS NA CONVERSIE")
print(df_dirkshorn['Fiets'].dtype)
print(df_dirkshorn.head())
print("\n")


print("DIT IS HET EINDE")
print("###################################")

exit()

# Maak opnieuw de "NaFiets" kolom aan door de tijden op te tellen
dataframes = add_nafiets(dataframes)

df_hhw, df_langedijk, df_dirkshorn, df_niedorp, df_schagen = dataframes

print(df_dirkshorn['NaFiets'].dtype)

# for df in dataframes:
#     print(df.head())


tijd_kolommen = ["Zwem", "Wis1", "Fiets", "NaFiets", "Wis2", "Loop", "Totaal"]
# Convert alle time tabellen naar TimeDelta
# m_dh = convert_to_time(m_dh, tijd_kolommen)
# m_ld = convert_to_time(m_ld, tijd_kolommen)
# m_nn = convert_to_time(m_nn, tijd_kolommen)
# m_hhw = convert_to_time(m_hhw, tijd_kolommen)
# m_sch = convert_to_time(m_sch, tijd_kolommen)







# Voeg de ranking kolommen per onderdeel toe
m_dh = add_ranking(m_dh)
m_ld = add_ranking(m_ld)
m_nn = add_ranking(m_nn)
m_hhw = add_ranking(m_hhw)
m_sch = add_ranking(m_sch)

triathlons =[m_hhw, m_ld, m_dh, m_nn, m_sch]
#triathlons =[["1: Stad van de Zon", m_hhw], ["2: Langedijk", m_ld], ["3: Dirkshorn", m_dh], ["4: Nieuwe Niedorp", m_nn]]

# print("\n" +"Resultaten Langedijk")
# print(m_ld.head())
#
# print("\n" + "Resultaten Dirkshorn")
# print(m_dh.head())
#
# print("\n" + "Resultaten Niedorp")
# print(m_nn.head())
#
# print("\n" + "Resultaten HHW")
# print(m_hhw.head())

#print(m_ld.loc[m_ld['Naam'].isin(dirkshorners)])




# plot_stacked_bar(m_hhw, "Stad v d Zon", dirkshorners)
# plot_stacked_bar(m_ld, "Langedijk", dirkshorners)
# plot_stacked_bar(m_dh, "Dirkshorn", dirkshorners)
# plot_stacked_bar(m_nn, "Nieuwe Niedorp")
plot_stacked_bar(m_sch, "Schagen", dirkshorners)


# Maakt DFs van enkel Rens & Mitchell
rens = filter_df(triathlons, 'Naam', 'Rens Zuurbier')
mitta = filter_df(triathlons, "Naam", 'Mitchell Tijsen')

# Controleer welke Triathlons beide atleten hebben gedaan
gemeenschappelijk = pd.merge(rens['Triathlon'], mitta['Triathlon'], on='Triathlon')

# Voeg data van Rens & Mitchell samen in 1 DF
combined_df = pd.concat([
rens[rens['Triathlon'].isin(gemeenschappelijk['Triathlon'])].reset_index(drop=True),
mitta[mitta['Triathlon'].isin(gemeenschappelijk['Triathlon'])].reset_index(drop=True)
], ignore_index=True)
#
#
visualize_differences(combined_df, "Rens Zuurbier", "Mitchell Tijsen")
#
#plot_line(m_hhw, dirkshorners)
#plot_line(m_ld, dirkshorners)
#plot_line(m_dh, dirkshorners)
#plot_line(m_nn, dirkshorners)
plot_line(m_sch, dirkshorners)
# #
# plot_bar(m_nn, "Zwem", "Zwem Niedorp", 27, dirkshorners)
# plot_bar(m_nn, "Wis2", "Wis2 Niedorp", 27, dirkshorners)
# plot_bar(m_nn, "Fiets", "Fiets Niedorp", 27, dirkshorners)
# plot_bar(m_nn, "Wis1", "Wis1 Niedorp", 27, dirkshorners)
# plot_bar(m_nn, "Loop", "Loop Niedorp", 27, dirkshorners)
plot_bar(m_sch, "Loop", "Schagen", 27, dirkshorners)


#     # Controleer welke Triathlons beide atleten gedaan hebben om te kunnen vergelijken
#     gemeenschappelijk = pd.merge(df_atleet1['Triathlon'], df_atleet2['Triathlon'], on='Triathlon')
#     print(gemeenschappelijk)
#
#     # Voeg de dataframes van de atleten samen
#     combined_df = pd.concat([
#     df_atleet1[df_atleet1['Triathlon'].isin(gemeenschappelijk['Triathlon'])].reset_index(drop=True),
#     df_atleet2[df_atleet2['Triathlon'].isin(gemeenschappelijk['Triathlon'])].reset_index(drop=True)
# ], ignore_index=True)


#triathlon_keuze(triathlons)
# atleet1, atleet2, gekozen_df = triathlon_keuze(triathlons)
# if gekozen_df is not None:
#     print("\nHead van het gekozen DataFrame:")
#     print(gekozen_df.head())
# else:
#     print("Er is iets misgegaan bij het kiezen van de triathlon.")
