import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time

########## FUNCTIES ##########
### Zet alle lege en "DNF" waarden om naar NaT waarden ###
def convert_to_int(value):
    """
    Convert a value to int, handling NaN and "DNF" values gracefully.
    """
    try:
        if pd.isna(value) or value == 'DNF':
            return np.nan  # Return NaN for NaN and "DNF" values
        else:
            return int(value)  # Convert other values to int
    except ValueError:
        return np.nan  # Return NaN if conversion to int fails

def  convert_empty(df):
    """
    Convert all empty values (NaN, '', 'DNF') to NaT in the given DataFrame.
    """
    df_copy = df.copy()

    for col in df_copy.columns:
        df_copy[col] = df_copy[col].apply(lambda x: pd.NaT if pd.isna(x) or x == '' or x == 'DNF' else x)
    return df_copy

### Voeg een kolom "Triathlon" toe aan ieder dataframe met de naam van de Triathlon ###
def add_name(dfs):
    """
    Adds the column "Triathlon" to every dataframe with the name based on the list "namen"
    to be able to identify which triathlon it is.
    """
    namen = ["Stad van de Zon", "Langedijk", "Dirkshorn", "Nieuwe Niedorp", "Schagen"]

    for index, df in enumerate(dfs):
        # Voeg 'Triathlon' kolom toe aan het dataframe met de juiste naam
        df.insert(0, 'Triathlon', namen[index])

    return dfs

### Functie om dataframes mee te filteren. Bv. op de waarde "M" in de kolom "MV" ###
def filter_df(dfs, column, filter_value):
    """
    Filters the dataframe or dataframes based on a specified value in a column. Returns the filtered df.

    parameters:
    - dfs (single DF or list of DFs): A single dataframe or a list of dataframes to filter.
    - column (str): The column name on which to apply the filter.
    - filter_value (str): The value to filter the rows by in the specified column.
    """
    # Check of het een losse DF is
    single_df = False
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
        single_df = True

    # Voor een lijst met DFs
    filtered_dfs = []

    for df in dfs:
        if column in df.columns:
            df[column] = df[column].str.strip()
            filtered_df = df[df[column] == filter_value]
            filtered_dfs.append(filtered_df)
        else:
            filtered_dfs.append(df)

    if single_df:
        print("ER KOMT EEN LIJST UIT")
        return filtered_dfs[0]
    return filtered_dfs

### Verwijdert opgegven kolommen ###
def drop_column(dfs):
    """
    Drops the list column(s) for DFs if existing.
    """
    drop_columns = ["#Tot", "StNr", "MV", "Cat", "#Cat", "Woonplaats", "#Z", "#W1", "#F", "#NaF", "#W2", "#L"]
    filtered_dfs = []

    for df in dfs:
        df_copy = df.copy()
        for column in drop_columns:
            if column in df_copy.columns:
                df_copy.drop(columns=[column], inplace=True)
        filtered_dfs.append(df_copy)
    return filtered_dfs

### Hernoemt de kolom "#MV" (wat de eindstand weergeeft) naar "#Tot" en plaatst de kolom achteraan ###
def create_tot(dfs):
    """
    Renames the "#MV" column to "#Tot" and places the columns as last column
    If it doesn't exist the "#Tot" column will be created
    """
    new_dfs = []

    for df in dfs:
        if '#MV' in df.columns:
            df.rename(columns={'#MV': "#Tot"}, inplace=True)
            df['#Tot'] = df['#Tot'].apply(convert_to_int)
            temp = df.pop('#Tot')
            df['#Tot'] = temp.fillna(-1).astype(int)

        else:
            df['#Tot'] = df['Totaal'].rank(method='min', na_option='bottom').apply(convert_to_int).astype(int)

        new_dfs.append(df)
    return new_dfs




## Zet alle float waardes om naar integers van de opgegven kolom ###
# def float_to_int(df, columns):
#     """
#     Convert given columns in the DF from float to INT type.
#     Also temporarily replaces NaN-values to be able to convert all values
#     """
#     placeholder_value = -999999
#     for col in columns:
#         if df[col].dtype == 'float64':
#                 # Vervang NaN-waarden en 'DNF' tijdelijk met een integer placeholder
#                 df[col] = df[col].apply(lambda x: placeholder_value if pd.isna(x) or x == 'DNF' else x)
#
#                 # Converteer naar integer type met pd.Int64Dtype() om NaN-waarden te behouden
#                 df[col] = df[col].astype(pd.Int64Dtype())
#
#                 # Herstel NaN-waarden door de placeholder-waarde terug te zetten naar NaN
#                 df[col] = df[col].replace(placeholder_value, pd.NA)
#     return df



### Zet data types om naar tijden in de gegeven kolommen ###
def convert_to_time(dfs):
    """
    Convert all data types of the given columns to timedelta to be able
    to perform calculations on these columns
    """
    tijd_kolommen = ["Zwem", "Wis1", "Fiets", "NaFiets", "Wis2", "Loop", "Totaal"]
#    df_copy = df.copy()
    new_dfs = []

    def convert_value(value, column):
        if pd.isna(value) or value == "DNF":
            return pd.NaT
        try:
            # Splits de waardes op de ":" om te kijken met wat voor format we
            # te maken hebben en zo op de juiste manier om te kunnen zetten
            value = str(value)
            parts = value.split(':')
            if column != "Totaal" and column != "Wis1" and len(parts) == 3 and parts[2] == '00':
                # Behandel als MM:SS
                time = pd.to_datetime(f"{parts[0]}:{parts[1]}", format='%M:%S').time()
                return pd.Timedelta(minutes=time.minute, seconds=time.second)
            else:
                # Probeer de waarde als HH:MM:SS
                time = pd.to_datetime(value, format='%H:%M:%S').time()
                return pd.Timedelta(hours=time.hour, minutes=time.minute, seconds=time.second)
        except ValueError:
            try:
                # Probeer de waarde als MM:SS
                time = pd.to_datetime(value, format='%M:%S').time()
                return pd.Timedelta(minutes=time.minute, seconds=time.second)
            except ValueError:
                try:
                    # Probeer de waarde als seconden met decimalen
                    seconds = round(float(value))
                    return pd.Timedelta(seconds=seconds)
                except ValueError:
                    return pd.NaT

    # Loop door de lijst dataframes | En door alle kolommen per DF
    for df in dfs:
        df_copy = df.copy()
        for col in tijd_kolommen:
            if col in df_copy.columns:
                # Toepassen van de conversie functie op de kolom
                df_copy[col] = df_copy[col].apply(lambda x: convert_value(x, col))
        new_dfs.append(df_copy)

    return new_dfs

### Zorgt ervoor dat er voor elk onderdeel een ranking kolom wordt gemaakt ###
def add_ranking(dfs):
    """
    Create a ranking column for each sport element
    """
    # Maak koppels, voor elk onderdeel-kolom moet een ranking kolom komen
    onderdelen = [['Zwem', '#Z'], ['Wis1', '#W1'], ['Fiets', '#F'],  ['NaFiets', '#NaF'],
                  ['Wis2', '#W2'],  ['Loop', '#L']]
    new_dfs = []

    for df in dfs:
        df_copy = df.copy()

        for onderdeel, ranking in onderdelen:
            if ranking not in df_copy.columns:  # Controleer of de ranking kolom nog niet bestaat
                # Bepaalt de plek van de ranking kolom (+1 op basis van het onderdeel)
                index = df_copy.columns.get_loc(onderdeel) + 1

                # Maak de rankingkolom aan en initialiseer deze als NaN (Not a Number)
                df_copy.insert(index, ranking, pd.NA)

                # Voeg de ranking alleen toe aan rijen die geen NaN-waarden hebben in het onderdeel kolom
                ranked_values = df_copy.loc[~df_copy[onderdeel].isna(), onderdeel].rank(method='min', na_option='keep').astype(pd.Int64Dtype())
                df_copy.loc[~df_copy[onderdeel].isna(), ranking] = ranked_values
        new_dfs.append(df_copy)

    return new_dfs

### Dropt bestaande "NaFiets" kolommen en maakt deze opnieuw aan ###
def add_nafiets(dfs):
    """
    Drop all existing "#NaFiets" columns and recreate it based upon calculations
    of all of the elements.
    """
    # Controleer of 'NaFiets' al in de kolommen zit en verwijder deze zo nodig
    new_dfs = []

    for df in dfs:
        columns = df.columns.tolist()
        if 'NaFiets' in df.columns:
            df.drop('NaFiets', axis=1, inplace=True)

    # Maak een nieuwe kolom 'NaFiets' aan met de som van Zwem, Wis1 en Fiets
        index_fiets = columns.index('Fiets')
        df['NaFiets'] = df['Zwem'] + df['Wis1'] + df['Fiets']
        nafiets_col = df.pop('NaFiets')
        df.insert(index_fiets + 1, 'NaFiets', nafiets_col)
        new_dfs.append(df)

    return new_dfs


################################################################################
############## ZOEKT DE ATLEET OP IN DE DF EN MAAKT EEN LOSSE DF  ##############
def search_df(dfs, search_value):
    results = []

    for df in dfs:
        if 'Naam' in df.columns:
            hits = df[df['Naam'].str.contains(search_value, case=False, na=False)]
            results.append(hits)
    combined_results = pd.concat(results).drop_duplicates().reset_index(drop=True)

    if combined_results.empty:
        print(f"Geen deelnemmer gevonden met \'{search_value}\'.")
        exit()
    return combined_results

def get_user_choice(results):
    if results.empty:
        print(f"Geen resultaten gevonden voor de opgegeven zoekterm.")
        return None

    unieke_resultaten = results.drop_duplicates(subset=['Naam']).reset_index(drop=True)
    print("\n" + "Resultaten gevonden:" + "\n")
    for i, row in unieke_resultaten.iterrows():
        print(f"{i+1} {row['Naam']}")

    while True:
        try:
            selectie = int(input(f"Maak een keuze (1-{len(unieke_resultaten)}): "))
            if 1 <= selectie <= len(unieke_resultaten):
                naam = unieke_resultaten["Naam"][selectie - 1]
                print(f"{naam} is geselecteerd")
                print('\n')

                return unieke_resultaten.iloc[selectie - 1], naam
            else:
                print("Ongeldig indexnummer. Probeer het opnieuw.")
        except ValueError:
            print("Voer een geldig getal in.")

def search_and_select(dfs, search_value):
    results = search_df(dfs, search_value)

    if not results.empty:
        selected_name, naam = get_user_choice(results)
        if selected_name is not None:
            filtered_df = pd.concat([df[df['Naam'] == selected_name['Naam']] for df in dfs if 'Naam' in df.columns]).drop_duplicates().reset_index(drop=True)
            return filtered_df, naam
    return None
############## ZOEKT DE ATLEET OP IN DE DF EN MAAKT EEN LOSSE DF  ##############
#################################################################################



### maakt een Bar plot voor de opgegeven kolom van het DF. Geef weer hoe ###
### atleten met verschillende onderdelen hebben gepresteerd tov elkaar   ###
def plot_bar(df, column, title, top_nr=None, dirkshorners=None):
    """
    Genereert een plot van de geveven kolom (het onderdeel). Optioneel is om
    aan te geven welke selectie gegevens wilt inzien. Bv. top 3, 5 of top 10.

    Parameteres:
    - df: DataFrame
    - column: Welke kolom (Onderdeel, bv "Fiets")
    - top_nr: X aantal deelnemers, standaard allemaal
    - dirkshorners: filter op alleen DH'ers, standaard niet
    """
    # Selecteer alleen de rijen met deelnemers uit de lijst dirkshorners
    if dirkshorners:
        df = df[df['Naam'].isin(dirkshorners)]
    else:
        df = df

    # Hier kan er optioneel gekozen worden voor een selectie van de deelnemers
    if top_nr is None:
        top_nr = len(df[column])
    df = df.nsmallest(top_nr, 'Totaal')

    # Filtert alle rijden met lege velden en DNF'ers uit het dataframe
    df = df.dropna(subset=[column])
    df = df[df['#Tot'] != 'DNF']

    # Omzetten van Timedelta naar seconden (Wis1 & Wis2)
    if column == 'Wis1' or column == 'Wis2':
        df['tijd'] = df[column].dt.total_seconds()

        plt.bar(df['Naam'], df['tijd'], color=['blue', 'green'])
        plt.xlabel('Naam')
        plt.ylabel('Tijd (seconden)')
        plt.title(title)
        plt.xticks(rotation=45)

        # output_directory = 'plots'
        # filename = (f'bar_{title}_DH')
        # full_path = f'{output_directory}/{filename}'
        # plt.savefig(full_path)

        plt.show()

    # Omzetten van Timedelta naar minuten
    else:
        df['tijd'] = df[column].dt.total_seconds() / 60

        plt.bar(df['Naam'], df['tijd'], color=['blue', 'green'])
        plt.xlabel('Naam')
        plt.ylabel('Tijd (minuten)')
        plt.title(title)
        plt.xticks(rotation=45)


        # output_directory = 'plots'
        # filename = (f'bar_{title}_DH')
        # full_path = f'{output_directory}/{filename}'
        # plt.savefig(full_path)

        plt.show()

### Zet TimeDelta waarden om naar secondes (integers) ###
def convert_timedelta_to_seconds(df):
    """
    Converteer alle kolommen van het type 'timedelta64' naar seconden (integer).
    Negeer NaN-waarden en niet-converteerbare waarden zoals 'DNF'. In het leven
    geroepen zodat er met de data een plot gemaakt kan worden
    """
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_timedelta64_dtype(df[col]):
            df[col] = df[col].dt.total_seconds()  # Converteer timedelta naar seconden
    return df

### Maakt een line chart per atleet voor de progressie per onderdeel ###
def plot_line(df, atleten=None):
    """
    Plot de voortgang van atleten per onderdeel middels een lijn.

    Parameters:
    - df: DataFrame met onderdelen en tijden in seconden.
    - atleten: Optionele lijst met namen van atleten om te plotten. Als None, dan alle atleten.
    """
    if atleten is not None:
        df = df[df['Naam'].isin(atleten)]


    # Converteer timedelta waardes naar seconden voor berkeningen
    data = convert_timedelta_to_seconds(df)

    # Lijst van onderdelen waarvoor we de progressie willen plotten
    onderdelen = ['Zwem', 'Wis1', 'Fiets', 'Wis2', 'Loop']

    # Maak een DataFrame voor het plotten van de cumulatieve tijden per atleet
    plot_data = pd.DataFrame(index=['Start'] + onderdelen)
    for _, row in data.iterrows():
        atleet = row['Naam']
        cumulative_times = [0]  # Start met een beginpunt van 0 seconden
        for col in onderdelen:
            value = row[col]
            if pd.notna(value) and value != 'DNF':
                cumulative_times.append(cumulative_times[-1] + value)
            else:
                cumulative_times.append(np.nan)  # Negeer DNF of lege waarden
        plot_data[atleet] = cumulative_times

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 8))

    for atleet in plot_data.columns:
        # Verwijder NaN-waarden voor het plotten
        data_to_plot = plot_data[atleet].dropna()
        x_values = plot_data.index[:len(data_to_plot)]  # X-waarden: indices van 'plot_data'
        y_values = data_to_plot.values  # Y-waarden: cumulatieve tijden per atleet
        ax1.plot(x_values, y_values, marker='o', label=atleet)

    # Bereken de maximale tijd in minuten
    tot_minutes = int(np.ceil(data['Totaal'].max() / 60)) + 5

    # Bereken de y-ticks op basis van de maximale tijd
    y_ticks = np.arange(0, tot_minutes * 60 + 1, 300)  # Verhoog per 5 minuten (300 seconden)

    # Stel de y-ticks en y-as labels in
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels([f"{int(y//60)}:{int(y%60):02d}" for y in y_ticks])
    ax1.set_ylim(0, tot_minutes * 60)
    ax1.set_ylabel('Tijd (minuten)')

    # Voeg een tweede y-as toe aan de rechterkant
    ax2 = ax1.twinx()
    ax2.set_yticks(ax1.get_yticks())  # Kopieer de y-ticks van ax1 naar ax2
    ax2.set_yticklabels([f"{int(y//60)}:{int(y%60):02d}" for y in y_ticks])  # Dezelfde labels in minuten
    ax2.set_ylim(ax1.get_ylim())  # Zorg dat de y-as limieten hetzelfde zijn
    ax2.set_ylabel('Tijd (minuten)')

    # Verplaats de legenda naar rechtsonder
    ax1.legend(loc='lower right')

    plt.xlabel('Onderdelen')
    plt.title('Progressie per Onderdeel')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    # output_directory = 'plots'
    # filename = "line"
    # full_path = f'{output_directory}/{filename}'
    # plt.savefig(full_path)

    plt.show()

### Creeert een stacked plot om zo de tijden per onderdeel per atleet weer te geven ###
def plot_stacked_bar(df, title, dirkshorners=None):
    """
    Genereert een stacked bar plot per atleet voor ieder onderdeel. Elke bar(kleur)
    toont de tijd per onderdeel.

    Parameters:
    - df: DataFrame
    - title: Geeft een titel voor de plot (stringwaarde)
    - dirkshorns: filtert atleten op lijst van Dirskhorners *optioneel
    """
    # Filter op dirkshorners als dirkshorners is meegegeven als paramter
    if dirkshorners:
        df = df[df['Naam'].isin(dirkshorners)]

    # Filtert alle rijen met lege velden en DNF'ers uit de data
    df = df.dropna(subset=['Zwem', 'Wis1', 'Fiets', 'Wis2', 'Loop'])
    df = df[df['#Tot'] != 'DNF']

    # Omzetten van Timedelta naar minuten
    df['Zwem'] = df['Zwem'].dt.total_seconds() / 60
    df['Wis1'] = df['Wis1'].dt.total_seconds() / 60
    df['Fiets'] = df['Fiets'].dt.total_seconds() / 60
    df['Wis2'] = df['Wis2'].dt.total_seconds() / 60
    df['Loop'] = df['Loop'].dt.total_seconds() / 60

    # Bereken de totale tijd per atleet
    df['Totaal'] = df['Zwem'] + df['Wis1'] + df['Fiets'] + df['Wis2'] + df['Loop']

    # Sorteer de dataframe op totale tijd, snelste bovenaan
    df = df.sort_values('Totaal', ascending=False)

    # Maak een kolom voor de labels met ranking en naam
    df['Rank_Naam'] = df['#Tot'].astype(str) + ' - ' + df['Naam']

    # Instellen van de kleuren per onderdeel
    colors = {
        'Zwem': 'paleturquoise',
        'Wis1': 'lightgray',
        'Fiets': 'yellowgreen',
        'Wis2': 'lightgray',
        'Loop': 'tomato'
    }

    # Maak de stacked bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    rank_naam = df['Rank_Naam']
    zwem = df['Zwem']
    wis1 = df['Wis1']
    fiets = df['Fiets']
    wis2 = df['Wis2']
    loop = df['Loop']

    bar_width = 1.0
    edge_color = 'black'

    ax.barh(rank_naam, zwem, color=colors['Zwem'], edgecolor=edge_color, label='Zwem', height=bar_width)
    ax.barh(rank_naam, wis1, left=zwem, color=colors['Wis1'], edgecolor=edge_color, label='Wis1', height=bar_width)
    ax.barh(rank_naam, fiets, left=zwem + wis1, color=colors['Fiets'], edgecolor=edge_color, label='Fiets', height=bar_width)
    ax.barh(rank_naam, wis2, left=zwem + wis1 + fiets, color=colors['Wis2'], edgecolor=edge_color, label='Wis2', height=bar_width)
    ax.barh(rank_naam, loop, left=zwem + wis1 + fiets + wis2, color=colors['Loop'], edgecolor=edge_color, label='Loop', height=bar_width)

    ax.set_xlabel('Tijd (minuten)')
    ax.set_title(title)
    ax.legend()

    # Sla de plots op
    # output_directory = 'plots'
    # filename = (f'stacked_{title}dirkshorners')
    # full_path = f'{output_directory}/{filename}'
    # plt.savefig(full_path)

    plt.show()



def convert_to_seconds(time_str):
    """Converteer een tijdsstring naar seconden."""
    t = pd.to_timedelta(time_str)
    return t.total_seconds()


def combined_df(dfs):
    """
    Checks which triathlons both atletes have in comen, and combine the results
    of both given atletes into 1 DF for measuring
    """
    # Controleer de gemeenschappelijke triathlons
    if len(dfs) < 2 or len(dfs) > 4:
        raise ValueError("Geef minimaal 2 en maximaal 4 atleten op voor een radar diagram.")

    # Zoek de gemeenschappelijke triathlons
    gemeenschappelijk = dfs[0][['Triathlon']]
    for df in dfs:
        gemeenschappelijk = pd.merge(gemeenschappelijk, df[['Triathlon']], on='Triathlon')

    gefilterde_dfs = [df[df['Triathlon'].isin(gemeenschappelijk['Triathlon'])].reset_index(drop=True) for df in dfs]
    combined_df = pd.concat(gefilterde_dfs, ignore_index=True)

    return combined_df


def visualize_differences(df):
    """
    Toont de verschillen tussen atleten in verschillende triathlons.

    Parameters:
    - df: Gecombineerde DataFrame met prestaties van de atleten
    """
    atleten = df['Naam'].unique()

    if len(atleten) < 2 or len(atleten) > 4:
        raise ValueError(f"Geef minimaal 2 en maximaal 4 atleten op voor de radar diagram vergelijking, {len(atleten)} gegeven")

    dfs = [df[df['Naam'] == atleet].copy() for atleet in atleten]

    # Converteer tijden naar seconden voor gemakkelijke berekening
    onderdelen = ['Zwem', 'Fiets', 'Loop', 'Totaal']
    for df_atleet in dfs:
        for onderdeel in onderdelen:
            df_atleet[f'{onderdeel}_sec'] = df_atleet[onderdeel].apply(convert_to_seconds)

    ### Radar- of spindiagram per triathlon vergelijking tussen 2 atleten ###
    for triathlon in df['Triathlon'].unique():
        stats = []
        for df_atleet in dfs:
            df_temp = df_atleet[df_atleet['Triathlon'] == triathlon]
            stats.append([df_temp[f'{onderdeel}_sec'].values[0] for onderdeel in onderdelen])

        # Voor de radarplot
        angles = np.linspace(0, 2 * np.pi, len(onderdelen), endpoint=False).tolist()
        stats += [stats[0]]  # Herhaal het eerste element om de plot te sluiten

        # Plot instellingen
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        colors = ['blue', 'red', 'green', 'purple'][:len(atleten)]  # Kleuren voor de atleten

        # Plot elk atleet op de radarplot
        for i, (stat, color) in enumerate(zip(stats, colors)):
            ax.fill(angles, stat, color=color, alpha=0.25)
            ax.plot(angles, stat, color=color, linewidth=2, label=atleten[i])

        ax.set_yticklabels([])
        ax.set_xticks(angles)
        ax.set_xticklabels(onderdelen)
        ax.set_title(f'Vergelijking van Atleten - {triathlon}')
        ax.legend()

        # Sla de plots op
        # output_directory = 'plots'
        # filename = (f'Rentah vs Mitta {triathlon}')
        # full_path = f'{output_directory}/{filename}'
        # plt.savefig(full_path)

        plt.show()
    ### Radar- of spindiagram per triathlon vergelijking tussen 2 atleten ###









############# OLD PART OF visualize_differences #############
    # Bereken de tijdsverschillen
    # verschillen = []
    # for triathlon in gemeenschappelijk['Triathlon']:
    #     data1 = df_atleet1[df_atleet1['Triathlon'] == triathlon]
    #     data2 = df_atleet2[df_atleet2['Triathlon'] == triathlon]
    #     verschil = {
    #         'Triathlon': triathlon,
    #         'Zwem': data1['Zwem_sec'].values[0] - data2['Zwem_sec'].values[0],
    #         'Fiets': data1['Fiets_sec'].values[0] - data2['Fiets_sec'].values[0],
    #         'Loop': data1['Loop_sec'].values[0] - data2['Loop_sec'].values[0],
    #         'Totaal': data1['Totaal_sec'].values[0] - data2['Totaal_sec'].values[0]
    #     }
    #     verschillen.append(verschil)
    #
    # verschil_df = pd.DataFrame(verschillen)
    #
    # # Staafdiagram voor tijdsverschillen per onderdeel
    # fig, axs = plt.subplots(len(onderdelen), 1, figsize=(14, 8 * len(onderdelen)), sharex=True)
    #
    # for i, onderdeel in enumerate(onderdelen):
    #     axs[i].bar(verschil_df['Triathlon'], verschil_df[onderdeel], color='blue', alpha=0.7)
    #     axs[i].axhline(0, color='black', linewidth=0.8)
    #     axs[i].set_ylabel(f'Verschil in {onderdeel} (seconden)')
    #     axs[i].set_title(f'Verschil in {onderdeel} tussen {atleet1} en {atleet2}')
    #
    # plt.xlabel('Triathlon')
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.show()




# def triathlon_keuze(triathlons):
#     #triathlons =[["1: Stad van de Zon", m_hhw], ["2: Langedijk", m_ld], ["3: Dirkshorn", m_dh], ["4: Nieuwe Niedorp", m_nn]]
#     print("Geef de atleten op voor de vergelijking (Volledige Naam)")
#     atleet1 = input("Atleet 1: ")
#     atleet2 = input("Atleet 2: ")
#
#     print("\n", "Er wordt een visualisatie van " + atleet1 + " en", atleet2, "gegenereerd")
#     print("\n", "Voor welke Triathlon wil je de vergelijking maken?")
#
#     for triathlon, df in triathlons:
#         print(triathlon)
#
#     print("\n")
#     keuze = int(input("Triathlon: "))
#
#     if keuze < 1 or keuze > len(triathlons):
#         print("Ongeldige Triathlon keuze.")
#         return None
#
#     gekozen_triathlon = triathlons[keuze - 1]
#     gekozen_triathlon, gekozen_df = gekozen_triathlon
#     print(gekozen_df.head())
#
#     print("\nKeuze is:", gekozen_triathlon)
#     return atleet1, atleet2, gekozen_df
