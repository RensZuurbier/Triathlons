import pandas as pd
import re

# Functie om de tijd te formatteren naar uren:minuten:seconden
def format_time(time_str):
    parts = time_str.split(":")
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    else:
        return time_str  # Geef de oorspronkelijke waarde terug als er iets misgaat

# Lijst van clubnamen
club_names = [
    "Triathlon- en Zwemclub Oude Veer",
    "TV Almere",
    "Unltd Attitudes",
    "Zwefilo",
    "NTB",
    "DTC",
    "Zomer Cursisten",
    "-",
    "Panthera Triathlon Team",
    "AZVD",
    "Hollandia",
    "DEM"
]

# Functie om de naam en club te scheiden en de tijd te formatteren
def process_row(row):
    # Verwijder overbodige spaties in de rij
    row = " ".join(row.split())

    # Gebruik regex om de rij in te delen in de juiste delen
    match = re.match(r'(\d+) (\d+) (.+?) (\d{2}:\d{2}) (\d{2}:\d{2}) (\d{2}:\d{2}) (\d{2}:\d{2}) (\d{2}:\d{2}) (\d{2}:\d{2})', row)

    # Als er geen overeenkomst is, retourneer None
    if not match:
        return None

    # Extracteer de individuele velden uit de regex-match
    plaats = match.group(1)
    bib = match.group(2)
    naam_club = match.group(3)
    swim = match.group(4)
    wis1 = match.group(5)
    bike = match.group(6)
    wis2 = match.group(7)
    run = match.group(8)
    eindtijd = match.group(9)

    # Scheid de naam en club op basis van de lijst van clubnamen
    for club in club_names:
        if club in naam_club:
            naam = naam_club.replace(club, "").strip()
            club = club  # Update de club naar de juiste waarde
            break
    else:
        naam = naam_club
        club = ""

    # Formatteer de eindtijd naar uren:minuten:seconden
    eindtijd_formatted = format_time(eindtijd)

    # Combineer alle velden in een lijst
    flattened_data = [plaats, bib, naam, club, swim, wis1, bike, wis2, run, eindtijd_formatted]

    return flattened_data

# Bestandsnamen
input_file = "data.csv"
output_file = "updated_data.csv"

# Lees het bestaande CSV-bestand in een lijst van rijen
with open(input_file, 'r') as f:
    lines = f.readlines()

# Verwerk elk record in de lijst van rijen
processed_data = []
for line in lines:
    processed_row = process_row(line.strip())
    if processed_row:
        processed_data.append(processed_row)

# Kolomnamen voor het nieuwe DataFrame
header = "Plaats bib Naam Club Swim Wis1 Bike Wis2 Run Eindtijd".split()

# Maak een nieuw DataFrame van de verwerkte gegevens
processed_df = pd.DataFrame(processed_data, columns=header)

# Opslaan naar CSV
processed_df.to_csv(output_file, index=False)

print(f"Verwerkte gegevens zijn opgeslagen naar '{output_file}'")
