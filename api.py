import flask
import main
import pandas as pd
import numpy as np
from flask import request, jsonify

app = flask.Flask(__name__)
app.config["DEBUG"] = True


df_niedorp = pd.read_csv('triathlons_data/niedorp.csv')

# Voorbeeld van data (dit zou normaal gesproken uit een database komen)
# triathlons = [
#     {'id': 1, 'name': 'Ironman Triathlon', 'participants': ['Alice', 'Bob', 'Charlie']},
#     {'id': 2, 'name': 'Olympic Triathlon', 'participants': ['David', 'Eve', 'Frank']},
# ]

# Route voor het ophalen van triathlons en deelnemers
@app.route('/triathlons', methods=['GET'])
def get_triathlons():
    return jsonify(df_niedorp)

# Route voor het toevoegen van een deelnemer aan een triathlon
@app.route('/triathlons/add_participant', methods=['POST'])
def add_participant():
    triathlon_id = request.args.get('triathlon_id')
    participant_name = request.args.get('name')

    # Voeg deelnemer toe aan de juiste triathlon
    for triathlon in df_niedorp:
        if triathlon['id'] == int(triathlon_id):
            triathlon['participants'].append(participant_name)
            return jsonify({'message': 'Participant added successfully'})

    return jsonify({'error': 'Triathlon not found'})

# Start de Flask-applicatie
if __name__ == '__main__':
    app.run()
