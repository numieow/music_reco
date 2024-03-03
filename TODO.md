### Étape 1 

- Générer un csv avec la correspondance entre les instruments et les pistes audio.
  - On crée une ligne par couple (piste, instrument), et on pourra les récupérer facilement sous forme de dataframe.
  
- Première étape : on input une piste .wav au modèle, qui output une liste des instruments

- Étape finale : on fait écouter un audio au modèle, qui output en temps réel une liste d'instruments