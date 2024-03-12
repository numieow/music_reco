### Étape 1 

- Générer un csv avec la correspondance entre les instruments et les pistes audio.
  - On crée une ligne par couple (piste, instrument), et on pourra les récupérer facilement sous forme de dataframe.
  
- Première étape : on input une piste .wav au modèle, qui output une liste des instruments
  - Prendre 3.0 secondes de chaque extrait pour les tests (avec le bout de code commenté)

- Étape finale : on fait écouter un audio au modèle, qui output en temps réel une liste d'instruments

Raccourcis : cello (cel), clarinet (cla), flute (flu), acoustic guitar (gac), electric guitar (gel), organ (org), piano (pia), saxophone (sax), trumpet (tru), violin (vio), and human singing voice (voi)