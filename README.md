Progetto IA II — Classificazione di condizioni biomeccaniche da segnali di gait

📌 DESCRIZIONE DEL PROGETTO

Questo progetto ha l’obiettivo di classificare diverse condizioni sperimentali/cliniche a partire da segnali biomeccanici di gait, utilizzando tecniche di feature engineering, machine learning supervisionato e validazione rigorosa per soggetto.

Il lavoro si concentra non solo sulle prestazioni predittive, ma soprattutto sulla correttezza metodologica, evitando fenomeni di data leakage e garantendo la generalizzazione a soggetti non visti.


🔬 DATASET

Tipo di dati: segnali temporali di angoli articolari durante il cammino

Struttura originale: formato long (tempo × joint × soggetto × condizione)

Numero di soggetti: ~10

Numero di condizioni: 3

Dopo la fase di feature engineering:

dataset long: ~90 campioni

dataset wide: ~30 campioni (1 riga = 1 soggetto × condizione)

🧩 FEATURE ENGINEERING

Dai segnali grezzi vengono estratte feature descrittive, tra cui:

Statistiche: mean, std

Ampiezza: Range of Motion (ROM)

Intensità: Root Mean Square (RMS)

Dinamica: frequenza dominante (FFT)

Successivamente i dati vengono ristrutturati in formato wide, in cui:

ogni riga rappresenta una coppia (subject, condition)

le feature delle diverse articolazioni diventano colonne separate

Questo passaggio è cruciale per catturare le relazioni inter-articolari.

🎯 TARGET

Target: condition

Tipo di problema: classificazione supervisionata multi-classe (3 classi)

Le feature estratte sono utilizzate esclusivamente come variabili esplicative (X).

🤖 MODELLI UTILIZZATI

I seguenti modelli sono stati confrontati:

-Logistic Regression (con StandardScaler)

-Random Forest

-Ridge classifier

-Gaussian Naive Bayes

La scelta include modelli:

lineari

non lineari

generativi

per verificare la robustezza del segnale rispetto all’algoritmo.

🧪 STRATEGIA DI VALIDAZIONE

GroupKFold

La valutazione delle prestazioni è effettuata tramite:

GroupKFold cross-validation

gruppo = soggetto

Questo garantisce che:

nessun soggetto compaia sia in training che in test

le prestazioni riflettano la generalizzazione a nuovi soggetti

📊 METRICHE DI VALUTAZIONE

Accuracy

Precision / Recall / F1-score (macro e weighted)

Confusion Matrix (aggregata sui fold)

Le matrici di confusione sono salvate come immagini PNG.

🔁 TEST DI ROBUSTEZZA: REAL VS SHUFFLE

Per verificare l’assenza di leakage è stato effettuato un permutation test:

le etichette di classe (y) vengono randomizzate

i modelli sono rivalutati con la stessa CV (GroupKFold)

Risultato atteso:

Accuracy alta su dati reali

Accuracy prossima al caso su dati shuffle

Questo test conferma che le prestazioni elevate sono dovute a segnale reale, non a scorciatoie nei dati.

📈 RISULTATI PRINCIPALI

Nel formato long: accuracy ~0.55

Nel formato wide: accuracy ~0.95–0.97

Con shuffle test: accuracy ~0.25–0.35

Il forte incremento nel formato wide indica che la discriminazione tra condizioni è legata soprattutto alla coordinazione inter-articolare, più che a singole articolazioni.

🧠 CONCLUSIONE

La rappresentazione dei dati è più importante del modello

GroupKFold è essenziale per evitare leakage nei dati biomeccanici

Il formato wide consente di catturare pattern complessi anche con pochi soggetti

Le prestazioni elevate sono statisticamente e metodologicamente giustificate

▶️ ESEGUIRE IL PROGETTO

pip install -r requirements.txt


Eseguire poi i notebook in ordine:

EDA.py

Prediction_and_evaluation.py

📚 AUTORI

Progetto realizzato per il corso di Intelligenza Artificiale II.
Giorgio Ragnini, Gerardo Vittorio Marrocco, Matteo Ricci.
