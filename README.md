# Forza del calcestruzzo

Lo scopo del progetto è di prevedere la forza del composto (strength) sulla base delle variabili fornite dal cliente e
di analizzarne la loro importanza.

# Struttura della repository

- `main.py` contiene tutti gli steps necessari per predire la forza del composto ed analizzarne le variabili significative
tramite il miglior modello in termini di rmse selezionato sul test set. 
- `src` racchiude il codice dei singoli steps utilizzati in main, precisamente: eda, data_processing, modelling.
- `requirements.txt` incorpora tutti i requisiti per installare il progetto su altra macchina.
- `.gitignore` ignora file locali.
- `README.md` illustra scopo e struttura della repository.


 * [src](./src)
   * [eda.py](./src/eda.py)
   * [modelling.py](./src/modelling.py)
   * [data_processing.py](./src/data_processing.py)
 * [requirements.txt](./requirements.txt) 
 * [.gitignore](./src/modelling.py)
 * [README.md](./README.md)