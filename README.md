# learning-python-ai
Repository del corso di Python per l'intelligenza artificiale

Ecco un esempio di come utilizzare un LLM (Large Language Model) in Google Colab, utilizzando la libreria `transformers` di Hugging Face.

**Scenario:** Utilizziamo un LLM per generare testo a partire da un prompt.

**Passaggio 1: Apri Google Colab**

1.  Vai su colab.research.google.com.
2.  Accedi con il tuo account Google.
3.  Crea un nuovo notebook ("Nuovo notebook").

**Passaggio 2: Installa le librerie necessarie**

In una cella di codice, inserisci e esegui il seguente comando per installare la libreria `transformers`:

```python
!pip install transformers
```

**Passaggio 3: Importa le librerie e carica il modello**

Importiamo le librerie necessarie e carichiamo un modello LLM pre-addestrato. In questo esempio, utilizzeremo il modello "gpt2":

```python
from transformers import pipeline

generator = pipeline('text-generation', model='gpt2')
```

Questo codice crea un oggetto `pipeline` che può essere utilizzato per generare testo.

**Passaggio 4: Genera testo**

Ora, utilizziamo il modello per generare testo a partire da un prompt:

```python
prompt = "Scrivi una breve storia su un gatto che parla."
generated_text = generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
print(generated_text)
```

Questo codice genera un testo di massimo 100 token a partire dal prompt specificato. L'argomento `num_return_sequences=1` indica che vogliamo generare una sola sequenza di testo.

**Spiegazione:**

* `pipeline('text-generation', model='gpt2')`: Crea una pipeline per la generazione di testo utilizzando il modello GPT-2.
* `generator(prompt, max_length=100, num_return_sequences=1)`: Genera testo a partire dal prompt, limitando la lunghezza a 100 token e restituendo una sola sequenza.
* `[0]['generated_text']`: Estrae il testo generato dalla pipeline.

**Variazioni:**

* Puoi provare diversi modelli LLM cambiando il valore dell'argomento `model`. Hugging Face offre una vasta gamma di modelli pre-addestrati.
* Puoi modificare il valore dell'argomento `max_length` per controllare la lunghezza del testo generato.
* Puoi variare il prompt per ottenere risultati differenti.

**Considerazioni:**

* I modelli LLM possono richiedere risorse di calcolo significative. Colab offre GPU gratuite, che possono accelerare il processo di generazione del testo.
* La qualità del testo generato dipende dal modello utilizzato e dal prompt fornito.
* Alcuni modelli possono avere delle restrizioni d'uso, informarsi sempre sulle licenze dei vari modelli.

Ed ecco il codice completo che puoi copiare e incollare in un notebook di Google Colab:

```python
# Passaggio 1: Installa le librerie necessarie
!pip install transformers

# Passaggio 2: Importa le librerie e carica il modello
from transformers import pipeline

generator = pipeline('text-generation', model='gpt2')

# Passaggio 3: Genera testo
prompt = "Scrivi una breve storia su un gatto che parla."
generated_text = generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']

# Passaggio 4: Stampa il testo generato
print(generated_text)

# Passaggio 5: Variazioni (opzionali)
# Prova con un modello diverso
# generator = pipeline('text-generation', model='distilgpt2')

# Modifica la lunghezza del testo generato
# generated_text = generator(prompt, max_length=150, num_return_sequences=1)[0]['generated_text']

# Modifica il prompt
# prompt = "Racconta una barzelletta sui pinguini."

# Rigenera il testo con le modifiche
# generated_text = generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
# print(generated_text)
```

**Istruzioni:**

1.  Apri Google Colab (colab.research.google.com).
2.  Crea un nuovo notebook.
3.  Copia e incolla il codice sopra in una cella di codice.
4.  Esegui la cella (premi Shift+Invio).

In questo esempio abbiamo utilizzato la funzione pipeline. La funzione pipeline è un'astrazione di alto livello fornita dalla libreria transformers di Hugging Face. Essa incapsula l'intero processo di utilizzo di un modello, dalla pre-elaborazione dell'input alla post-elaborazione dell'output. In pratica, ti permette di utilizzare un modello con poche righe di codice, senza doverti preoccupare dei dettagli implementativi.

A cosa serve?

La funzione pipeline può essere utilizzata per una vasta gamma di compiti di NLP, tra cui:

    Generazione di testo: Generare testo a partire da un prompt, come nell'esempio precedente.
    Analisi del sentiment: Determinare il sentimento espresso in un testo (positivo, negativo, neutro).
    Risposta a domande: Rispondere a domande basate su un dato contesto.
    Traduzione automatica: Tradurre testo da una lingua all'altra.
    Riconoscimento di entità nominate (NER): Identificare entità nominate in un testo (persone, luoghi, organizzazioni).
    Riassunto di testo: Riassumere testi lunghi.
    Classificazione di testo: Classificare testi in diverse categorie.

Come funziona?

La funzione pipeline si occupa di:

    Caricare il modello: Scaricare e caricare il modello pre-addestrato specificato.
    Pre-elaborare l'input: Trasformare l'input (testo, immagine, audio) in un formato comprensibile dal modello.
    Eseguire l'inferenza: Inviare l'input al modello e ottenere l'output.
    Post-elaborare l'output: Trasformare l'output del modello in un formato leggibile.

Vantaggi dell'utilizzo di pipeline:

    Semplicità: Permette di utilizzare modelli complessi con poche righe di codice.
    Facilità d'uso: Non richiede una conoscenza approfondita dei dettagli implementativi dei modelli.
    Versatilità: Supporta una vasta gamma di compiti di NLP e altri compiti di intelligenza artificiale.
    Efficienza: Ottimizza il processo di utilizzo dei modelli.

Esempi di utilizzo:
Python

**Analisi del sentiment**
sentiment_analysis = pipeline("sentiment-analysis")
print(sentiment_analysis("I love Hugging Face!"))

**Risposta a domande**
question_answerer = pipeline("question-answering")
print(question_answerer(question="Where do I live?", context="My name is Sarah and I live in London"))

**Traduzione**
translator = pipeline("translation_en_to_de")
print(translator("Hello, how are you?"))

In sintesi, la funzione pipeline è uno strumento potente e versatile che semplifica l'utilizzo dei modelli di intelligenza artificiale, rendendoli accessibili anche a chi non ha una conoscenza approfondita dei dettagli implementativi.

E' possibile inserire personalizzazioni nelle pipeline di Hugging Face, e questo può essere fatto in diversi modi per adattare il comportamento della pipeline alle tue esigenze specifiche. Ecco alcune delle possibilità:

1. Scelta di modelli specifici:

    Quando crei una pipeline, puoi specificare quale modello pre-addestrato utilizzare. Questo ti permette di scegliere un modello che è stato addestrato per un compito specifico o che ha prestazioni migliori per il tuo caso d'uso.
    Ad esempio, puoi scegliere un modello di traduzione che è stato addestrato per una specifica coppia di lingue.

2. Configurazione dei parametri:

    Molte pipeline consentono di configurare i parametri per controllare il comportamento del modello.
    Ad esempio, puoi controllare la lunghezza del testo generato, la temperatura per la generazione di testo o la soglia di confidenza per l'analisi del sentiment.

3. Utilizzo di tokenizer personalizzati:

    I tokenizer sono utilizzati per convertire il testo in input in un formato comprensibile dal modello.
    In alcuni casi, potresti voler utilizzare un tokenizer personalizzato per gestire specifici tipi di testo o per migliorare le prestazioni del modello.

4. Creazione di pipeline personalizzate:

    Per compiti più complessi o per integrare funzionalità personalizzate, puoi creare le tue pipeline.
    Questo ti permette di definire il flusso di lavoro completo, dalla pre-elaborazione dell'input alla post-elaborazione dell'output.

5. Aggiunta di logica di post-elaborazione:

    Puoi aggiungere una logica di post-elaborazione personalizzata per modificare o filtrare l'output del modello.
    Questo può essere utile per formattare l'output in un modo specifico o per applicare regole aziendali.

Informazioni aggiuntive:

    La documentazione di Hugging Face fornisce informazioni dettagliate su come personalizzare le pipeline e creare pipeline personalizzate.
    È possibile trovare esempi e tutorial sulla piattaforma Hugging face, e nella documentazione della libreria "transformer".

In sintesi, le pipeline di Hugging Face offrono una notevole flessibilità e ti permettono di adattare il comportamento dei modelli alle tue esigenze specifiche.

Una possibile personalizzazione di una pipeline é la modifica della temperatura in una pipeline di Hugging Face per la generazione di testo:

**Cos'è la temperatura?**

La temperatura è un parametro che controlla la casualità della generazione di testo. Un valore di temperatura più basso rende il testo più deterministico e prevedibile, mentre un valore di temperatura più alto lo rende più casuale e creativo.

**Come modificare la temperatura:**

Puoi modificare la temperatura passando l'argomento `temperature` alla funzione `pipeline` quando crei la pipeline o quando la chiami per generare testo.

**Esempio:**

```python
from transformers import pipeline

# Crea una pipeline per la generazione di testo con temperatura 0.7
generator = pipeline('text-generation', model='gpt2', temperature=0.7)

# Genera testo con la pipeline
prompt = "Scrivi una breve storia su un gatto che parla."
generated_text = generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']

# Stampa il testo generato
print(generated_text)

# Esempio per modificare la temperatura a ogni chiamata.
generated_text_bassa_temp = generator(prompt, max_length=100, num_return_sequences=1, temperature=0.2)[0]['generated_text']
print(generated_text_bassa_temp)

generated_text_alta_temp = generator(prompt, max_length=100, num_return_sequences=1, temperature=1.5)[0]['generated_text']
print(generated_text_alta_temp)
```

**Spiegazione:**

* `temperature=0.7`: Imposta la temperatura della pipeline a 0.7. Puoi modificare questo valore per controllare la casualità del testo generato.
* `temperature=0.2`: Imposta una temperatura bassa, facendo risultare il testo molto conservativo.
* `temperature=1.5`: imposta una temperatura alta, rendendo il testo molto vario e a volte incoerente.

Sperimenta con diversi valori di temperatura per trovare quello che funziona meglio per il tuo caso d'uso. Valori di temperatura troppo alti possono portare a testo incoerente o senza senso, valori di temperatura troppo bassi possono portare a testo ripetitivo o noioso.

Oltre alla "temperatura", esistono diversi altri parametri che puoi modificare nelle pipeline di Hugging Face per controllare il comportamento dei modelli. Ecco alcuni esempi, suddivisi per tipologia di attività:

**Generazione di testo:**

* **`max_length`**:
    * Determina la lunghezza massima del testo generato.
    * Utile per controllare la quantità di testo prodotto.
* **`num_return_sequences`**:
    * Specifica il numero di sequenze di testo da generare.
    * Permette di ottenere più output dal modello.
* **`top_k`**:
    * Limita la selezione delle parole successive alle "k" parole più probabili.
    * Può migliorare la qualità del testo generato.
* **`top_p`**:
    * Utilizza il campionamento "nucleus" per selezionare le parole successive.
    * Considera solo le parole con una probabilità cumulativa superiore a "p".
* **`repetition_penalty`**:
    * Penalizza la ripetizione di parole o frasi nel testo generato.
    * Utile per evitare la ripetizione di contenuti.

**Analisi del sentiment:**

* **`top_k`**:
    * Restituisce le "k" etichette di sentiment più probabili.
    * Permette di ottenere una visione più dettagliata del sentimento.

**Risposta a domande:**

* **`max_answer_len`**:
    * Determina la lunghezza massima della risposta estratta dal contesto.
    * Utile per controllare la lunghezza delle risposte.
* **`top_k`**:
    * Restituisce le "k" risposte più probabili.
    * Permette di ottenere più risposte candidate.

**Traduzione automatica:**

* **`max_length`**:
    * Determina la lunghezza massima del testo tradotto.
    * Utile per controllare la lunghezza delle traduzioni.

**Riconoscimento di entità nominate (NER):**

* **`grouped_entities`**:
    * Raggruppa le entità nominate correlate.
    * Utile per ottenere una visione più strutturata delle entità.

**Informazioni importanti:**

* I parametri disponibili possono variare a seconda del modello e dell'attività.
* La documentazione di Hugging Face fornisce informazioni dettagliate sui parametri disponibili per ciascuna pipeline.
* È sempre utile consultare la documentazione dei modelli che si utilizzano, per avere un quadro completo delle possibilità di configurazione.
