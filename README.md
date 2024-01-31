# ChessVisionAI

### **Tim**
- SV 21/2020 Nikola Savić
- SV 33/2020 Jovan Šerbedžija

### **Asistent**
- Dragan Vidaković

## **Definicija problema**

- Projekat ima za cilj razvoj sistema dubokog učenja koji može analizirati video snimke šahovskih partija i prepoznati odigrane poteze, a zatim koristiti te informacije za učenje modela koji igra šah. 

### Skup podataka

- Skup podataka će se kreirati ručno, snimanjem šahovskih partija (sa [Chess.com](Chess.com))

### Metodologija
- Jedan video predstavlja jednu šahovsku partiju koja će biti podeljena na vremenske intervale koji predstavljaju poteze
- Pri svakom potezu, za detekciju početnog i krajnjeg polja figure će se koristiti Hough transformacija za detekciju ivica, kako bi se registrovala razlika između stanja table pre i posle poteza.
- Za identifikovanje tipa i boje figure koristiće se SVM klasifikator <em>kao i Sequential MLP Model klasifikator.</em>
- Izlaz iz obrade videa će biti skup poteza odigranih u toj šahovskoj partiji što će biti ulaz za treniranje <em>Feedforward</em> neuronske mreže koja će nakon treniranja biti u stanju da igra šah.  

### Evaluacija

- Evaluacija rezultata SVM klasifikatora i detekcije odigranog poteza će se meriti sledećim metrikama: tačnost, preciznost, odziv i f1 score
- Evaluacija rezultata Feedforward neuronske mreže će se meriti u odnosu na rezultate StockFish chess engine-a za istu poziciju.

## **Uputstvo za pokretanje**

```bash
#Svaka nova instanca terminala zahteva da se pokrene ova komanda:

export PYTHONPATH={Apsolutna putanja do direktorijuma projekta}

# primer export PYTHONPATH=/Users/user/Developer/projects/ChessVisionAI
```

```bash
# Instalirati sve neophodne pakete

pip install -r requirements.txt
```

- Pre pokretanja skripti potrebno je ažurirati apsolutne putanje u config.py.

- Prvo se pokreće skripta **code/chess_moves_extractor.ipynb** koja će na osnovu snimaka partija da izvuče odigrane poteze i pozicije i smestiće ih u data/training/raw_data folder (trajanje izvršavanja je oko 15 min).

- Nakon što se izvuku pozicije i odigrani potezi iz njih, pokreće se skripta **model_training/encoding.py** koja će podatke iz raw_data enkodirati u format za obučavanje mreže i smestiti ih u data/training/prepared_data.

<em>Rezultati prethodna dva koraka se već nalaze na repozitorijumu, tako da pokretanje skripti **code/chess_moves_extractor.ipynb** i **model_training/encoding.py** nije neophodno.</em>

- Treniranje modela se vrši pokretanjem **model_training/training.py** skripte i sačuvaće se model koji je ostvario najmanji loss na validacionom skupu.

- Nakon što se model istrenira potrebno je pozicionirati se u terminalu na folder **/uci** i pokrenuti komandu koja python skriptu **chess_vision.py** priprema u izvršnu datoteku:
```bash
pyinstaller --onefile chess_vision.py
```

- Na kraju je potrebno skinuti biblioteku cutechess sa linka https://github.com/cutechess/cutechess i pratiti uputstva kako je instalirati i pokrenuti.

- Kada se pokrene cutechess potrebno je otici na settings/engines i proslediti putanju do izvršne datoteke kreirane u prethodnom koraku.

- Na samom kraju potrebno je pokrenuti novu igru u cutechess (game/new) i odabrati istrenirani model.
