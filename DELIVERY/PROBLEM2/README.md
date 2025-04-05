# Description of the System
1. Systemet bruker Yolo small v11 for objekt deteksjon
2. Systemet bruker en dynamisk uv map funksjon for å lage bounding boxen til scanneområdet for hver video.
3. Modellen bruker SORT algoritmen for å tracke et objekt over tid innen scanneområdet.
4. Mens produktet er i scanneområdet klassifiserer YOLO objektet om den er over en gitt confidence verdi. For hver frame gis det en "stemme" om at det objektet som trackes i scannesonen kan være denne klassen. Denne metoden hindrer også multiscanning av samme objekt.
5. Om produktet forlater scanneområdet huskes det at produktet eksisterte i en kort periode for å se om det returnerer.
6. Hvis produktet returnerer, og bevegelsesvektoren er innover i scannefeltet gjenopptas klassifiseringen og samling av stemmer.
7. Hvis produktet ikke returnerer, brukes klassifiseringsstemmene til å avgjøre hva produktet som nettopp forlot scannesones skal klassifiserer som, og legges til i kvitteringen.

Vi har i tillegg implementert funksjonalitet for å unngå multi-scanning av det samme produktet ved å legge til en global nedkjøling som er knyttet til om objektet forlater scannesonen og returnerer innen nedkjølingsperioden. Hvis objektet returnerer og har en vektor retning scannesonen gjenopptas klassifiseringen. Hvis objektet derimot returnerer, men neste frame er på vei ut igjen av scannesonen, lagres objektet for sikkerhets skyld i kvitteringen. Dette er en edge-case vi oppdaget hvor noen objekter kan "clippe" inn i scannesonen ved uhell og forvirre algoritmen. Det går nok an å lage bedre håndteringer for dette med litt mer tid og søvn.

Om et produkt forsvinner og ikke kommer tilbake legges det til i "pending_scans" hvor det etter en gitt tid (rundt 0.4s) lagres i kvitteringen.

### Running the system
*Use python 3.10.0*

1. python -m venv venv
2. Activate venv
3. python -m pip install -r requirements.txt
4. You need to manually install the correct torch with the right cuda version for your pc. For example: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
5. python main.py --video myvideo.mp4

#### Cuda and MacOS silicon
The program should detect cuda and apple silicon. If not, make sure the correct pip drivers are installed. See main.py for more details

### Performance notes
Hardware recommendations for optimal performance - This runs realtime using .2GB RAM on a mobile 3060 using about 50% of GPU. A Jetson should be capable of this.
Yolo nano could run with relatively high accuracy (instead of yolo small) for better performance.