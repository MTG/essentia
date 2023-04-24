.. Essentia models

Essentia models
===============

This is a list of pre-trained TensorFlow models available in Essentia for various (music) audio analysis and classification tasks. To use the models in Essentia, see the `machine learning inference <machine_learning.html>`_ page. Some of the models are also available in other formats (ONNX, TensorFlow.js; contact us for more details).


All the models created by the MTG are licensed under CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/) and are also available under proprietary license upon request (https://www.upf.edu/web/mtg/contact). Check the `LICENSE <https://essentia.upf.edu/models/LICENSE>`_ of the models.


The models are serialized in `Protocol Buffer <https://developers.google.com/protocol-buffers/>`_ files suitable for inference with Essentia and TensorFlow. 
All available models in Protocol Buffer (``.pb``), `TensorFlow.js <https://www.tensorflow.org/js>`_ (``tfjs.zip``) and `ONNX <https://onnx.ai/>`_ (``.onnx``) formats can be found in our model repository.
For convenience, the documentation of each model includes direct links to the ``.pb`` and metadata ``.json`` files.
As this is an ongoing project we expect to keep adding new models and improved versions of the existing ones. These changes are tracked in this `CHANGELOG <https://essentia.upf.edu/models/CHANGELOG.md>`_.


We support models for the followings tasks:

* :ref:`Audio event recognition`
* :ref:`Music style classification`
* :ref:`Classifiers`
* :ref:`Feature extractors (embeddings)<Feature extractors>`
* :ref:`Pitch detection`
* :ref:`Source separation`
* :ref:`Tempo estimation`


Audio event recognition
^^^^^^^^^^^^^^^^^^^^^^^

AudioSet
--------

Audio event recognition (520 audio event classes):

`Speech`, `Child speech, kid speaking`, `Conversation`, `Narration, monologue`,
`Babbling`, `Speech synthesizer`, `Shout`, `Bellow`, `Whoop`, `Yell`, `Children
shouting`, `Screaming`, `Whispering`, `Laughter`, `Baby laughter`, `Giggle`,
`Snicker`, `Belly laugh`, `Chuckle, chortle`, `Crying, sobbing`, `Baby cry,
infant cry`, `Whimper`, `Wail, moan`, `Sigh`, `Singing`, `Choir`, `Yodeling`,
`Chant`, `Mantra`, `Child singing`, `Synthetic singing`, `Rapping`, `Humming`,
`Groan`, `Grunt`, `Whistling`, `Breathing`, `Wheeze`, `Snoring`, `Gasp`, `Pant`,
`Snort`, `Cough`, `Throat clearing`, `Sneeze`, `Sniff`, `Run`, `Shuffle`, `Walk,
footsteps`, `Chewing, mastication`, `Biting`, `Gargling`, `Stomach rumble`,
`Burping, eructation`, `Hiccup`, `Fart`, `Hands`, `Finger snapping`, `Clapping`,
`Heart sounds, heartbeat`, `Heart murmur`, `Cheering`, `Applause`, `Chatter`,
`Crowd`, `Hubbub, speech noise, speech babble`, `Children playing`, `Animal`,
`Domestic animals, pets`, `Dog`, `Bark`, `Yip`, `Howl`, `Bow-wow`, `Growling`,
`Whimper (dog)`, `Cat`, `Purr`, `Meow`, `Hiss`, `Caterwaul`, `Livestock, farm
animals, working animals`, `Horse`, `Clip-clop`, `Neigh, whinny`, `Cattle,
bovinae`, `Moo`, `Cowbell`, `Pig`, `Oink`, `Goat`, `Bleat`, `Sheep`, `Fowl`,
`Chicken, rooster`, `Cluck`, `Crowing, cock-a-doodle-doo`, `Turkey`, `Gobble`,
`Duck`, `Quack`, `Goose`, `Honk`, `Wild animals`, `Roaring cats (lions,
tigers)`, `Roar`, `Bird`, `Bird vocalization, bird call, bird song`, `Chirp,
tweet`, `Squawk`, `Pigeon, dove`, `Coo`, `Crow`, `Caw`, `Owl`, `Hoot`, `Bird
flight, flapping wings`, `Canidae, dogs, wolves`, `Rodents, rats, mice`,
`Mouse`, `Patter`, `Insect`, `Cricket`, `Mosquito`, `Fly, housefly`, `Buzz`,
`Bee, wasp, etc.`, `Frog`, `Croak`, `Snake`, `Rattle`, `Whale vocalization`,
`Music`, `Musical instrument`, `Plucked string instrument`, `Guitar`, `Electric
guitar`, `Bass guitar`, `Acoustic guitar`, `Steel guitar, slide guitar`,
`Tapping (guitar technique)`, `Strum`, `Banjo`, `Sitar`, `Mandolin`, `Zither`,
`Ukulele`, `Keyboard (musical)`, `Piano`, `Electric piano`, `Organ`, `Electronic
organ`, `Hammond organ`, `Synthesizer`, `Sampler`, `Harpsichord`, `Percussion`,
`Drum kit`, `Drum machine`, `Drum`, `Snare drum`, `Rimshot`, `Drum roll`, `Bass
drum`, `Timpani`, `Tabla`, `Cymbal`, `Hi-hat`, `Wood block`, `Tambourine`,
`Rattle (instrument)`, `Maraca`, `Gong`, `Tubular bells`, `Mallet percussion`,
`Marimba, xylophone`, `Glockenspiel`, `Vibraphone`, `Steelpan`, `Orchestra`,
`Brass instrument`, `French horn`, `Trumpet`, `Trombone`, `Bowed string
instrument`, `String section`, `Violin, fiddle`, `Pizzicato`, `Cello`, `Double
bass`, `Wind instrument, woodwind instrument`, `Flute`, `Saxophone`, `Clarinet`,
`Harp`, `Bell`, `Church bell`, `Jingle bell`, `Bicycle bell`, `Tuning fork`,
`Chime`, `Wind chime`, `Change ringing (campanology)`, `Harmonica`, `Accordion`,
`Bagpipes`, `Didgeridoo`, `Shofar`, `Theremin`, `Singing bowl`, `Scratching
(performance technique)`, `Pop music`, `Hip hop music`, `Beatboxing`, `Rock
music`, `Heavy metal`, `Punk rock`, `Grunge`, `Progressive rock`, `Rock and
roll`, `Psychedelic rock`, `Rhythm and blues`, `Soul music`, `Reggae`,
`Country`, `Swing music`, `Bluegrass`, `Funk`, `Folk music`, `Middle Eastern
music`, `Jazz`, `Disco`, `Classical music`, `Opera`, `Electronic music`, `House
music`, `Techno`, `Dubstep`, `Drum and bass`, `Electronica`, `Electronic dance
music`, `Ambient music`, `Trance music`, `Music of Latin America`, `Salsa
music`, `Flamenco`, `Blues`, `Music for children`, `New-age music`, `Vocal
music`, `A capella`, `Music of Africa`, `Afrobeat`, `Christian music`, `Gospel
music`, `Music of Asia`, `Carnatic music`, `Music of Bollywood`, `Ska`,
`Traditional music`, `Independent music`, `Song`, `Background music`, `Theme
music`, `Jingle (music)`, `Soundtrack music`, `Lullaby`, `Video game music`,
`Christmas music`, `Dance music`, `Wedding music`, `Happy music`, `Sad music`,
`Tender music`, `Exciting music`, `Angry music`, `Scary music`, `Wind`,
`Rustling leaves`, `Wind noise (microphone)`, `Thunderstorm`, `Thunder`,
`Water`, `Rain`, `Raindrop`, `Rain on surface`, `Stream`, `Waterfall`, `Ocean`,
`Waves, surf`, `Steam`, `Gurgling`, `Fire`, `Crackle`, `Vehicle`, `Boat, Water
vehicle`, `Sailboat, sailing ship`, `Rowboat, canoe, kayak`, `Motorboat,
speedboat`, `Ship`, `Motor vehicle (road)`, `Car`, `Vehicle horn, car horn,
honking`, `Toot`, `Car alarm`, `Power windows, electric windows`, `Skidding`,
`Tire squeal`, `Car passing by`, `Race car, auto racing`, `Truck`, `Air brake`,
`Air horn, truck horn`, `Reversing beeps`, `Ice cream truck, ice cream van`,
`Bus`, `Emergency vehicle`, `Police car (siren)`, `Ambulance (siren)`, `Fire
engine, fire truck (siren)`, `Motorcycle`, `Traffic noise, roadway noise`, `Rail
transport`, `Train`, `Train whistle`, `Train horn`, `Railroad car, train wagon`,
`Train wheels squealing`, `Subway, metro, underground`, `Aircraft`, `Aircraft
engine`, `Jet engine`, `Propeller, airscrew`, `Helicopter`, `Fixed-wing
aircraft, airplane`, `Bicycle`, `Skateboard`, `Engine`, `Light engine (high
frequency)`, `Dental drill, dentist's drill`, `Lawn mower`, `Chainsaw`, `Medium
engine (mid frequency)`, `Heavy engine (low frequency)`, `Engine knocking`,
`Engine starting`, `Idling`, `Accelerating, revving, vroom`, `Door`, `Doorbell`,
`Ding-dong`, `Sliding door`, `Slam`, `Knock`, `Tap`, `Squeak`, `Cupboard open or
close`, `Drawer open or close`, `Dishes, pots, and pans`, `Cutlery, silverware`,
`Chopping (food)`, `Frying (food)`, `Microwave oven`, `Blender`, `Water tap,
faucet`, `Sink (filling or washing)`, `Bathtub (filling or washing)`, `Hair
dryer`, `Toilet flush`, `Toothbrush`, `Electric toothbrush`, `Vacuum cleaner`,
`Zipper (clothing)`, `Keys jangling`, `Coin (dropping)`, `Scissors`, `Electric
shaver, electric razor`, `Shuffling cards`, `Typing`, `Typewriter`, `Computer
keyboard`, `Writing`, `Alarm`, `Telephone`, `Telephone bell ringing`,
`Ringtone`, `Telephone dialing, DTMF`, `Dial tone`, `Busy signal`, `Alarm
clock`, `Siren`, `Civil defense siren`, `Buzzer`, `Smoke detector, smoke alarm`,
`Fire alarm`, `Foghorn`, `Whistle`, `Steam whistle`, `Mechanisms`, `Ratchet,
pawl`, `Clock`, `Tick`, `Tick-tock`, `Gears`, `Pulleys`, `Sewing machine`,
`Mechanical fan`, `Air conditioning`, `Cash register`, `Printer`, `Camera`,
`Single-lens reflex camera`, `Tools`, `Hammer`, `Jackhammer`, `Sawing`, `Filing
(rasp)`, `Sanding`, `Power tool`, `Drill`, `Explosion`, `Gunshot, gunfire`,
`Machine gun`, `Fusillade`, `Artillery fire`, `Cap gun`, `Fireworks`,
`Firecracker`, `Burst, pop`, `Eruption`, `Boom`, `Wood`, `Chop`, `Splinter`,
`Crack`, `Glass`, `Chink, clink`, `Shatter`, `Liquid`, `Splash, splatter`,
`Slosh`, `Squish`, `Drip`, `Pour`, `Trickle, dribble`, `Gush`, `Fill (with
liquid)`, `Spray`, `Pump (liquid)`, `Stir`, `Boiling`, `Sonar`, `Arrow`,
`Whoosh, swoosh, swish`, `Thump, thud`, `Thunk`, `Electronic tuner`, `Effects
unit`, `Chorus effect`, `Basketball bounce`, `Bang`, `Slap, smack`, `Whack,
thwack`, `Smash, crash`, `Breaking`, `Bouncing`, `Whip`, `Flap`, `Scratch`,
`Scrape`, `Rub`, `Roll`, `Crushing`, `Crumpling, crinkling`, `Tearing`, `Beep,
bleep`, `Ping`, `Ding`, `Clang`, `Squeal`, `Creak`, `Rustle`, `Whir`, `Clatter`,
`Sizzle`, `Clicking`, `Clickety-clack`, `Rumble`, `Plop`, `Jingle, tinkle`,
`Hum`, `Zing`, `Boing`, `Crunch`, `Silence`, `Sine wave`, `Harmonic`, `Chirp
tone`, `Sound effect`, `Pulse`, `Inside, small room`, `Inside, large room or
hall`, `Inside, public space`, `Outside, urban or manmade`, `Outside, rural or
natural`, `Reverberation`, `Echo`, `Noise`, `Environmental noise`, `Static`,
`Mains hum`, `Distortion`, `Sidetone`, `Cacophony`, `White noise`, `Pink noise`,
`Throbbing`, `Vibration`, `Television`, `Radio`, `Field recording`

Dataset: AudioSet.

Outputs: audio event predictions and embeddings.

Naming convention: ``<task>-<architecture>-<version>.pb``

* ``task``: multi-label classification based on audioset (``audioset``).
* ``architecture``: a Mobilenet architecture (``yamnet``).
* ``version``: the version of the model.

Models:

* .. collapse:: <a class="reference external">audioset-yamnet</a>

    [`weights <https://essentia.upf.edu/models/audio-event-recognition/yamnet/audioset-yamnet-1.pb>`_, `metadata <https://essentia.upf.edu/models/audio-event-recognition/yamnet/audioset-yamnet-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/audio-event-recognition/yamnet/audioset-yamnet-1_predictions.py

    Python code for embedding extraction:

    .. literalinclude:: ../../src/examples/python/models/scripts/audio-event-recognition/yamnet/audioset-yamnet-1_embeddings.py


FSD-SINet
---------

Audio event recognition using the `FSD50K <https://zenodo.org/record/4060432>`_ dataset targeting 200 classes drawn from the `AudioSet Ontology <https://research.google.com/audioset/ontology/index.html>`_:

`Accelerating and revving and vroom`, `Accordion`, `Acoustic guitar`,
`Aircraft`, `Alarm`, `Animal`, `Applause`, `Bark`, `Bass drum`, `Bass guitar`,
`Bathtub (filling or washing)`, `Bell`, `Bicycle`, `Bicycle bell`, `Bird`, `Bird
vocalization and bird call and bird song`, `Boat and Water vehicle`, `Boiling`,
`Boom`, `Bowed string instrument`, `Brass instrument`, `Breathing`, `Burping and
eructation`, `Bus`, `Buzz`, `Camera`, `Car`, `Car passing by`, `Cat`, `Chatter`,
`Cheering`, `Chewing and mastication`, `Chicken and rooster`, `Child speech and
kid speaking`, `Chime`, `Chink and clink`, `Chirp and tweet`, `Chuckle and
chortle`, `Church bell`, `Clapping`, `Clock`, `Coin (dropping)`, `Computer
keyboard`, `Conversation`, `Cough`, `Cowbell`, `Crack`, `Crackle`, `Crash
cymbal`, `Cricket`, `Crow`, `Crowd`, `Crumpling and crinkling`, `Crushing`,
`Crying and sobbing`, `Cupboard open or close`, `Cutlery and silverware`,
`Cymbal`, `Dishes and pots and pans`, `Dog`, `Domestic animals and pets`,
`Domestic sounds and home sounds`, `Door`, `Doorbell`, `Drawer open or close`,
`Drill`, `Drip`, `Drum`, `Drum kit`, `Electric guitar`, `Engine`, `Engine
starting`, `Explosion`, `Fart`, `Female singing`, `Female speech and woman
speaking`, `Fill (with liquid)`, `Finger snapping`, `Fire`, `Fireworks`,
`Fixed-wing aircraft and airplane`, `Fowl`, `Frog`, `Frying (food)`, `Gasp`,
`Giggle`, `Glass`, `Glockenspiel`, `Gong`, `Growling`, `Guitar`, `Gull and
seagull`, `Gunshot and gunfire`, `Gurgling`, `Hammer`, `Hands`, `Harmonica`,
`Harp`, `Hi-hat`, `Hiss`, `Human group actions`, `Human voice`, `Idling`,
`Insect`, `Keyboard (musical)`, `Keys jangling`, `Knock`, `Laughter`, `Liquid`,
`Livestock and farm animals and working animals`, `Male singing`, `Male speech
and man speaking`, `Mallet percussion`, `Marimba and xylophone`, `Mechanical
fan`, `Mechanisms`, `Meow`, `Microwave oven`, `Motor vehicle (road)`,
`Motorcycle`, `Music`, `Musical instrument`, `Ocean`, `Organ`, `Packing tape and
duct tape`, `Percussion`, `Piano`, `Plucked string instrument`, `Pour`, `Power
tool`, `Printer`, `Purr`, `Race car and auto racing`, `Rail transport`, `Rain`,
`Raindrop`, `Ratchet and pawl`, `Rattle`, `Rattle (instrument)`, `Respiratory
sounds`, `Ringtone`, `Run`, `Sawing`, `Scissors`, `Scratching (performance
technique)`, `Screaming`, `Screech`, `Shatter`, `Shout`, `Sigh`, `Singing`,
`Sink (filling or washing)`, `Siren`, `Skateboard`, `Slam`, `Sliding door`,
`Snare drum`, `Sneeze`, `Speech`, `Speech synthesizer`, `Splash and splatter`,
`Squeak`, `Stream`, `Strum`, `Subway and metro and underground`, `Tabla`,
`Tambourine`, `Tap`, `Tearing`, `Telephone`, `Thump and thud`, `Thunder`,
`Thunderstorm`, `Tick`, `Tick-tock`, `Toilet flush`, `Tools`, `Traffic noise and
roadway noise`, `Train`, `Trickle and dribble`, `Truck`, `Trumpet`,
`Typewriter`, `Typing`, `Vehicle`, `Vehicle horn and car horn and honking`,
`Walk and footsteps`, `Water`, `Water tap and faucet`, `Waves and surf`,
`Whispering`, `Whoosh and swoosh and swish`, `Wild animals`, `Wind`, `Wind
chime`, `Wind instrument and woodwind instrument`, `Wood`, `Writing`, `Yell`,
`Zipper (clothing)`

Dataset: FSD50K.

Output: audio event predictions or embeddings.

Naming convention: ``<task>-<architecture>-<variation>-<si_technique>-<version>.pb``

* ``task``: multi-label classification based on FSD50K (``fsd``).
* ``architecture``: shift invariant net (``sinet``).
* ``variation``: ``vgg42`` is a variation of ``vgg41`` with twice the number of filters for each convolutional layer.
* ``si_technique``: the shift-invariance technique may be trainable low-pass filters (``tlpf``), or adaptative polyphase sampling (``aps``), or both (``tlpf_aps``).
* ``version``: the model version.

Models:

* .. collapse:: <a class="reference external">fsd-sinet-vgg41-tlpf</a>

    [`weights <https://essentia.upf.edu/models/audio-event-recognition/fsd-sinet/fsd-sinet-vgg41-tlpf-1.pb>`_, `metadata <https://essentia.upf.edu/models/audio-event-recognition/fsd-sinet/fsd-sinet-vgg41-tlpf-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/audio-event-recognition/fsd-sinet/fsd-sinet-vgg41-tlpf-1_predictions.py

    Python code for embedding extraction:

    .. literalinclude:: ../../src/examples/python/models/scripts/audio-event-recognition/fsd-sinet/fsd-sinet-vgg41-tlpf-1_embeddings.py

* .. collapse:: <a class="reference external">fsd-sinet-vgg42-aps</a>

    [`weights <https://essentia.upf.edu/models/audio-event-recognition/fsd-sinet/fsd-sinet-vgg42-aps-1.pb>`_, `metadata <https://essentia.upf.edu/models/audio-event-recognition/fsd-sinet/fsd-sinet-vgg42-aps-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/audio-event-recognition/fsd-sinet/fsd-sinet-vgg42-aps-1_predictions.py

    Python code for embedding extraction:

    .. literalinclude:: ../../src/examples/python/models/scripts/audio-event-recognition/fsd-sinet/fsd-sinet-vgg42-aps-1_embeddings.py

* .. collapse:: <a class="reference external">fsd-sinet-vgg42-tlpf_aps</a>

    [`weights <https://essentia.upf.edu/models/audio-event-recognition/fsd-sinet/fsd-sinet-vgg42-tlpf_aps-1.pb>`_, `metadata <https://essentia.upf.edu/models/audio-event-recognition/fsd-sinet/fsd-sinet-vgg42-tlpf_aps-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/audio-event-recognition/fsd-sinet/fsd-sinet-vgg42-tlpf_aps-1_predictions.py

    Python code for embedding extraction:

    .. literalinclude:: ../../src/examples/python/models/scripts/audio-event-recognition/fsd-sinet/fsd-sinet-vgg42-tlpf_aps-1_embeddings.py

* .. collapse:: <a class="reference external">fsd-sinet-vgg42-tlpf</a>

    [`weights <https://essentia.upf.edu/models/audio-event-recognition/fsd-sinet/fsd-sinet-vgg42-tlpf-1.pb>`_, `metadata <https://essentia.upf.edu/models/audio-event-recognition/fsd-sinet/fsd-sinet-vgg42-tlpf-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/audio-event-recognition/fsd-sinet/fsd-sinet-vgg42-tlpf-1_predictions.py

    Python code for embedding extraction:

    .. literalinclude:: ../../src/examples/python/models/scripts/audio-event-recognition/fsd-sinet/fsd-sinet-vgg42-tlpf-1_embeddings.py



Music style classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

Discogs-Effnet
--------------

Music style classification by 400 styles from the Discogs taxonomy:

* Blues `Boogie Woogie`, `Chicago Blues`, `Country Blues`, `Delta Blues`, `Electric Blues`, `Harmonica Blues`, `Jump Blues`, `Louisiana Blues`, `Modern Electric Blues`, `Piano Blues`, `Rhythm & Blues`, `Texas Blues`
* Brass & Military: `Brass Band`, `Marches`, `Military`
* Children's: `Educational`, `Nursery Rhymes`, `Story`
* Classical: `Baroque`, `Choral`, `Classical`, `Contemporary`, `Impressionist`, `Medieval`, `Modern`, `Neo-Classical`, `Neo-Romantic`, `Opera`, `Post-Modern`, `Renaissance`, `Romantic`
* Electronic: `Abstract`, `Acid`, `Acid House`, `Acid Jazz`, `Ambient`, `Bassline`, `Beatdown`, `Berlin-School`, `Big Beat`, `Bleep`, `Breakbeat`, `Breakcore`, `Breaks`, `Broken Beat`, `Chillwave`, `Chiptune`, `Dance-pop`, `Dark Ambient`, `Darkwave`, `Deep House`, `Deep Techno`, `Disco`, `Disco Polo`, `Donk`, `Downtempo`, `Drone`, `Drum n Bass`, `Dub`, `Dub Techno`, `Dubstep`, `Dungeon Synth`, `EBM`, `Electro`, `Electro House`, `Electroclash`, `Euro House`, `Euro-Disco`, `Eurobeat`, `Eurodance`, `Experimental`, `Freestyle`, `Future Jazz`, `Gabber`, `Garage House`, `Ghetto`, `Ghetto House`, `Glitch`, `Goa Trance`, `Grime`, `Halftime`, `Hands Up`, `Happy Hardcore`, `Hard House`, `Hard Techno`, `Hard Trance`, `Hardcore`, `Hardstyle`, `Hi NRG`, `Hip Hop`, `Hip-House`, `House`, `IDM`, `Illbient`, `Industrial`, `Italo House`, `Italo-Disco`, `Italodance`, `Jazzdance`, `Juke`, `Jumpstyle`, `Jungle`, `Latin`, `Leftfield`, `Makina`, `Minimal`, `Minimal Techno`, `Modern Classical`, `Musique Concrète`, `Neofolk`, `New Age`, `New Beat`, `New Wave`, `Noise`, `Nu-Disco`, `Power Electronics`, `Progressive Breaks`, `Progressive House`, `Progressive Trance`, `Psy-Trance`, `Rhythmic Noise`, `Schranz`, `Sound Collage`, `Speed Garage`, `Speedcore`, `Synth-pop`, `Synthwave`, `Tech House`, `Tech Trance`, `Techno`, `Trance`, `Tribal`, `Tribal House`, `Trip Hop`, `Tropical House`, `UK Garage`, `Vaporwave`
* Folk, World, & Country: `African`, `Bluegrass`, `Cajun`, `Canzone Napoletana`, `Catalan Music`, `Celtic`, `Country`, `Fado`, `Flamenco`, `Folk`, `Gospel`, `Highlife`, `Hillbilly`, `Hindustani`, `Honky Tonk`, `Indian Classical`, `Laïkó`, `Nordic`, `Pacific`, `Polka`, `Raï`, `Romani`, `Soukous`, `Séga`, `Volksmusik`, `Zouk`, `Éntekhno`
* Funk / Soul: `Afrobeat`, `Boogie`, `Contemporary R&B`, `Disco`, `Free Funk`, `Funk`, `Gospel`, `Neo Soul`, `New Jack Swing`, `P.Funk`, `Psychedelic`, `Rhythm & Blues`, `Soul`, `Swingbeat`, `UK Street Soul`
* Hip Hop: `Bass Music`, `Boom Bap`, `Bounce`, `Britcore`, `Cloud Rap`, `Conscious`, `Crunk`, `Cut-up/DJ`, `DJ Battle Tool`, `Electro`, `G-Funk`, `Gangsta`, `Grime`, `Hardcore Hip-Hop`, `Horrorcore`, `Instrumental`, `Jazzy Hip-Hop`, `Miami Bass`, `Pop Rap`, `Ragga HipHop`, `RnB/Swing`, `Screw`, `Thug Rap`, `Trap`, `Trip Hop`, `Turntablism`
* Jazz: `Afro-Cuban Jazz`, `Afrobeat`, `Avant-garde Jazz`, `Big Band`, `Bop`, `Bossa Nova`, `Contemporary Jazz`, `Cool Jazz`, `Dixieland`, `Easy Listening`, `Free Improvisation`, `Free Jazz`, `Fusion`, `Gypsy Jazz`, `Hard Bop`, `Jazz-Funk`, `Jazz-Rock`, `Latin Jazz`, `Modal`, `Post Bop`, `Ragtime`, `Smooth Jazz`, `Soul-Jazz`, `Space-Age`, `Swing`
* Latin: `Afro-Cuban`, `Baião`, `Batucada`, `Beguine`, `Bolero`, `Boogaloo`, `Bossanova`, `Cha-Cha`, `Charanga`, `Compas`, `Cubano`, `Cumbia`, `Descarga`, `Forró`, `Guaguancó`, `Guajira`, `Guaracha`, `MPB`, `Mambo`, `Mariachi`, `Merengue`, `Norteño`, `Nueva Cancion`, `Pachanga`, `Porro`, `Ranchera`, `Reggaeton`, `Rumba`, `Salsa`, `Samba`, `Son`, `Son Montuno`, `Tango`, `Tejano`, `Vallenato`
* Non-Music: `Audiobook`, `Comedy`, `Dialogue`, `Education`, `Field Recording`, `Interview`, `Monolog`, `Poetry`, `Political`, `Promotional`, `Radioplay`, `Religious`, `Spoken Word`
* Pop: `Ballad`, `Bollywood`, `Bubblegum`, `Chanson`, `City Pop`, `Europop`, `Indie Pop`, `J-pop`, `K-pop`, `Kayōkyoku`, `Light Music`, `Music Hall`, `Novelty`, `Parody`, `Schlager`, `Vocal`
* Reggae: `Calypso`, `Dancehall`, `Dub`, `Lovers Rock`, `Ragga`, `Reggae`, `Reggae-Pop`, `Rocksteady`, `Roots Reggae`, `Ska`, `Soca`
* Rock: `AOR`, `Acid Rock`, `Acoustic`, `Alternative Rock`, `Arena Rock`, `Art Rock`, `Atmospheric Black Metal`, `Avantgarde`, `Beat`, `Black Metal`, `Blues Rock`, `Brit Pop`, `Classic Rock`, `Coldwave`, `Country Rock`, `Crust`, `Death Metal`, `Deathcore`, `Deathrock`, `Depressive Black Metal`, `Doo Wop`, `Doom Metal`, `Dream Pop`, `Emo`, `Ethereal`, `Experimental`, `Folk Metal`, `Folk Rock`, `Funeral Doom Metal`, `Funk Metal`, `Garage Rock`, `Glam`, `Goregrind`, `Goth Rock`, `Gothic Metal`, `Grindcore`, `Grunge`, `Hard Rock`, `Hardcore`, `Heavy Metal`, `Indie Rock`, `Industrial`, `Krautrock`, `Lo-Fi`, `Lounge`, `Math Rock`, `Melodic Death Metal`, `Melodic Hardcore`, `Metalcore`, `Mod`, `Neofolk`, `New Wave`, `No Wave`, `Noise`, `Noisecore`, `Nu Metal`, `Oi`, `Parody`, `Pop Punk`, `Pop Rock`, `Pornogrind`, `Post Rock`, `Post-Hardcore`, `Post-Metal`, `Post-Punk`, `Power Metal`, `Power Pop`, `Power Violence`, `Prog Rock`, `Progressive Metal`, `Psychedelic Rock`, `Psychobilly`, `Pub Rock`, `Punk`, `Rock & Roll`, `Rockabilly`, `Shoegaze`, `Ska`, `Sludge Metal`, `Soft Rock`, `Southern Rock`, `Space Rock`, `Speed Metal`, `Stoner Rock`, `Surf`, `Symphonic Rock`, `Technical Death Metal`, `Thrash`, `Twist`, `Viking Metal`, `Yé-Yé`
* Stage & Screen: `Musical`, `Score`, `Soundtrack`, `Theme`

Dataset: in-house (MTG).

Outputs: music style predictions and embeddings.

Naming convention: ``<task>-<architecture>-bs<batch_size>-<version>.pb``

* ``task``: multi-label classification based on discogs labels (``discogs``).
* ``architecture``: an efficientnet b0 architecture (``effnet``).
* ``batch_size``: the model is only available with a fixed batch size of 64.

Models:

* .. collapse:: <a class="reference external">discogs-effnet-bs64</a>

    [`weights <https://essentia.upf.edu/models/music-style-classification/discogs-effnet/discogs-effnet-bs64-1.pb>`_, `metadata <https://essentia.upf.edu/models/music-style-classification/discogs-effnet/discogs-effnet-bs64-1.json>`_, `demo <https://replicate.com/mtg/effnet-discogs>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/music-style-classification/discogs-effnet/discogs-effnet-bs64-1_predictions.py

    Python code for embedding extraction:

    .. literalinclude:: ../../src/examples/python/models/scripts/music-style-classification/discogs-effnet/discogs-effnet-bs64-1_embeddings.py


*Note: The batch size limitation is a work-arround due to a problem porting the model from ONNX to TensorFlow. Additionally, an ONNX version of the model with* `dynamic batch <https://essentia.upf.edu/models/music-style-classification/discogs-effnet/discogs-effnet-bsdynamic-1.onnx>`_ *size is provided.*





Feature extractors
^^^^^^^^^^^^^^^^^^


OpenL3
------

Audio embedding model trained on audio-visual correspondence in a self-supervised manner.

Dataset: AudioSet subsets of videos with environmental sounds and musical content.

Output: embeddings.

Naming convention: ``<architecture>-<source_task>-mel<n_mel_bands>-emb<n_embeddings>-<version>.pb``

* ``architecture``: the OpenL3 architecture (``openl3``).
* ``source_task``: can be enviromental sounds (``env``) or music (``music``).
* ``n_mel_bands``: number of input mel-bands.
* ``n_embeddings``: the number of dimensions in the output embedding layer.
* ``version``: the version of the model.

Models:

* .. collapse:: <a class="reference external">openl3-env-mel128-emb512</a>

    [`weights <https://essentia.upf.edu/models/feature-extractors/openl3/openl3-env-mel128-emb512-3.pb>`_, `metadata <https://essentia.upf.edu/models/feature-extractors/openl3/openl3-env-mel128-emb512-3.json>`_]

    We do not have a dedicated algorithm to extract embeddings with this model. For now, OpenL3 embeddings can be extracted using this `script <https://gist.github.com/palonso/cfebe37e5492b5a3a31775d8eae8d9a8>`_.

* .. collapse:: <a class="reference external">openl3-env-mel128-emb6144</a>

    [`weights <https://essentia.upf.edu/models/feature-extractors/openl3/openl3-env-mel128-emb6144-3.pb>`_, `metadata <https://essentia.upf.edu/models/feature-extractors/openl3/openl3-env-mel128-emb6144-3.json>`_]

    We do not have a dedicated algorithm to extract embeddings with this model. For now, OpenL3 embeddings can be extracted using this `script <https://gist.github.com/palonso/cfebe37e5492b5a3a31775d8eae8d9a8>`_.

* .. collapse:: <a class="reference external">openl3-env-mel256-emb512</a>

    [`weights <https://essentia.upf.edu/models/feature-extractors/openl3/openl3-env-mel256-emb512-3.pb>`_, `metadata <https://essentia.upf.edu/models/feature-extractors/openl3/openl3-env-mel256-emb512-3.json>`_]

    We do not have a dedicated algorithm to extract embeddings with this model. For now, OpenL3 embeddings can be extracted using this `script <https://gist.github.com/palonso/cfebe37e5492b5a3a31775d8eae8d9a8>`_.

* .. collapse:: <a class="reference external">openl3-env-mel256-emb6144</a>

    [`weights <https://essentia.upf.edu/models/feature-extractors/openl3/openl3-env-mel256-emb6144-3.pb>`_, `metadata <https://essentia.upf.edu/models/feature-extractors/openl3/openl3-env-mel256-emb6144-3.json>`_]

    We do not have a dedicated algorithm to extract embeddings with this model. For now, OpenL3 embeddings can be extracted using this `script <https://gist.github.com/palonso/cfebe37e5492b5a3a31775d8eae8d9a8>`_.

* .. collapse:: <a class="reference external">openl3-music-mel128-emb512</a>

    [`weights <https://essentia.upf.edu/models/feature-extractors/openl3/openl3-music-mel128-emb512-3.pb>`_, `metadata <https://essentia.upf.edu/models/feature-extractors/openl3/openl3-music-mel128-emb512-3.json>`_]

    We do not have a dedicated algorithm to extract embeddings with this model. For now, OpenL3 embeddings can be extracted using this `script <https://gist.github.com/palonso/cfebe37e5492b5a3a31775d8eae8d9a8>`_.

* .. collapse:: <a class="reference external">openl3-music-mel128-emb6144</a>

    [`weights <https://essentia.upf.edu/models/feature-extractors/openl3/openl3-music-mel128-emb6144-3.pb>`_, `metadata <https://essentia.upf.edu/models/feature-extractors/openl3/openl3-music-mel128-emb6144-3.json>`_]

    We do not have a dedicated algorithm to extract embeddings with this model. For now, OpenL3 embeddings can be extracted using this `script <https://gist.github.com/palonso/cfebe37e5492b5a3a31775d8eae8d9a8>`_.

* .. collapse:: <a class="reference external">openl3-music-mel256-emb512</a>

    [`weights <https://essentia.upf.edu/models/feature-extractors/openl3/openl3-music-mel256-emb512-3.pb>`_, `metadata <https://essentia.upf.edu/models/feature-extractors/openl3/openl3-music-mel256-emb512-3.json>`_]

    We do not have a dedicated algorithm to extract embeddings with this model. For now, OpenL3 embeddings can be extracted using this `script <https://gist.github.com/palonso/cfebe37e5492b5a3a31775d8eae8d9a8>`_.

* .. collapse:: <a class="reference external">openl3-music-mel256-emb6144</a>

    [`weights <https://essentia.upf.edu/models/feature-extractors/openl3/openl3-music-mel256-emb6144-3.pb>`_, `metadata <https://essentia.upf.edu/models/feature-extractors/openl3/openl3-music-mel256-emb6144-3.json>`_]

    We do not have a dedicated algorithm to extract embeddings with this model. For now, OpenL3 embeddings can be extracted using this `script <https://gist.github.com/palonso/cfebe37e5492b5a3a31775d8eae8d9a8>`_.


AudioSet-VGGish
---------------

Audio embedding model accompanying the AudioSet dataset, trained in a supervised manner using tag information for YouTube videos.

Dataset: a subset of Youtube-8M.

Output: embeddings.

Naming convention: ``<task>-<architecture>-<version>.pb``

* ``task``: multi-label classification using an in-house dataset related to AudioSet (``audioset``).
* ``architecture``: a CNN model with vgg-like convolutional layers (``vggish``).
* ``version``: the model version.

Models:

* .. collapse:: <a class="reference external">audioset-vggish</a>

    [`weights <https://essentia.upf.edu/models/feature-extractors/vggish/audioset-vggish-3.pb>`_, `metadata <https://essentia.upf.edu/models/feature-extractors/vggish/audioset-vggish-3.json>`_]

    Python code for embedding extraction:

    .. literalinclude:: ../../src/examples/python/models/scripts/feature-extractors/vggish/audioset-vggish-3_embeddings.py



Discogs-EffNet
--------------

Audio embedding models trained with a contrastive learning objective using Discogs metadata.
There are versions trained on artist, label, release, and track similarity, as well as a multi-task model trained in all of them simusltaneously.

Dataset: In-house dataset annotated with Discogs metadata.

Output: embeddings.

Naming convention: ``discogs_<task>_embeddings-<architecture>-bs<batch-size>-<version>.pb``

* ``task``: contrastive learning targeting artist (``artist``), label (``label``), album (``release``), track (``track``), or a multi-task (``multi``) similarity objective.
* ``architecture``: an efficientnet b0 architecture (``effnet``).
* ``batch_size``: for now, the models are only available with a fixed batch size of 64 samples.
* ``version``: the version of the model.

Models:

* .. collapse:: <a class="reference external">discogs-effnet-bs64</a>

    [`weights <https://essentia.upf.edu/models/music-style-classification/discogs-effnet/discogs-effnet-bs64-1.pb>`_, `metadata <https://essentia.upf.edu/models/music-style-classification/discogs-effnet/discogs-effnet-bs64-1.json>`_, `demo <https://replicate.com/mtg/effnet-discogs>`_]


    Python code for embedding extraction:

    .. literalinclude:: ../../src/examples/python/models/scripts/music-style-classification/discogs-effnet/discogs-effnet-bs64-1_embeddings.py

* .. collapse:: <a class="reference external">discogs_artist_embeddings-effnet-bs64</a>

    [`weights <https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs_artist_embeddings-effnet-bs64-1.pb>`_, `metadata <https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs_artist_embeddings-effnet-bs64-1.json>`_]

    Python code for embedding extraction:

    .. literalinclude:: ../../src/examples/python/models/scripts/feature-extractors/discogs-effnet/discogs_artist_embeddings-effnet-bs64-1_embeddings.py

* .. collapse:: <a class="reference external">discogs_label_embeddings-effnet-bs64</a>

    [`weights <https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs_label_embeddings-effnet-bs64-1.pb>`_, `metadata <https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs_label_embeddings-effnet-bs64-1.json>`_]

    Python code for embedding extraction:

    .. literalinclude:: ../../src/examples/python/models/scripts/feature-extractors/discogs-effnet/discogs_label_embeddings-effnet-bs64-1_embeddings.py

* .. collapse:: <a class="reference external">discogs_multi_embeddings-effnet-bs64</a>

    [`weights <https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs_multi_embeddings-effnet-bs64-1.pb>`_, `metadata <https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs_multi_embeddings-effnet-bs64-1.json>`_]

    Python code for embedding extraction:

    .. literalinclude:: ../../src/examples/python/models/scripts/feature-extractors/discogs-effnet/discogs_multi_embeddings-effnet-bs64-1_embeddings.py

* .. collapse:: <a class="reference external">discogs_release_embeddings-effnet-bs64</a>

    [`weights <https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs_release_embeddings-effnet-bs64-1.pb>`_, `metadata <https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs_release_embeddings-effnet-bs64-1.json>`_]

    Python code for embedding extraction:

    .. literalinclude:: ../../src/examples/python/models/scripts/feature-extractors/discogs-effnet/discogs_release_embeddings-effnet-bs64-1_embeddings.py

* .. collapse:: <a class="reference external">discogs_track_embeddings-effnet-bs64</a>

    [`weights <https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs_track_embeddings-effnet-bs64-1.pb>`_, `metadata <https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs_track_embeddings-effnet-bs64-1.json>`_]

    Python code for embedding extraction:

    .. literalinclude:: ../../src/examples/python/models/scripts/feature-extractors/discogs-effnet/discogs_track_embeddings-effnet-bs64-1_embeddings.py


MSD-MusiCNN
-----------

Music embedding extraction based on  auto-tagging with 50 common music tags.

Dataset: Million Song Dataset.

Outputs: embeddings.

Naming convention: ``<task>-<architecture>-<version>.pb``

* ``task``: multi-label classification based on the Million Song Dataset (``msd``).
* ``architecture``: musicnn (``musicnn``) or vgg-like (``vgg``) architecture.
* ``version``: the version of the model.

Models:

* .. collapse:: <a class="reference external">msd-musicnn</a>

    [`weights <https://essentia.upf.edu/models/autotagging/msd/msd-musicnn-1.pb>`_, `metadata <https://essentia.upf.edu/models/autotagging/msd/msd-musicnn-1.json>`_]

    Python code for predictions:


    Python code for embedding extraction:

    .. literalinclude:: ../../src/examples/python/models/scripts/autotagging/msd/msd-musicnn-1_embeddings.py



Pitch detection
^^^^^^^^^^^^^^^

Monophonic pitch tracker (CREPE)
--------------------------------

Monophonic pitch detection (360 20-cent pitch bins, C1-B7).

Dataset: RWC-synth, MDB-stem-synth.

Output: pitch predictions.

Naming convention: ``<architecture>-<model_size>-<version>.pb``

* ``architecture``: the CREPE architecture (``crepe``).
* ``model_size``: the model size ranging from ``tiny`` to ``full``. A larger model is expected to perform better at the expense of additional computational cost.
* ``version``: the version of the model.

Models:

* .. collapse:: <a class="reference external">crepe-full</a>

    [`weights <https://essentia.upf.edu/models/pitch/crepe/crepe-full-1.pb>`_, `metadata <https://essentia.upf.edu/models/pitch/crepe/crepe-full-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/pitch/crepe/crepe-full-1_predictions.py

* .. collapse:: <a class="reference external">crepe-large</a>

    [`weights <https://essentia.upf.edu/models/pitch/crepe/crepe-large-1.pb>`_, `metadata <https://essentia.upf.edu/models/pitch/crepe/crepe-large-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/pitch/crepe/crepe-large-1_predictions.py

* .. collapse:: <a class="reference external">crepe-medium</a>

    [`weights <https://essentia.upf.edu/models/pitch/crepe/crepe-medium-1.pb>`_, `metadata <https://essentia.upf.edu/models/pitch/crepe/crepe-medium-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/pitch/crepe/crepe-medium-1_predictions.py

* .. collapse:: <a class="reference external">crepe-small</a>

    [`weights <https://essentia.upf.edu/models/pitch/crepe/crepe-small-1.pb>`_, `metadata <https://essentia.upf.edu/models/pitch/crepe/crepe-small-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/pitch/crepe/crepe-small-1_predictions.py

* .. collapse:: <a class="reference external">crepe-tiny</a>

    [`weights <https://essentia.upf.edu/models/pitch/crepe/crepe-tiny-1.pb>`_, `metadata <https://essentia.upf.edu/models/pitch/crepe/crepe-tiny-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/pitch/crepe/crepe-tiny-1_predictions.py



Source separation
^^^^^^^^^^^^^^^^^

Spleeter
--------

Source separation into 2, 4, or 5 stems.

Dataset: in-house (Deezer).

Output: waveform of the separated sources.

Naming convention: ``<architecture>-<number_of_stems>s-<version>.pb``

* ``architecture``: a spleeter architecture (``spleeter``).
* ``number_of_stems``: can be 2 (vocals and accompaniment), 4 (vocals, drums, bass, and other separation) or 5 (vocals, drums, bass, piano, and other separation).
* ``version``: the version of the model.

Models:

* .. collapse:: <a class="reference external">speeter-2s</a>

    [`weights <https://essentia.upf.edu/models/source-separation/spleeter/spleeter-2s-3.pb>`_, `metadata <https://essentia.upf.edu/models/source-separation/spleeter/spleeter-2s-3.json>`_]

    Python code for source separation:

    .. code-block:: python

        from essentia.standard import AudioLoader, TensorflowPredict
        from essentia import Pool
        import numpy as np

        # Input should be audio @48kHz.
        audio, sr, _, _, _, _ = AudioLoader(filename="audio.wav")()

        pool = Pool()
        # The input needs to have 4 dimensions so that it is interpreted as an Essentia tensor.
        pool.set("waveform", audio[..., np.newaxis, np.newaxis])

        model = TensorflowPredict(
            graphFilename="spleeter-2s-3.pb",
            inputs=["waveform"],
            outputs=["waveform_vocals", "waveform_accompaniment"]
        )

        out_pool = model(pool)
        vocals = out_pool["waveform_vocals"].squeeze()
        accompaniment = out_pool["waveform_accompaniment"].squeeze()

* .. collapse:: <a class="reference external">speeter-4s</a>

    [`weights <https://essentia.upf.edu/models/source-separation/spleeter/spleeter-4s-3.pb>`_, `metadata <https://essentia.upf.edu/models/source-separation/spleeter/spleeter-4s-3.json>`_]

    Python code for source separation:

    .. code-block:: python

        from essentia.standard import AudioLoader, TensorflowPredict
        from essentia import Pool
        import numpy as np

        # Input should be audio @48kHz.
        audio, sr, _, _, _, _ = AudioLoader(filename="audio.wav")()

        pool = Pool()
        # The input needs to have 4 dimensions so that it is interpreted as an Essentia tensor.
        pool.set("waveform", audio[..., np.newaxis, np.newaxis])

        model = TensorflowPredict(
            graphFilename="spleeter-4s-3.pb",
            inputs=["waveform"],
            outputs=["waveform_vocals", "waveform_drums", "waveform_bass", "waveform_other"]
        )

        out_pool = model(pool)
        vocals = out_pool["waveform_vocals"].squeeze()
        drums = out_pool["waveform_drums"].squeeze()
        bass = out_pool["waveform_bass"].squeeze()
        other = out_pool["waveform_other"].squeeze()

* .. collapse:: <a class="reference external">speeter-5s</a>

    [`weights <https://essentia.upf.edu/models/source-separation/spleeter/spleeter-5s-3.pb>`_, `metadata <https://essentia.upf.edu/models/source-separation/spleeter/spleeter-5s-3.json>`_]

    Python code for source separation:

    .. code-block:: python

        from essentia.standard import AudioLoader, TensorflowPredict
        from essentia import Pool
        import numpy as np

        # Input should be audio @48kHz.
        audio, sr, _, _, _, _ = AudioLoader(filename="audio.wav")()

        pool = Pool()
        # The input needs to have 4 dimensions so that it is interpreted as an Essentia tensor.
        pool.set("waveform", audio[..., np.newaxis, np.newaxis])

        model = TensorflowPredict(
            graphFilename="spleeter-5s-3.pb",
            inputs=["waveform"],
            outputs=["waveform_vocals", "waveform_drums", "waveform_bass", "waveform_piano", "waveform_other"]
        )

        out_pool = model(pool)
        vocals = out_pool["waveform_vocals"].squeeze()
        drums = out_pool["waveform_drums"].squeeze()
        bass = out_pool["waveform_bass"].squeeze()
        bass = out_pool["waveform_piano"].squeeze()
        other = out_pool["waveform_other"].squeeze()


Tempo estimation
^^^^^^^^^^^^^^^^

TempoCNN
--------

Tempo classification (256 BPM classes, 30-286 BPM).

Dataset: Extended Ballroom, LMDTempo, MTGTempo.

Output: tempo predictions.

Naming convention: ``<architecture>-k<model_size>-<version>.pb``

* ``architecture``: a TempoCNN architecture feature square filters (``deepsquare``) or longitudinal ones (``deeptemp``).
* ``model_size``: a model size factor (4 or 16). A larger model is expected to perform better at the expense of additional computational cost.
* ``version``: the version of the model.

Models:

* .. collapse:: <a class="reference external">deepsquare-k16</a>

    [`weights <https://essentia.upf.edu/models/tempo/tempocnn/deepsquare-k16-3.pb>`_, `metadata <https://essentia.upf.edu/models/tempo/tempocnn/deepsquare-k16-3.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/tempo/tempocnn/deepsquare-k16-3_predictions.py

* .. collapse:: <a class="reference external">deeptemp-k4</a>

    [`weights <https://essentia.upf.edu/models/tempo/tempocnn/deeptemp-k4-3.pb>`_, `metadata <https://essentia.upf.edu/models/tempo/tempocnn/deeptemp-k4-3.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/tempo/tempocnn/deeptemp-k4-3_predictions.py

* .. collapse:: <a class="reference external">deeptemp-k16</a>

    [`weights <https://essentia.upf.edu/models/tempo/tempocnn/deeptemp-k16-3.pb>`_, `metadata <https://essentia.upf.edu/models/tempo/tempocnn/deeptemp-k16-3.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/tempo/tempocnn/deeptemp-k16-3_predictions.py



Classifiers
^^^^^^^^^^^

Classification and regression models.

Naming convention: ``<target_task>-<model>-<version>.pb``

* ``target_task``: the single-class, multi-class, or regression task to perform. See options below.
* ``model``: the model used to compute the input embeddings.
* ``version``: the model version.

Most of the classifiers are not standalone and require to pre-extract embeddings with the correspodent :ref:`embedding model<Feature extractors>`.

*Note: TensorflowPredict2D has to be configured with the correct output layer name for each classifier. Check the attached JSON file to find the name of the output layer on each case.*

Approachability
---------------

Music approachability predicting whether the music is likely to be accessible for the general public (e.g., belonging to common mainstream music genres vs. niche and experimental genres).
The models output rather two (``approachability_2c``) or three (``approachability_3c``) levels of approachability or continous values (``approachability_regression``).

Dataset: in-house (MTG).

Output: approachability predictions as class activations or regression values.

Models:

* .. collapse:: <a class="reference external">approachability_2c-effnet-discogs</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/approachability/approachability_2c-effnet-discogs-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/approachability/approachability_2c-effnet-discogs-1.json>`_, `demo <https://replicate.com/mtg/music-approachability-engagement>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/approachability/approachability_2c-effnet-discogs-1_predictions.py

* .. collapse:: <a class="reference external">approachability_3c-effnet-discogs</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/approachability/approachability_3c-effnet-discogs-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/approachability/approachability_3c-effnet-discogs-1.json>`_, `demo <https://replicate.com/mtg/music-approachability-engagement>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/approachability/approachability_3c-effnet-discogs-1_predictions.py

* .. collapse:: <a class="reference external">approachability_regression-effnet-discogs</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/approachability/approachability_regression-effnet-discogs-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/approachability/approachability_regression-effnet-discogs-1.json>`_, `demo <https://replicate.com/mtg/music-approachability-engagement>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/approachability/approachability_regression-effnet-discogs-1_predictions.py


Arousal/valence DEAM
--------------------

Music arousal and valence regression with the DEAM dataset:

`valence`, `arousal`

Dataset: `DEAM <https://cvml.unige.ch/databases/DEAM/>`_.

Output: arousal/valence predictions in the range [1,9].

Models:

* .. collapse:: <a class="reference external">deam-musicnn-msd</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/deam/deam-musicnn-msd-2.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/deam/deam-musicnn-msd-2.json>`_, `demo <https://replicate.com/mtg/music-arousal-valence>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/deam/deam-musicnn-msd-2_predictions.py

* .. collapse:: <a class="reference external">deam-vggish-audioset</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/deam/deam-vggish-audioset-2.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/deam/deam-vggish-audioset-2.json>`_, `demo <https://replicate.com/mtg/music-arousal-valence>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/deam/deam-vggish-audioset-2_predictions.py


Arousal/valence emoMusic
------------------------

Music arousal and valence regression with the emoMusic dataset:

`valence`, `arousal`

Dataset: `emoMusic <https://cvml.unige.ch/databases/emoMusic/>`_.

Output: arousal/valence predictions in the range [1,9].

Models:

* .. collapse:: <a class="reference external">emomusic-musicnn-msd</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/emomusic/emomusic-musicnn-msd-2.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/emomusic/emomusic-musicnn-msd-2.json>`_, `demo <https://replicate.com/mtg/music-arousal-valence>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/emomusic/emomusic-musicnn-msd-2_predictions.py

* .. collapse:: <a class="reference external">emomusic-vggish-audioset</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/emomusic/emomusic-vggish-audioset-2.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/emomusic/emomusic-vggish-audioset-2.json>`_, `demo <https://replicate.com/mtg/music-arousal-valence>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/emomusic/emomusic-vggish-audioset-2_predictions.py


Arousal/valence MuSe
--------------------

Music arousal and valence regression with the MuSe dataset.

`valence`, `arousal`

Dataset: `MuSE <https://aclanthology.org/2020.lrec-1.187/>`_.

Output: arousal/valence predictions in the range [1,9].

Models:

* .. collapse:: <a class="reference external">muse-musicnn-msd</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/muse/muse-musicnn-msd-2.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/muse/muse-musicnn-msd-2.json>`_, `demo <https://replicate.com/mtg/music-arousal-valence>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/muse/muse-musicnn-msd-2_predictions.py

* .. collapse:: <a class="reference external">muse-vggish-audioset</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/muse/muse-vggish-audioset-2.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/muse/muse-vggish-audioset-2.json>`_, `demo <https://replicate.com/mtg/music-arousal-valence>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/muse/muse-vggish-audioset-2_predictions.py


Engagement
----------

Music engagement predicting whether the music evokes active attention of the listener (high-engagement "lean forward" active listening vs. low-engagement "lean back" background listening).
The models output rather two  (``engagement_2c``) or three (``engagement_3c``) levels of engagement or continous (``engagement_regression``) values (regression).

Dataset: in-house (MTG).


Models:

* .. collapse:: <a class="reference external">engagement_2c-effnet-discogs</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/engagement/engagement_2c-effnet-discogs-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/engagement/engagement_2c-effnet-discogs-1.json>`_, `demo <https://replicate.com/mtg/music-approachability-engagement>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/engagement/engagement_2c-effnet-discogs-1_predictions.py

* .. collapse:: <a class="reference external">engagement_3c-effnet-discogs</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/engagement/engagement_3c-effnet-discogs-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/engagement/engagement_3c-effnet-discogs-1.json>`_, `demo <https://replicate.com/mtg/music-approachability-engagement>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/engagement/engagement_3c-effnet-discogs-1_predictions.py

* .. collapse:: <a class="reference external">engagement_regression-effnet-discogs</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/engagement/engagement_regression-effnet-discogs-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/engagement/engagement_regression-effnet-discogs-1.json>`_, `demo <https://replicate.com/mtg/music-approachability-engagement>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/engagement/engagement_regression-effnet-discogs-1_predictions.py


Free Music Archive small
------------------------

Music genre classfication (10 classes):

`Electronic`, `Experimental`, `Folk`, `Hip-Hop`, `Instrumental`, `International`, `Pop`, `Rock`

Dataset: `Free Music Archive small <https://www.kaggle.com/datasets/imsparsh/fma-free-music-archive-small-medium>`_.

Output: genre predictions.

Models:

* .. collapse:: <a class="reference external">fma_small-effnet-discogs_artist_embeddings</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/fma_small/fma_small-effnet-discogs_artist_embeddings-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/fma_small/fma_small-effnet-discogs_artist_embeddings-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/fma_small/fma_small-effnet-discogs_artist_embeddings-1_predictions.py

* .. collapse:: <a class="reference external">fma_small-effnet-discogs_label_embeddings</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/fma_small/fma_small-effnet-discogs_label_embeddings-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/fma_small/fma_small-effnet-discogs_label_embeddings-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/fma_small/fma_small-effnet-discogs_label_embeddings-1_predictions.py

* .. collapse:: <a class="reference external">fma_small-effnet-discogs_multi_embeddings</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/fma_small/fma_small-effnet-discogs_multi_embeddings-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/fma_small/fma_small-effnet-discogs_multi_embeddings-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/fma_small/fma_small-effnet-discogs_multi_embeddings-1_predictions.py

* .. collapse:: <a class="reference external">fma_small-effnet-discogs_release_embeddings</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/fma_small/fma_small-effnet-discogs_release_embeddings-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/fma_small/fma_small-effnet-discogs_release_embeddings-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/fma_small/fma_small-effnet-discogs_release_embeddings-1_predictions.py

* .. collapse:: <a class="reference external">fma_small-effnet-discogs_track_embeddings</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/fma_small/fma_small-effnet-discogs_track_embeddings-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/fma_small/fma_small-effnet-discogs_track_embeddings-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/fma_small/fma_small-effnet-discogs_track_embeddings-1_predictions.py


MTG-Jamendo genre
-----------------

Multi-label genre classification (87 classes):

`60s`, `70s`, `80s`, `90s`, `acidjazz`, `alternative`, `alternativerock`, `ambient`, `atmospheric`, `blues`, `bluesrock`, `bossanova`, `breakbeat`, `celtic`, `chanson`, `chillout`, `choir`, `classical`, `classicrock`, `club`, `contemporary`, `country`, `dance`, `darkambient`, `darkwave`, `deephouse`, `disco`, `downtempo`, `drumnbass`, `dub`, `dubstep`, `easylistening`, `edm`, `electronic`, `electronica`, `electropop`, `ethno`, `eurodance`, `experimental`, `folk`, `funk`, `fusion`, `groove`, `grunge`, `hard`, `hardrock`, `hiphop`, `house`, `idm`, `improvisation`, `indie`, `industrial`, `instrumentalpop`, `instrumentalrock`, `jazz`, `jazzfusion`, `latin`, `lounge`, `medieval`, `metal`, `minimal`, `newage`, `newwave`, `orchestral`, `pop`, `popfolk`, `poprock`, `postrock`, `progressive`, `psychedelic`, `punkrock`, `rap`, `reggae`, `rnb`, `rock`, `rocknroll`, `singersongwriter`, `soul`, `soundtrack`, `swing`, `symphonic`, `synthpop`, `techno`, `trance`, `triphop`, `world`, `worldfusion`

Dataset: MTG-Jamendo Dataset (genre subset).

Output: genre predictions.

Models:

* .. collapse:: <a class="reference external">mtg_jamendo_genre-effnet-discogs_artist_embeddings</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_genre/mtg_jamendo_genre-effnet-discogs_artist_embeddings-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_genre/mtg_jamendo_genre-effnet-discogs_artist_embeddings-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mtg_jamendo_genre/mtg_jamendo_genre-effnet-discogs_artist_embeddings-1_predictions.py

* .. collapse:: <a class="reference external">mtg_jamendo_genre-effnet-discogs_label_embeddings</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_genre/mtg_jamendo_genre-effnet-discogs_label_embeddings-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_genre/mtg_jamendo_genre-effnet-discogs_label_embeddings-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mtg_jamendo_genre/mtg_jamendo_genre-effnet-discogs_label_embeddings-1_predictions.py

* .. collapse:: <a class="reference external">mtg_jamendo_genre-effnet-discogs_multi_embeddings</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_genre/mtg_jamendo_genre-effnet-discogs_multi_embeddings-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_genre/mtg_jamendo_genre-effnet-discogs_multi_embeddings-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mtg_jamendo_genre/mtg_jamendo_genre-effnet-discogs_multi_embeddings-1_predictions.py

* .. collapse:: <a class="reference external">mtg_jamendo_genre-effnet-discogs_release_embeddings</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_genre/mtg_jamendo_genre-effnet-discogs_release_embeddings-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_genre/mtg_jamendo_genre-effnet-discogs_release_embeddings-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mtg_jamendo_genre/mtg_jamendo_genre-effnet-discogs_release_embeddings-1_predictions.py

* .. collapse:: <a class="reference external">mtg_jamendo_genre-effnet-discogs_track_embeddings</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_genre/mtg_jamendo_genre-effnet-discogs_track_embeddings-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_genre/mtg_jamendo_genre-effnet-discogs_track_embeddings-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mtg_jamendo_genre/mtg_jamendo_genre-effnet-discogs_track_embeddings-1_predictions.py


MTG-Jamendo instrument
----------------------

Multi-label instrument classification (40 classes):

`accordion`, `acousticbassguitar`, `acousticguitar`, `bass`, `beat`, `bell`, `bongo`, `brass`, `cello`, `clarinet`, `classicalguitar`, `computer`, `doublebass`, `drummachine`, `drums`, `electricguitar`, `electricpiano`, `flute`, `guitar`, `harmonica`, `harp`, `horn`, `keyboard`, `oboe`, `orchestra`, `organ`, `pad`, `percussion`, `piano`, `pipeorgan`, `rhodes`, `sampler`, `saxophone`, `strings`, `synthesizer`, `trombone`, `trumpet`, `viola`, `violin`, `voice`

Dataset: MTG-Jamendo Dataset (instrument subset).

Output: instrument class predictions.

Models:

* .. collapse:: <a class="reference external">mtg_jamendo_instrument-effnet-discogs_artist_embeddings</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-effnet-discogs_artist_embeddings-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-effnet-discogs_artist_embeddings-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-effnet-discogs_artist_embeddings-1_predictions.py

* .. collapse:: <a class="reference external">mtg_jamendo_instrument-effnet-discogs_label_embeddings</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-effnet-discogs_label_embeddings-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-effnet-discogs_label_embeddings-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-effnet-discogs_label_embeddings-1_predictions.py

* .. collapse:: <a class="reference external">mtg_jamendo_instrument-effnet-discogs_multi_embeddings</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-effnet-discogs_multi_embeddings-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-effnet-discogs_multi_embeddings-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-effnet-discogs_multi_embeddings-1_predictions.py

* .. collapse:: <a class="reference external">mtg_jamendo_instrument-effnet-discogs_release_embeddings</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-effnet-discogs_release_embeddings-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-effnet-discogs_release_embeddings-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-effnet-discogs_release_embeddings-1_predictions.py

* .. collapse:: <a class="reference external">mtg_jamendo_instrument-effnet-discogs_track_embeddings</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-effnet-discogs_track_embeddings-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-effnet-discogs_track_embeddings-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-effnet-discogs_track_embeddings-1_predictions.py


MTG-Jamendo moodtheme
---------------------

Multi-label mood/theme classification (56 classes):

`action`, `adventure`, `advertising`, `background`, `ballad`, `calm`, `children`, `christmas`, `commercial`, `cool`, `corporate`, `dark`, `deep`, `documentary`, `drama`, `dramatic`, `dream`, `emotional`, `energetic`, `epic`, `fast`, `film`, `fun`, `funny`, `game`, `groovy`, `happy`, `heavy`, `holiday`, `hopeful`, `inspiring`, `love`, `meditative`, `melancholic`, `melodic`, `motivational`, `movie`, `nature`, `party`, `positive`, `powerful`, `relaxing`, `retro`, `romantic`, `sad`, `sexy`, `slow`, `soft`, `soundscape`, `space`, `sport`, `summer`, `trailer`, `travel`, `upbeat`, `uplifting`

Dataset: MTG-Jamendo Dataset (moodtheme subset).

Output: mood/theme predictions.

Models:

* .. collapse:: <a class="reference external">mtg_jamendo_moodtheme-effnet-discogs_artist_embeddings</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-effnet-discogs_artist_embeddings-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-effnet-discogs_artist_embeddings-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-effnet-discogs_artist_embeddings-1_predictions.py

* .. collapse:: <a class="reference external">mtg_jamendo_moodtheme-effnet-discogs_label_embeddings</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-effnet-discogs_label_embeddings-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-effnet-discogs_label_embeddings-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-effnet-discogs_label_embeddings-1_predictions.py

* .. collapse:: <a class="reference external">mtg_jamendo_moodtheme-effnet-discogs_multi_embeddings</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-effnet-discogs_multi_embeddings-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-effnet-discogs_multi_embeddings-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-effnet-discogs_multi_embeddings-1_predictions.py

* .. collapse:: <a class="reference external">mtg_jamendo_moodtheme-effnet-discogs_release_embeddings</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-effnet-discogs_release_embeddings-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-effnet-discogs_release_embeddings-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-effnet-discogs_release_embeddings-1_predictions.py

* .. collapse:: <a class="reference external">mtg_jamendo_moodtheme-effnet-discogs_track_embeddings</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-effnet-discogs_track_embeddings-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-effnet-discogs_track_embeddings-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-effnet-discogs_track_embeddings-1_predictions.py


MTG-Jamendo top50tags
---------------------

Auto-tagging with top-50 MTG-Jamendo classes:

`alternative`, `ambient`, `atmospheric`, `chillout`, `classical`, `dance`, `downtempo`, `easylistening`, `electronic`, `experimental`, `folk`, `funk`, `hiphop`, `house`, `indie`, `instrumentalpop`, `jazz`, `lounge`, `metal`, `newage`, `orchestral`, `pop`, `popfolk`, `poprock`, `reggae`, `rock`, `soundtrack`, `techno`, `trance`, `triphop`, `world`, `acousticguitar`, `bass`, `computer`, `drummachine`, `drums`, `electricguitar`, `electricpiano`, `guitar`, `keyboard`, `piano`, `strings`, `synthesizer`, `violin`, `voice`, `emotional`, `energetic`, `film`, `happy`, `relaxing`

Dataset: MTG-Jamendo Dataset (top50tags subset).

Output: top-50 tag predictions.

Models:

* .. collapse:: <a class="reference external">mtg_jamendo_top50tags-effnet-discogs_artist_embeddings</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_top50tags/mtg_jamendo_top50tags-effnet-discogs_artist_embeddings-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_top50tags/mtg_jamendo_top50tags-effnet-discogs_artist_embeddings-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mtg_jamendo_top50tags/mtg_jamendo_top50tags-effnet-discogs_artist_embeddings-1_predictions.py

* .. collapse:: <a class="reference external">mtg_jamendo_top50tags-effnet-discogs_label_embeddings</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_top50tags/mtg_jamendo_top50tags-effnet-discogs_label_embeddings-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_top50tags/mtg_jamendo_top50tags-effnet-discogs_label_embeddings-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mtg_jamendo_top50tags/mtg_jamendo_top50tags-effnet-discogs_label_embeddings-1_predictions.py

* .. collapse:: <a class="reference external">mtg_jamendo_top50tags-effnet-discogs_multi_embeddings</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_top50tags/mtg_jamendo_top50tags-effnet-discogs_multi_embeddings-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_top50tags/mtg_jamendo_top50tags-effnet-discogs_multi_embeddings-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mtg_jamendo_top50tags/mtg_jamendo_top50tags-effnet-discogs_multi_embeddings-1_predictions.py

* .. collapse:: <a class="reference external">mtg_jamendo_top50tags-effnet-discogs_release_embeddings</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_top50tags/mtg_jamendo_top50tags-effnet-discogs_release_embeddings-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_top50tags/mtg_jamendo_top50tags-effnet-discogs_release_embeddings-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mtg_jamendo_top50tags/mtg_jamendo_top50tags-effnet-discogs_release_embeddings-1_predictions.py

* .. collapse:: <a class="reference external">mtg_jamendo_top50tags-effnet-discogs_track_embeddings</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_top50tags/mtg_jamendo_top50tags-effnet-discogs_track_embeddings-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mtg_jamendo_top50tags/mtg_jamendo_top50tags-effnet-discogs_track_embeddings-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mtg_jamendo_top50tags/mtg_jamendo_top50tags-effnet-discogs_track_embeddings-1_predictions.py


MagnaTagATune
-------------

Auto-tagging with the top-50 MagnaTagATune classes:

`ambient`, `beat`, `beats`, `cello`, `choir`, `choral`, `classic`, `classical`, `country`, `dance`, `drums`, `electronic`, `fast`, `female`, `female vocal`, `female voice`, `flute`, `guitar`, `harp`, `harpsichord`, `indian`, `loud`, `male`, `male vocal`, `male voice`, `man`, `metal`, `new age`, `no vocal`, `no vocals`, `no voice`, `opera`, `piano`, `pop`, `quiet`, `rock`, `singing`, `sitar`, `slow`, `soft`, `solo`, `strings`, `synth`, `techno`, `violin`, `vocal`, `vocals`, `voice`, `weird`, `woman`

Dataset: MagnaTagATune.

Output: auto-tagging predictions.

Models:

* .. collapse:: <a class="reference external">mtt-effnet-discogs_artist_embeddings</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mtt/mtt-effnet-discogs_artist_embeddings-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mtt/mtt-effnet-discogs_artist_embeddings-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mtt/mtt-effnet-discogs_artist_embeddings-1_predictions.py

* .. collapse:: <a class="reference external">mtt-effnet-discogs_label_embeddings</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mtt/mtt-effnet-discogs_label_embeddings-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mtt/mtt-effnet-discogs_label_embeddings-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mtt/mtt-effnet-discogs_label_embeddings-1_predictions.py

* .. collapse:: <a class="reference external">mtt-effnet-discogs_multi_embeddings</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mtt/mtt-effnet-discogs_multi_embeddings-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mtt/mtt-effnet-discogs_multi_embeddings-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mtt/mtt-effnet-discogs_multi_embeddings-1_predictions.py

* .. collapse:: <a class="reference external">mtt-effnet-discogs_release_embeddings</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mtt/mtt-effnet-discogs_release_embeddings-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mtt/mtt-effnet-discogs_release_embeddings-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mtt/mtt-effnet-discogs_release_embeddings-1_predictions.py

* .. collapse:: <a class="reference external">mtt-effnet-discogs_track_embeddings</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mtt/mtt-effnet-discogs_track_embeddings-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mtt/mtt-effnet-discogs_track_embeddings-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mtt/mtt-effnet-discogs_track_embeddings-1_predictions.py

* .. collapse:: <a class="reference external">mtt-musicnn</a>

    [`weights <https://essentia.upf.edu/models/autotagging/mtt/mtt-musicnn-1.pb>`_, `metadata <https://essentia.upf.edu/models/autotagging/mtt/mtt-musicnn-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/autotagging/mtt/mtt-musicnn-1_predictions.py

Danceability
------------

Music danceability (2 classes):

`danceable`, `not_danceable`

Dataset: in-house (MTG).

Output: danceability predictions.

Models:

* .. collapse:: <a class="reference external">danceability-audioset-vggish</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/danceability/danceability-audioset-vggish-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/danceability/danceability-audioset-vggish-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/danceability/danceability-audioset-vggish-1_predictions.py

* .. collapse:: <a class="reference external">danceability-audioset-yamnet</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/danceability/danceability-audioset-yamnet-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/danceability/danceability-audioset-yamnet-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/danceability/danceability-audioset-yamnet-1_predictions.py

* .. collapse:: <a class="reference external">danceability-effnet-discogs</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/danceability/danceability-effnet-discogs-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/danceability/danceability-effnet-discogs-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/danceability/danceability-effnet-discogs-1_predictions.py

* .. collapse:: <a class="reference external">danceability-msd-musicnn</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/danceability/danceability-msd-musicnn-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/danceability/danceability-msd-musicnn-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/danceability/danceability-msd-musicnn-1_predictions.py

* .. collapse:: <a class="reference external">danceability-openl3-music-mel128-emb512</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/danceability/danceability-openl3-music-mel128-emb512-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/danceability/danceability-openl3-music-mel128-emb512-1.json>`_]

    We do not have a dedicated algorithm to extract embeddings with this model. For now, OpenL3 embeddings can be extracted using this `script <https://gist.github.com/palonso/cfebe37e5492b5a3a31775d8eae8d9a8>`_.


Voice / Instrumental
--------------------

Classification of music by presence or absence of voice (2 classes):

`instrumental`, `voice`

Dataset: in-house (MTG).

Output: voice / instrumental predictions.

Models:

* .. collapse:: <a class="reference external">voice_instrumental-audioset-vggish</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-audioset-vggish-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-audioset-vggish-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/voice_instrumental/voice_instrumental-audioset-vggish-1_predictions.py

* .. collapse:: <a class="reference external">voice_instrumental-audioset-yamnet</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-audioset-yamnet-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-audioset-yamnet-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/voice_instrumental/voice_instrumental-audioset-yamnet-1_predictions.py

* .. collapse:: <a class="reference external">voice_instrumental-effnet-discogs</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-effnet-discogs-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-effnet-discogs-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/voice_instrumental/voice_instrumental-effnet-discogs-1_predictions.py

* .. collapse:: <a class="reference external">voice_instrumental-msd-musicnn</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-msd-musicnn-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-msd-musicnn-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/voice_instrumental/voice_instrumental-msd-musicnn-1_predictions.py

* .. collapse:: <a class="reference external">voice_instrumental-openl3-music-mel128-emb512</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-openl3-music-mel128-emb512-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-openl3-music-mel128-emb512-1.json>`_]

    We do not have a dedicated algorithm to extract embeddings with this model. For now, OpenL3 embeddings can be extracted using this `script <https://gist.github.com/palonso/cfebe37e5492b5a3a31775d8eae8d9a8>`_.


Gender
------

Classification of music by singing voice gender (2 classes):

`female`, `male`

Dataset: in-house (MTG).

Output: singing voice gender predictions.

Models:

* .. collapse:: <a class="reference external">gender-audioset-vggish</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/gender/gender-audioset-vggish-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/gender/gender-audioset-vggish-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/gender/gender-audioset-vggish-1_predictions.py

* .. collapse:: <a class="reference external">gender-audioset-yamnet</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/gender/gender-audioset-yamnet-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/gender/gender-audioset-yamnet-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/gender/gender-audioset-yamnet-1_predictions.py

* .. collapse:: <a class="reference external">gender-effnet-discogs</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/gender/gender-effnet-discogs-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/gender/gender-effnet-discogs-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/gender/gender-effnet-discogs-1_predictions.py

* .. collapse:: <a class="reference external">gender-msd-musicnn</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/gender/gender-msd-musicnn-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/gender/gender-msd-musicnn-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/gender/gender-msd-musicnn-1_predictions.py

* .. collapse:: <a class="reference external">gender-openl3-music-mel128-emb512</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/gender/gender-openl3-music-mel128-emb512-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/gender/gender-openl3-music-mel128-emb512-1.json>`_]

    We do not have a dedicated algorithm to extract embeddings with this model. For now, OpenL3 embeddings can be extracted using this `script <https://gist.github.com/palonso/cfebe37e5492b5a3a31775d8eae8d9a8>`_.


Genre Dortmund
--------------

Music genre classification (9 genres):

`alternative`, `blues`, `electronic`, `folkcountry`, `funksoulrnb`, `jazz`, `pop`, `raphiphop`, `rock`

Dataset: Music Audio Benchmark Data Set.

Output: genre predictions.

Models:

* .. collapse:: <a class="reference external">genre_dortmund-audioset-vggish</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/genre_dortmund/genre_dortmund-audioset-vggish-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/genre_dortmund/genre_dortmund-audioset-vggish-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/genre_dortmund/genre_dortmund-audioset-vggish-1_predictions.py

* .. collapse:: <a class="reference external">genre_dortmund-audioset-yamnet</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/genre_dortmund/genre_dortmund-audioset-yamnet-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/genre_dortmund/genre_dortmund-audioset-yamnet-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/genre_dortmund/genre_dortmund-audioset-yamnet-1_predictions.py

* .. collapse:: <a class="reference external">genre_dortmund-effnet-discogs</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/genre_dortmund/genre_dortmund-effnet-discogs-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/genre_dortmund/genre_dortmund-effnet-discogs-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/genre_dortmund/genre_dortmund-effnet-discogs-1_predictions.py

* .. collapse:: <a class="reference external">genre_dortmund-msd-musicnn</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/genre_dortmund/genre_dortmund-msd-musicnn-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/genre_dortmund/genre_dortmund-msd-musicnn-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/genre_dortmund/genre_dortmund-msd-musicnn-1_predictions.py

* .. collapse:: <a class="reference external">genre_dortmund-openl3-music-mel128-emb512</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/genre_dortmund/genre_dortmund-openl3-music-mel128-emb512-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/genre_dortmund/genre_dortmund-openl3-music-mel128-emb512-1.json>`_]

    We do not have a dedicated algorithm to extract embeddings with this model. For now, OpenL3 embeddings can be extracted using this `script <https://gist.github.com/palonso/cfebe37e5492b5a3a31775d8eae8d9a8>`_.


Genre Electronic
----------------

Electronic music genre classification (5 genres):

`ambient`, `dnb`, `house`, `techno`, `trance`

Dataset: in-house (MTG).

Output: genre predictions.

Models:

* .. collapse:: <a class="reference external">genre_electronic-effnet-discogs</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/genre_electronic/genre_electronic-effnet-discogs-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/genre_electronic/genre_electronic-effnet-discogs-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/genre_electronic/genre_electronic-effnet-discogs-1_predictions.py


Genre Rosamerica
----------------

Music genre classification (8 genres):

`classical`, `dance`, `hip hop`, `jazz`, `pop`, `rhythm and blues`, `rock`, `speech`

Dataset: in-house (MTG).

Output: genre predictions.

Models:

* .. collapse:: <a class="reference external">genre_rosamerica-audioset-vggish</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/genre_rosamerica/genre_rosamerica-audioset-vggish-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/genre_rosamerica/genre_rosamerica-audioset-vggish-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/genre_rosamerica/genre_rosamerica-audioset-vggish-1_predictions.py

* .. collapse:: <a class="reference external">genre_rosamerica-audioset-yamnet</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/genre_rosamerica/genre_rosamerica-audioset-yamnet-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/genre_rosamerica/genre_rosamerica-audioset-yamnet-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/genre_rosamerica/genre_rosamerica-audioset-yamnet-1_predictions.py

* .. collapse:: <a class="reference external">genre_rosamerica-effnet-discogs</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/genre_rosamerica/genre_rosamerica-effnet-discogs-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/genre_rosamerica/genre_rosamerica-effnet-discogs-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/genre_rosamerica/genre_rosamerica-effnet-discogs-1_predictions.py

* .. collapse:: <a class="reference external">genre_rosamerica-msd-musicnn</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/genre_rosamerica/genre_rosamerica-msd-musicnn-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/genre_rosamerica/genre_rosamerica-msd-musicnn-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/genre_rosamerica/genre_rosamerica-msd-musicnn-1_predictions.py

* .. collapse:: <a class="reference external">genre_rosamerica-openl3-music-mel128-emb512</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/genre_rosamerica/genre_rosamerica-openl3-music-mel128-emb512-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/genre_rosamerica/genre_rosamerica-openl3-music-mel128-emb512-1.json>`_]

    We do not have a dedicated algorithm to extract embeddings with this model. For now, OpenL3 embeddings can be extracted using this `script <https://gist.github.com/palonso/cfebe37e5492b5a3a31775d8eae8d9a8>`_.


Genre Tzanetakis
----------------

Music genre classification (10 genres):

`blues`, `classic`, `country`, `disco`, `hip hop`, `jazz`, `metal`, `pop`, `reggae`, `rock`

Dataset: in-house (MTG).

Output: genre predictions.

Models:

* .. collapse:: <a class="reference external">genre_tzanetakis-audioset-vggish</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/genre_tzanetakis/genre_tzanetakis-audioset-vggish-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/genre_tzanetakis/genre_tzanetakis-audioset-vggish-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/genre_tzanetakis/genre_tzanetakis-audioset-vggish-1_predictions.py

* .. collapse:: <a class="reference external">genre_tzanetakis-audioset-yamnet</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/genre_tzanetakis/genre_tzanetakis-audioset-yamnet-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/genre_tzanetakis/genre_tzanetakis-audioset-yamnet-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/genre_tzanetakis/genre_tzanetakis-audioset-yamnet-1_predictions.py

* .. collapse:: <a class="reference external">genre_tzanetakis-effnet-discogs</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/genre_tzanetakis/genre_tzanetakis-effnet-discogs-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/genre_tzanetakis/genre_tzanetakis-effnet-discogs-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/genre_tzanetakis/genre_tzanetakis-effnet-discogs-1_predictions.py

* .. collapse:: <a class="reference external">genre_tzanetakis-msd-musicnn</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/genre_tzanetakis/genre_tzanetakis-msd-musicnn-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/genre_tzanetakis/genre_tzanetakis-msd-musicnn-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/genre_tzanetakis/genre_tzanetakis-msd-musicnn-1_predictions.py

* .. collapse:: <a class="reference external">genre_tzanetakis-openl3-music-mel128-emb512</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/genre_tzanetakis/genre_tzanetakis-openl3-music-mel128-emb512-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/genre_tzanetakis/genre_tzanetakis-openl3-music-mel128-emb512-1.json>`_]

    We do not have a dedicated algorithm to extract embeddings with this model. For now, OpenL3 embeddings can be extracted using this `script <https://gist.github.com/palonso/cfebe37e5492b5a3a31775d8eae8d9a8>`_.


Mood Acoustic
-------------

Music classification by type of sound (2 classes):

`acoustic`, `non_acoustic`

Dataset: in-house (MTG).

Output: mood acoustic predictions.

Models:

* .. collapse:: <a class="reference external">mood_acoustic-audioset-vggish</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_acoustic/mood_acoustic-audioset-vggish-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_acoustic/mood_acoustic-audioset-vggish-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mood_acoustic/mood_acoustic-audioset-vggish-1_predictions.py

* .. collapse:: <a class="reference external">mood_acoustic-audioset-yamnet</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_acoustic/mood_acoustic-audioset-yamnet-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_acoustic/mood_acoustic-audioset-yamnet-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mood_acoustic/mood_acoustic-audioset-yamnet-1_predictions.py

* .. collapse:: <a class="reference external">mood_acoustic-effnet-discogs</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_acoustic/mood_acoustic-effnet-discogs-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_acoustic/mood_acoustic-effnet-discogs-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mood_acoustic/mood_acoustic-effnet-discogs-1_predictions.py

* .. collapse:: <a class="reference external">mood_acoustic-msd-musicnn</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_acoustic/mood_acoustic-msd-musicnn-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_acoustic/mood_acoustic-msd-musicnn-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mood_acoustic/mood_acoustic-msd-musicnn-1_predictions.py

* .. collapse:: <a class="reference external">mood_acoustic-openl3-music-mel128-emb512</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_acoustic/mood_acoustic-openl3-music-mel128-emb512-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_acoustic/mood_acoustic-openl3-music-mel128-emb512-1.json>`_]

    We do not have a dedicated algorithm to extract embeddings with this model. For now, OpenL3 embeddings can be extracted using this `script <https://gist.github.com/palonso/cfebe37e5492b5a3a31775d8eae8d9a8>`_.


Mood Aggressive
---------------

Music classification by mood (2 classes):

`aggressive`, `non_aggressive`

Dataset: in-house (MTG).

Output: mood aggressive predictions.

Models:

* .. collapse:: <a class="reference external">mood_aggressive-audioset-vggish</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_aggressive/mood_aggressive-audioset-vggish-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_aggressive/mood_aggressive-audioset-vggish-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mood_aggressive/mood_aggressive-audioset-vggish-1_predictions.py

* .. collapse:: <a class="reference external">mood_aggressive-audioset-yamnet</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_aggressive/mood_aggressive-audioset-yamnet-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_aggressive/mood_aggressive-audioset-yamnet-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mood_aggressive/mood_aggressive-audioset-yamnet-1_predictions.py

* .. collapse:: <a class="reference external">mood_aggressive-effnet-discogs</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_aggressive/mood_aggressive-effnet-discogs-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_aggressive/mood_aggressive-effnet-discogs-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mood_aggressive/mood_aggressive-effnet-discogs-1_predictions.py

* .. collapse:: <a class="reference external">mood_aggressive-msd-musicnn</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_aggressive/mood_aggressive-msd-musicnn-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_aggressive/mood_aggressive-msd-musicnn-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mood_aggressive/mood_aggressive-msd-musicnn-1_predictions.py

* .. collapse:: <a class="reference external">mood_aggressive-openl3-music-mel128-emb512</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_aggressive/mood_aggressive-openl3-music-mel128-emb512-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_aggressive/mood_aggressive-openl3-music-mel128-emb512-1.json>`_]

    We do not have a dedicated algorithm to extract embeddings with this model. For now, OpenL3 embeddings can be extracted using this `script <https://gist.github.com/palonso/cfebe37e5492b5a3a31775d8eae8d9a8>`_.


Mood Electronic
---------------

Music classification by type of sound (2 classes):

`electronic`, `non_electronic`

Dataset: in-house (MTG).

Output: mood electronic predictions.

Models:

* .. collapse:: <a class="reference external">mood_electronic-audioset-vggish</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_electronic/mood_electronic-audioset-vggish-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_electronic/mood_electronic-audioset-vggish-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mood_electronic/mood_electronic-audioset-vggish-1_predictions.py

* .. collapse:: <a class="reference external">mood_electronic-audioset-yamnet</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_electronic/mood_electronic-audioset-yamnet-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_electronic/mood_electronic-audioset-yamnet-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mood_electronic/mood_electronic-audioset-yamnet-1_predictions.py

* .. collapse:: <a class="reference external">mood_electronic-effnet-discogs</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_electronic/mood_electronic-effnet-discogs-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_electronic/mood_electronic-effnet-discogs-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mood_electronic/mood_electronic-effnet-discogs-1_predictions.py

* .. collapse:: <a class="reference external">mood_electronic-msd-musicnn</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_electronic/mood_electronic-msd-musicnn-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_electronic/mood_electronic-msd-musicnn-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mood_electronic/mood_electronic-msd-musicnn-1_predictions.py

* .. collapse:: <a class="reference external">mood_electronic-openl3-music-mel128-emb512</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_electronic/mood_electronic-openl3-music-mel128-emb512-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_electronic/mood_electronic-openl3-music-mel128-emb512-1.json>`_]

    We do not have a dedicated algorithm to extract embeddings with this model. For now, OpenL3 embeddings can be extracted using this `script <https://gist.github.com/palonso/cfebe37e5492b5a3a31775d8eae8d9a8>`_.


Mood Happy
----------

Music classification by mood (2 classes):

`happy`, `non_happy`

Dataset: in-house (MTG).

Output: mood happy predictions.

Models:

* .. collapse:: <a class="reference external">mood_happy-audioset-vggish</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_happy/mood_happy-audioset-vggish-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_happy/mood_happy-audioset-vggish-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mood_happy/mood_happy-audioset-vggish-1_predictions.py

* .. collapse:: <a class="reference external">mood_happy-audioset-yamnet</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_happy/mood_happy-audioset-yamnet-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_happy/mood_happy-audioset-yamnet-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mood_happy/mood_happy-audioset-yamnet-1_predictions.py

* .. collapse:: <a class="reference external">mood_happy-effnet-discogs</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_happy/mood_happy-effnet-discogs-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_happy/mood_happy-effnet-discogs-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mood_happy/mood_happy-effnet-discogs-1_predictions.py

* .. collapse:: <a class="reference external">mood_happy-msd-musicnn</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_happy/mood_happy-msd-musicnn-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_happy/mood_happy-msd-musicnn-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mood_happy/mood_happy-msd-musicnn-1_predictions.py

* .. collapse:: <a class="reference external">mood_happy-openl3-music-mel128-emb512</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_happy/mood_happy-openl3-music-mel128-emb512-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_happy/mood_happy-openl3-music-mel128-emb512-1.json>`_]

    We do not have a dedicated algorithm to extract embeddings with this model. For now, OpenL3 embeddings can be extracted using this `script <https://gist.github.com/palonso/cfebe37e5492b5a3a31775d8eae8d9a8>`_.


Mood Party
----------

Music classification by mood (2 classes):

`party`, `non_party`

Dataset: in-house (MTG).

Output: mood pary predictions.

Models:

* .. collapse:: <a class="reference external">mood_party-audioset-vggish</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_party/mood_party-audioset-vggish-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_party/mood_party-audioset-vggish-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mood_party/mood_party-audioset-vggish-1_predictions.py

* .. collapse:: <a class="reference external">mood_party-audioset-yamnet</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_party/mood_party-audioset-yamnet-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_party/mood_party-audioset-yamnet-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mood_party/mood_party-audioset-yamnet-1_predictions.py

* .. collapse:: <a class="reference external">mood_party-effnet-discogs</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_party/mood_party-effnet-discogs-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_party/mood_party-effnet-discogs-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mood_party/mood_party-effnet-discogs-1_predictions.py

* .. collapse:: <a class="reference external">mood_party-msd-musicnn</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_party/mood_party-msd-musicnn-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_party/mood_party-msd-musicnn-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mood_party/mood_party-msd-musicnn-1_predictions.py

* .. collapse:: <a class="reference external">mood_party-openl3-music-mel128-emb512</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_party/mood_party-openl3-music-mel128-emb512-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_party/mood_party-openl3-music-mel128-emb512-1.json>`_]

    We do not have a dedicated algorithm to extract embeddings with this model. For now, OpenL3 embeddings can be extracted using this `script <https://gist.github.com/palonso/cfebe37e5492b5a3a31775d8eae8d9a8>`_.


Mood Relaxed
------------

Music classification by mood (2 classes):

`relaxed`, `non_relaxed`

Dataset: in-house (MTG).

Output: moosd relaxed predictions.

Models:

* .. collapse:: <a class="reference external">mood_relaxed-audioset-vggish</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_relaxed/mood_relaxed-audioset-vggish-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_relaxed/mood_relaxed-audioset-vggish-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mood_relaxed/mood_relaxed-audioset-vggish-1_predictions.py

* .. collapse:: <a class="reference external">mood_relaxed-audioset-yamnet</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_relaxed/mood_relaxed-audioset-yamnet-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_relaxed/mood_relaxed-audioset-yamnet-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mood_relaxed/mood_relaxed-audioset-yamnet-1_predictions.py

* .. collapse:: <a class="reference external">mood_relaxed-effnet-discogs</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_relaxed/mood_relaxed-effnet-discogs-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_relaxed/mood_relaxed-effnet-discogs-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mood_relaxed/mood_relaxed-effnet-discogs-1_predictions.py

* .. collapse:: <a class="reference external">mood_relaxed-msd-musicnn</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_relaxed/mood_relaxed-msd-musicnn-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_relaxed/mood_relaxed-msd-musicnn-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mood_relaxed/mood_relaxed-msd-musicnn-1_predictions.py

* .. collapse:: <a class="reference external">mood_relaxed-openl3-music-mel128-emb512</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_relaxed/mood_relaxed-openl3-music-mel128-emb512-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_relaxed/mood_relaxed-openl3-music-mel128-emb512-1.json>`_]

    We do not have a dedicated algorithm to extract embeddings with this model. For now, OpenL3 embeddings can be extracted using this `script <https://gist.github.com/palonso/cfebe37e5492b5a3a31775d8eae8d9a8>`_.


Mood Sad
--------

Music classification by mood (2 classes):

`sad`, `non_sad`

Dataset: in-house (MTG).

Output: mood sad predictions.

Models:

* .. collapse:: <a class="reference external">mood_sad-audioset-vggish</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_sad/mood_sad-audioset-vggish-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_sad/mood_sad-audioset-vggish-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mood_sad/mood_sad-audioset-vggish-1_predictions.py

* .. collapse:: <a class="reference external">mood_sad-audioset-yamnet</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_sad/mood_sad-audioset-yamnet-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_sad/mood_sad-audioset-yamnet-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mood_sad/mood_sad-audioset-yamnet-1_predictions.py

* .. collapse:: <a class="reference external">mood_sad-effnet-discogs</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_sad/mood_sad-effnet-discogs-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_sad/mood_sad-effnet-discogs-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mood_sad/mood_sad-effnet-discogs-1_predictions.py

* .. collapse:: <a class="reference external">mood_sad-msd-musicnn</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_sad/mood_sad-msd-musicnn-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_sad/mood_sad-msd-musicnn-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/mood_sad/mood_sad-msd-musicnn-1_predictions.py

* .. collapse:: <a class="reference external">mood_sad-openl3-music-mel128-emb512</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/mood_sad/mood_sad-openl3-music-mel128-emb512-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/mood_sad/mood_sad-openl3-music-mel128-emb512-1.json>`_]

    We do not have a dedicated algorithm to extract embeddings with this model. For now, OpenL3 embeddings can be extracted using this `script <https://gist.github.com/palonso/cfebe37e5492b5a3a31775d8eae8d9a8>`_.


Timbre
------

Classification of music by timbre color (dark/bright timbre):

`bright`, `dark`

Dataset: in-house (MTG).

Output: timbre predictions.

Models:

* .. collapse:: <a class="reference external">timbre-effnet-discogs</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/timbre/timbre-effnet-discogs-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/timbre/timbre-effnet-discogs-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/timbre/timbre-effnet-discogs-1_predictions.py


Tonal / Atonal
--------------

Music classification by tonality (2 classes):

`tonal`, `atonal`

Dataset: in-house (MTG).

Output: tonal / atonal predictions.

Models:

* .. collapse:: <a class="reference external">tonal_atonal-audioset-vggish</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/tonal_atonal/tonal_atonal-audioset-vggish-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/tonal_atonal/tonal_atonal-audioset-vggish-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/tonal_atonal/tonal_atonal-audioset-vggish-1_predictions.py

* .. collapse:: <a class="reference external">tonal_atonal-audioset-yamnet</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/tonal_atonal/tonal_atonal-audioset-yamnet-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/tonal_atonal/tonal_atonal-audioset-yamnet-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/tonal_atonal/tonal_atonal-audioset-yamnet-1_predictions.py

* .. collapse:: <a class="reference external">tonal_atonal-effnet-discogs</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/tonal_atonal/tonal_atonal-effnet-discogs-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/tonal_atonal/tonal_atonal-effnet-discogs-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/tonal_atonal/tonal_atonal-effnet-discogs-1_predictions.py

* .. collapse:: <a class="reference external">tonal_atonal-msd-musicnn</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/tonal_atonal/tonal_atonal-msd-musicnn-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/tonal_atonal/tonal_atonal-msd-musicnn-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/classification-heads/tonal_atonal/tonal_atonal-msd-musicnn-1_predictions.py

* .. collapse:: <a class="reference external">tonal_atonal-openl3-music-mel128-emb512</a>

    [`weights <https://essentia.upf.edu/models/classification-heads/tonal_atonal/tonal_atonal-openl3-music-mel128-emb512-1.pb>`_, `metadata <https://essentia.upf.edu/models/classification-heads/tonal_atonal/tonal_atonal-openl3-music-mel128-emb512-1.json>`_]

    We do not have a dedicated algorithm to extract embeddings with this model. For now, OpenL3 embeddings can be extracted using this `script <https://gist.github.com/palonso/cfebe37e5492b5a3a31775d8eae8d9a8>`_.

Million Song Dataset
--------------------

Music auto-tagging with 50 common music tags:

`rock`, `pop`, `alternative`, `indie`, `electronic`, `female vocalists`, `dance`, `00s`, `alternative rock`, `jazz`, `beautiful`, `metal`, `chillout`, `male vocalists`, `classic rock`, `soul`, `indie rock`, `Mellow`, `electronica`, `80s`, `folk`, `90s`, `chill`, `instrumental`, `punk`, `oldies`, `blues`, `hard rock`, `ambient`, `acoustic`, `experimental`, `female vocalist`, `guitar`, `Hip-Hop`, `70s`, `party`, `country`, `easy listening`, `sexy`, `catchy`, `funk`, `electro`, `heavy metal`, `Progressive rock`, `60s`, `rnb`, `indie pop`, `sad`, `House`, `happy`

Dataset: Million Song Dataset.

Outputs: Auto-tagging with the top-50 Million Song Dataset classes.


Models:

* .. collapse:: <a class="reference external">msd-musicnn</a>

    [`weights <https://essentia.upf.edu/models/autotagging/msd/msd-musicnn-1.pb>`_, `metadata <https://essentia.upf.edu/models/autotagging/msd/msd-musicnn-1.json>`_]

    Python code for predictions:

    .. literalinclude :: ../../src/examples/python/models/scripts/autotagging/msd/msd-musicnn-1_predictions.py

