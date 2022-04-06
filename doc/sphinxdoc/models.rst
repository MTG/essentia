.. Essentia models

Essentia models
===============

This is a list of pre-trained TensorFlow models available in Essentia for various (music) audio analysis and classification tasks. To use the models in Essentia, see the `machine learning inference <machine_learning.html>`_ page. Some of the models are also available in other formats (ONNX, TensorFlow.js; contact us for more details).


All the models created by the MTG are licensed under CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/) and are also available under proprietary license upon request (https://www.upf.edu/web/mtg/contact). Check the `LICENSE <https://essentia.upf.edu/models/LICENSE>`_ of the models.


The models are serialized in `Protocol Buffer <https://developers.google.com/protocol-buffers/>`_ files suitable for inference with Essentia and TensorFlow. Additionally, some models are also available in `TensorFlow.js <https://www.tensorflow.org/js/models>`_ format. Each ``.pb`` file is coupled with a ``.json`` file containing its metadata.

As this is an ongoing project we expect to keep adding new models and improved versions of the existing ones. These changes are tracked in this `CHANGELOG <https://essentia.upf.edu/models/CHANGELOG.md>`_.


We support models for the followings tasks:

* :ref:`Audio event recognition`
* :ref:`Music style classification`
* :ref:`Music auto-tagging (genre, mood, epoch, instrumentation, etc.)<Music auto-tagging>`
* :ref:`Transfer learning classifiers (genre, mood, danceability, voice, instrumentation, etc.)<Transfer learning classifiers>`
* :ref:`Feature extractors`
* :ref:`Pitch detection`
* :ref:`Source separation`
* :ref:`Tempo estimation`


Audio event recognition
^^^^^^^^^^^^^^^^^^^^^^^
`Download model files <https://essentia.upf.edu/models/audio-event-recognition/>`_

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

Output: activations.

This model is useful for audio embeddings.

Models:

* ``audioset-yamnet``

Naming convention: ``<task>-<architecture>-<version>.pb``

Usage for audio event detection:

.. code-block:: python

    from essentia.standard import MonoLoader, TensorflowPredictVGGish

    audio = MonoLoader(filename="audio.wav", sampleRate=16000)()
    model = TensorflowPredictVGGish(graphFilename="audioset-yamnet-1.pb", input="melspectrogram", output="activations")
    activations = model(audio)

Usage for embedding extraction:

.. code-block:: python

    from essentia.standard import MonoLoader, TensorflowPredictVGGish

    audio = MonoLoader(filename="audio.wav", sampleRate=16000)()
    model = TensorflowPredictVGGish(graphFilename="audioset-yamnet-1.pb", input="melspectrogram", output="embeddings")
    embeddings = model(audio)


Music style classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

`Download model files <https://essentia.upf.edu/models/music-style-classification/>`_


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

Demo: https://replicate.com/mtg/effnet-discogs

Dataset: inhouse (MTG).

Output: activations.

This model is useful for music audio embeddings.

Models:

* ``discogs-effnet``

Naming convention: ``<task>-<architecture>-bs<batch_size>-<version>.pb``

Usage for music style classification:

.. code-block:: python

    from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs
    audio = MonoLoader(filename="audio.wav", sampleRate=16000)()
    model = TensorflowPredictEffnetDiscogs(graphFilename="discogs-effnet-bs64-1.pb")
    activations = model(audio)

Usage for embedding extraction:

.. code-block:: python

    from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs
    audio = MonoLoader(filename="audio.wav", sampleRate=16000)()
    model = TensorflowPredictEffnetDiscogs(graphFilename="discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
    activations = model(audio)


Music auto-tagging
^^^^^^^^^^^^^^^^^^

`Download model files <https://essentia.upf.edu/models/autotagging/>`_


Million Song Dataset
--------------------

Music auto-tagging with 50 common music tags:

`rock`, `pop`, `alternative`, `indie`, `electronic`, `female vocalists`, `dance`, `00s`, `alternative rock`, `jazz`, `beautiful`, `metal`, `chillout`, `male vocalists`, `classic rock`, `soul`, `indie rock`, `Mellow`, `electronica`, `80s`, `folk`, `90s`, `chill`, `instrumental`, `punk`, `oldies`, `blues`, `hard rock`, `ambient`, `acoustic`, `experimental`, `female vocalist`, `guitar`, `Hip-Hop`, `70s`, `party`, `country`, `easy listening`, `sexy`, `catchy`, `funk`, `electro`, `heavy metal`, `Progressive rock`, `60s`, `rnb`, `indie pop`, `sad`, `House`, `happy`

Dataset: Million Song Dataset.

Output: activations.

This model is useful for music audio embeddings.

Models:

* ``msd-musicnn``
* ``msd-vgg``

Naming convention: ``<task>-<architecture>-<version>.pb``

Usage for audio event detection:

.. code-block:: python

    from essentia.standard import MonoLoader, TensorflowPredictMusiCNN

    audio = MonoLoader(filename="audio.wav", sampleRate=16000)()
    model = TensorflowPredictMusiCNN(graphFilename="msd-musicnn-1.pb")
    activations = model(audio)

Usage for embedding extraction:

.. code-block:: python

    from essentia.standard import MonoLoader, TensorflowPredictMusiCNN

    audio = MonoLoader(filename="audio.wav", sampleRate=16000)()
    model = TensorflowPredictMusiCNN(graphFilename="msd-musicnn-1.pb", output="model/dense/BiasAdd")
    activations = model(audio)


MagnaTagATune
-------------

Music auto-tagging with 50 common music tags:

`guitar`, `classical`, `slow`, `techno`, `strings`, `drums`, `electronic`, `rock`, `fast`, `piano`, `ambient`, `beat`, `violin`, `vocal`, `synth`, `female`, `indian`, `opera`, `male`, `singing`, `vocals`, `no vocals`, `harpsichord`, `loud`, `quiet`, `flute`, `woman`, `male vocal`, `no vocal`, `pop`, `soft`, `sitar`, `solo`, `man`, `classic`, `choir`, `voice`, `new age`, `dance`, `male voice`, `female vocal`, `beats`, `harp`, `cello`, `no voice`, `weird`, `country`, `metal`, `female voice`, `choral`

Dataset: MagnaTagATune.

Output: activations.

This model is useful for music audio embeddings.

Models:

* ``mtt-musicnn``
* ``mtt-vgg``

Naming convention: ``<task>-<architecture>-<version>.pb``

Usage for audio event detection:

.. code-block:: python

    from essentia.standard import MonoLoader, TensorflowPredictMusiCNN

    audio = MonoLoader(filename="audio.wav", sampleRate=16000)()
    model = TensorflowPredictMusiCNN(graphFilename="mtt-musicnn-1.pb")
    activations = model(audio)

Usage for embedding extraction:

.. code-block:: python

    from essentia.standard import MonoLoader, TensorflowPredictMusiCNN

    audio = MonoLoader(filename="audio.wav", sampleRate=16000)()
    model = TensorflowPredictMusiCNN(graphFilename="mtt-musicnn-1.pb", output="model/dense/BiasAdd")
    activations = model(audio)


Transfer learning classifiers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Classifiers trained on various datasets and audio embeddings.

`Download model files <https://essentia.upf.edu/models/classifiers/>`_

Demo: https://replicate.com/mtg/music-classifiers/

Naming convention: ``<target_task>-<architecture>-<source_task>-<version>.pb``

Usage for music classification with the `MusiCNN` or `VGG` architectures:

.. code-block:: python

    from essentia.standard import MonoLoader, TensorflowPredictMusiCNN

    audio = MonoLoader(filename="audio.wav", sampleRate=16000)()
    model = TensorflowPredictMusiCNN(graphFilename="genre_rosamerica-musicnn-msd-2.pb")
    activations = model(audio)

Usage for music classification with the `VGGish` architecture:

.. code-block:: python

    from essentia.standard import MonoLoader, TensorflowPredictVGGish

    audio = MonoLoader(filename="audio.wav", sampleRate=16000)()
    model = TensorflowPredictVGGish(graphFilename="genre_rosamerica-vggish-audioset-1.pb")
    activations = model(audio)

Danceability
------------

Music danceability (2 classes):

`danceable`, `not_danceable`

Dataset: inhouse (MTG).

Output: activations.

Models:

* ``danceability-musicnn-msd``
* ``danceability-musicnn-mtt``
* ``danceability-vgg-msd``
* ``danceability-vgg-mtt``
* ``danceability-vggish-audioset``


Music loop instrument role
--------------------------

Classification of music loops by their instrument role (5 classes):

`bass`, `chords`, `fx`, `melody`, `percussion`

Dataset: Freesound Loop Dataset.

Output: activations.

Models:

* ``fs_loop_ds-musicnn-msd``


Voice / Instrumental
--------------------

Classification of music by presence or absence of voice (2 classes):

`instrumental`, `voice`

Dataset: inhouse (MTG).

Output: activations.

Models:

* ``voice_instrumental-musicnn-msd``
* ``voice_instrumental-musicnn-mtt``
* ``voice_instrumental-vgg-msd``
* ``voice_instrumental-vgg-mtt``
* ``voice_instrumental-vggish-audioset``


Gender
------

Classification of music by singing voice gender (2 classes):

`female`, `male`

Dataset: inhouse (MTG).

Output: activations.

Models:

* ``gender-musicnn-msd``
* ``gender-musicnn-mtt``
* ``gender-vgg-msd``
* ``gender-vgg-mtt``
* ``gender-vggish-audioset``


Genre Dortmund
--------------

Music genre classification (9 genres):

`alternative`, `blues`, `electronic`, `folkcountry`, `funksoulrnb`, `jazz`, `pop`, `raphiphop`, `rock`

Dataset: Music Audio Benchmark Data Set.

Output: activations.

Models:

* ``genre_dortmund-musicnn-msd``
* ``genre_dortmund-musicnn-mtt``
* ``genre_dortmund-vgg-msd``
* ``genre_dortmund-vgg-mtt``
* ``genre_dortmund-vggish-audioset``


Genre Electronic
----------------

Electronic music genre classification (5 genres)

`ambient`, `dnb`, `house`, `techno`, `trance`

Dataset: inhouse (MTG).

Output: activations.

Models:

* ``genre_electronic-musicnn-msd``
* ``genre_electronic-musicnn-mtt``
* ``genre_electronic-vgg-msd``
* ``genre_electronic-vgg-mtt``
* ``genre_electronic-vggish-audioset``


Genre Rosamerica
----------------

Music genre classification (8 genres):

`classical`, `dance`, `hip hop`, `jazz`, `pop`, `rhythm and blues`, `rock`, `speech`

Dataset: inhouse (MTG).

Output: activations.

Models:

* ``genre_rosamerica-musicnn-msd``
* ``genre_rosamerica-musicnn-mtt``
* ``genre_rosamerica-vgg-msd``
* ``genre_rosamerica-vgg-mtt``
* ``genre_rosamerica-vggish-audioset``


Genre Tzanetakis
----------------

Music genre classification (10 genres):

`blues`, `classic`, `country`, `disco`, `hip hop`, `jazz`, `metal`, `pop`, `reggae`, `rock`

Dataset: inhouse (MTG).

Output: activations.

Models:

* ``genre_tzanetakis-musicnn-msd``
* ``genre_tzanetakis-musicnn-mtt``
* ``genre_tzanetakis-vgg-msd``
* ``genre_tzanetakis-vgg-mtt``
* ``genre_tzanetakis-vggish-audioset``


Mood Acoustic
-------------

Music classification by type of sound (2 classes):

`acoustic`, `non_acoustic`

Dataset: inhouse (MTG).

Output: activations.

Models:

* ``mood_acoustic-musicnn-msd``
* ``mood_acoustic-musicnn-mtt``
* ``mood_acoustic-vgg-msd``
* ``mood_acoustic-vgg-mtt``
* ``mood_acoustic-vggish-audioset``


Mood Aggressive
---------------

Music classification by mood (2 classes):

`aggressive`, `non_aggressive`

Dataset: inhouse (MTG).

Output: activations.

Models:

* ``mood_aggressive-musicnn-msd``
* ``mood_aggressive-musicnn-mtt``
* ``mood_aggressive-vgg-msd``
* ``mood_aggressive-vgg-mtt``
* ``mood_aggressive-vggish-audioset``


Mood Electronic
---------------

Music classification by type of sound (2 classes):

`electronic`, `non_electronic`

Dataset: inhouse (MTG).

Output: activations.

Models:

* ``mood_electronic-musicnn-msd``
* ``mood_electronic-musicnn-mtt``
* ``mood_electronic-vgg-msd``
* ``mood_electronic-vgg-mtt``
* ``mood_electronic-vggish-audioset``


Mood Happy
----------

Music classification by mood (2 classes):

`happy`, `non_happy`

Dataset: inhouse (MTG).

Output: activations.

Models:

* ``mood_happy-musicnn-msd``
* ``mood_happy-musicnn-mtt``
* ``mood_happy-vgg-msd``
* ``mood_happy-vgg-mtt``
* ``mood_happy-vggish-audioset``


Mood Party
----------

Music classification by mood (2 classes):

`party`, `non_party`

Dataset: inhouse (MTG).

Output: activations.

Models:

* ``mood_party-musicnn-msd``
* ``mood_party-musicnn-mtt``
* ``mood_party-vgg-msd``
* ``mood_party-vgg-mtt``
* ``mood_party-vggish-audioset``


Mood Relaxed
------------

Music classification by mood (2 classes):

`relaxed`, `non_relaxed`

Dataset: inhouse (MTG).

Output: activations.

Models:

* ``mood_relaxed-musicnn-msd``
* ``mood_relaxed-musicnn-mtt``
* ``mood_relaxed-vgg-msd``
* ``mood_relaxed-vgg-mtt``
* ``mood_relaxed-vggish-audioset``


Mood Sad
--------

Music classification by mood (2 classes):

`sad`, `non_sad`

Dataset: inhouse (MTG).

Output: activations.

Models:

* ``mood_sad-musicnn-msd``
* ``mood_sad-musicnn-mtt``
* ``mood_sad-vgg-msd``
* ``mood_sad-vgg-mtt``
* ``mood_sad-vggish-audioset``


Moods MIREX
-----------

Music classification by mood (5 mood clusters):

`1: passionate, rousing, confident, boisterous, rowdy`,
`2: rollicking, cheerful, fun, sweet, amiable/good natured`,
`3: literate, poignant, wistful, bittersweet, autumnal, brooding`,
`4: humorous, silly, campy, quirky, whimsical, witty, wry`,
`5: aggressive, fiery, tense/anxious, intense, volatile, visceral`

Dataset: MIREX Audio Mood Classification Dataset.

Output: activations.

Models:

* ``moods_mirex-musicnn-msd``
* ``moods_mirex-musicnn-mtt``
* ``moods_mirex-vgg-msd``
* ``moods_mirex-vgg-mtt``
* ``moods_mirex-vggish-audioset``


Tonal / Atonal
--------------

Music classification by tonality (classes):

`tonal`, `atonal`

Dataset: inhouse (MTG).

Output: activations.

Models:

* ``tonal_atonal-musicnn-msd``
* ``tonal_atonal-musicnn-mtt``
* ``tonal_atonal-vgg-msd``
* ``tonal_atonal-vgg-mtt``
* ``tonal_atonal-vggish-audioset``


Urban sound classification
--------------------------

Urban environment sound classification (10 classes):

`air conditioner`, `car horn`, `children playing`, `dog bark`, `drilling`, `engine idling`, `gun shot`, `jackhammer`, `siren`, `street music`

Dataset: UrbanSound8K.

Output: activations.

Models:

* ``urbansound8k-musicnn-msd``


Feature extractors
^^^^^^^^^^^^^^^^^^

`Download model files <https://essentia.upf.edu/models/feature-extractors/>`_


OpenL3
------

Audio embeddings model trained in a self-supervised manner using audio-visual correspondence information.

Dataset: AudioSet subsets of videos with environmental sounds and musical content.

Output: embeddings.

Models:

* ``openl3-env-mel128-emb512``
* ``openl3-env-mel128-emb6144``
* ``openl3-env-mel256-emb512``
* ``openl3-env-mel256-emb6144``
* ``openl3-music-mel128-emb512``
* ``openl3-music-mel128-emb6144``
* ``openl3-music-mel256-emb512``
* ``openl3-music-mel256-emb6144``

Naming convention: ``<architecture>-<source_task>-<number_of_mel_bands>-<embedding_dimensions>-<version>.pb``

Usage for embedding extraction:

We are currently working on a dedicated algorithm to extract embeddings with the OpenL3 models. For now this can be achieved with `this script <https://gist.github.com/palonso/cfebe37e5492b5a3a31775d8eae8d9a8>`_.

AudioSet-VGGish
---------------

Audio embeddings model accompanying the AudioSet dataset, trained in a supervised manner using tag information for YouTube videos.

Dataset: Subset of Youtube-8M.

Output: embeddings.

Models:

* ``audioset-vggish``

Naming convention: ``<task>-<architecture>-<version>.pb``

Usage for embedding extraction:

.. code-block:: python

    from essentia.standard import MonoLoader, TensorflowPredictVGGish

    audio = MonoLoader(filename="audio.wav", sampleRate=16000)()
    model = TensorflowPredictVGGish(graphFilename="audioset-vggish-3.pb", output='model/vggish/embeddings')
    embeddings = model(audio)


Pitch detection
^^^^^^^^^^^^^^^

`Download model files <https://essentia.upf.edu/models/pitch/>`_

Monophonic pitch tracker (CREPE)
--------------------------------

Monophonic pitch detection (360 20-cent pitch bins, C1-B7).

Dataset: RWC-synth, MDB-stem-synth.

Output: activations.

Models:

* ``crepe-full``
* ``crepe-large``
* ``crepe-medium``
* ``crepe-small``
* ``crepe-tiny``

Naming convention: ``<architecture>-<model_size>-<version>.pb``

Usage for pitch estimation:

.. code-block:: python

    from essentia.standard import MonoLoader, PitchCREPE

    audio = MonoLoader(filename="audio.wav", sampleRate=16000)()
    model = PitchCREPE(graphFilename="crepe-full-1.pb")
    time, frequency, confidence, activations = model(audio)


Source separation
^^^^^^^^^^^^^^^^^

`Download model files <https://essentia.upf.edu/models/source-separation/>`_

Spleeter
--------

Source separation into 2 (`vocals`, `accompaniment`),  4, and 5 (`vocals`, `drums`, `bass`, `piano`, `other`) stems.

Dataset: inhouse (Deezer).

Output: waveforms.

Models:

* ``spleeter-2s``
* ``spleeter-4s``
* ``spleeter-5s``

Naming convention: ``<architecture>-<number_of_stems>-<version>.pb``

Performing source separation:

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


Tempo estimation
^^^^^^^^^^^^^^^^

`Download model files <https://essentia.upf.edu/models/tempo/>`_

TempoCNN
--------

Tempo classification (256 BPM classes, 30-286 BPM).

Dataset: Extended Ballroom, LMDTempo, MTGTempo.

Output: activations.

Models:

* ``deepsquare-k16``
* ``deeptemp-k4``
* ``deeptemp-k16``

Naming convention: ``<architecture>-<model_size>-<version>.pb``

Usage for tempo estimation:

.. code-block:: python

    from essentia.standard import MonoLoader, TempoCNN

    audio = MonoLoader(filename="audio.wav", sampleRate=11025)()
    model = TempoCNN(graphFilename="deepsquare-k16-3.pb")
    global_tempo, local_tempo, local_tempo_probabilities = model(audio)

