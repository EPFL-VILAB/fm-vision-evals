# ==Dataset Labels==================================================================


COCO_DETECT_LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


COCO_SEMSEG_LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other', 'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged', 'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged', 'food-other-merged', 'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged']


IMAGENET_LABELS = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead", "electric ray", "stingray", "cock", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "robin", "bulbul", "jay", "magpie", "chickadee", "water ouzel", "kite", "bald eagle", "vulture", "great grey owl", "European fire salamander", "common newt", "eft", "spotted salamander", "axolotl", "bullfrog", "tree frog", "tailed frog", "loggerhead", "leatherback turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "common iguana", "American chameleon", "whiptail", "agama", "frilled lizard", "alligator lizard", "Gila monster", "green lizard", "African chameleon", "Komodo dragon", "African crocodile", "American alligator", "triceratops", "thunder snake", "ringneck snake", "hognose snake", "green snake", "king snake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor", "rock python", "Indian cobra", "green mamba", "sea snake", "horned viper", "diamondback", "sidewinder", "trilobite", "harvestman", "scorpion", "black and gold garden spider", "barn spider", "garden spider", "black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie chicken", "peacock", "quail", "partridge", "African grey", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "drake", "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "American egret", "bittern", "crane (bird)", "limpkin", "European gallinule", "American coot", "bustard", "ruddy turnstone", "red-backed sandpiper", "redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese spaniel", "Maltese dog", "Pekinese", "Shih-Tzu", "Blenheim spaniel", "papillon", "toy terrier", "Rhodesian ridgeback", "Afghan hound", "basset", "beagle", "bloodhound", "bluetick", "black-and-tan coonhound", "Walker hound", "English foxhound", "redbone", "borzoi", "Irish wolfhound", "Italian greyhound", "whippet", "Ibizan hound", "Norwegian elkhound", "otterhound", "Saluki", "Scottish deerhound", "Weimaraner", "Staffordshire bullterrier", "American Staffordshire terrier", "Bedlington terrier", "Border terrier", "Kerry blue terrier", "Irish terrier", "Norfolk terrier", "Norwich terrier", "Yorkshire terrier", "wire-haired fox terrier", "Lakeland terrier", "Sealyham terrier", "Airedale", "cairn", "Australian terrier", "Dandie Dinmont", "Boston bull", "miniature schnauzer", "giant schnauzer", "standard schnauzer", "Scotch terrier", "Tibetan terrier", "silky terrier", "soft-coated wheaten terrier", "West Highland white terrier", "Lhasa", "flat-coated retriever", "curly-coated retriever", "golden retriever", "Labrador retriever", "Chesapeake Bay retriever", "German short-haired pointer", "vizsla", "English setter", "Irish setter", "Gordon setter", "Brittany spaniel", "clumber", "English springer", "Welsh springer spaniel", "cocker spaniel", "Sussex spaniel", "Irish water spaniel", "kuvasz", "schipperke", "groenendael", "malinois", "briard", "kelpie", "komondor", "Old English sheepdog", "Shetland sheepdog", "collie", "Border collie", "Bouvier des Flandres", "Rottweiler", "German shepherd", "Doberman", "miniature pinscher", "Greater Swiss Mountain dog", "Bernese mountain dog", "Appenzeller", "EntleBucher", "boxer", "bull mastiff", "Tibetan mastiff", "French bulldog", "Great Dane", "Saint Bernard", "Eskimo dog", "malamute", "Siberian husky", "dalmatian", "affenpinscher", "basenji", "pug", "Leonberg", "Newfoundland", "Great Pyrenees", "Samoyed", "Pomeranian", "chow", "keeshond", "Brabancon griffon", "Pembroke", "Cardigan", "toy poodle", "miniature poodle", "standard poodle", "Mexican hairless", "timber wolf", "white wolf", "red wolf", "coyote", "dingo", "dhole", "African hunting dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", "tabby", "tiger cat", "Persian cat", "Siamese cat", "Egyptian cat", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "ice bear", "sloth bear", "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle", "long-horned beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket", "walking stick", "cockroach", "mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "admiral", "ringlet", "monarch", "cabbage butterfly", "sulphur butterfly", "lycaenid", "starfish", "sea urchin", "sea cucumber", "wood rabbit", "hare", "Angora", "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "sorrel", "zebra", "hog", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram", "bighorn", "ibex", "hartebeest", "impala", "gazelle", "Arabian camel", "llama", "weasel", "mink", "polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas", "baboon", "macaque", "langur", "colobus", "proboscis monkey", "marmoset", "capuchin", "howler monkey", "titi", "spider monkey", "squirrel monkey", "Madagascar cat", "indri", "Indian elephant", "African elephant", "lesser panda", "giant panda", "barracouta", "eel", "coho", "rock beauty", "anemone fish", "sturgeon", "gar", "lionfish", "puffer", "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibian", "analog clock", "apiary", "apron", "ashcan", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint", "Band Aid", "banjo", "bannister", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "barrow", "baseball", "basketball", "bassinet", "bassoon", "bathing cap", "bath towel", "bathtub", "beach wagon", "beacon", "beaker", "bearskin", "beer bottle", "beer glass", "bell cote", "bib", "bicycle-built-for-two", "bikini", "binder", "binoculars", "birdhouse", "boathouse", "bobsled", "bolo tie", "bonnet", "bookcase", "bookshop", "bottlecap", "bow", "bow tie", "brass", "brassiere", "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "bullet train", "butcher shop", "cab", "caldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "carpenter's kit", "carton", "car wheel", "cash machine", "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "cellular telephone", "chain", "chainlink fence", "chain mail", "chain saw", "chest", "chiffonier", "chime", "china cabinet", "Christmas stocking", "church", "cinema", "cleaver", "cliff dwelling", "cloak", "clog", "cocktail shaker", "coffee mug", "coffeepot", "coil", "combination lock", "computer keyboard", "confectionery", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "crane", "crash helmet", "crate", "crib", "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishrag", "dishwasher", "disk brake", "dock", "dogsled", "dome", "doormat", "drilling platform", "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope", "espresso maker", "face powder", "feather boa", "file", "fireboat", "fire engine", "fire screen", "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster", "freight car", "French horn", "frying pan", "fur coat", "garbage truck", "gasmask", "gas pump", "goblet", "go-kart", "golf ball", "golfcart", "gondola", "gong", "gown", "grand piano", "greenhouse", "grille", "grocery store", "guillotine", "hair slide", "hair spray", "half track", "hammer", "hamper", "hand blower", "hand-held computer", "handkerchief", "hard disc", "harmonica", "harp", "harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", "hoopskirt", "horizontal bar", "horse cart", "hourglass", "iPod", "iron", "jack-o'-lantern", "jean", "jeep", "jersey", "jigsaw puzzle", "jinrikisha", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop", "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine", "liner", "lipstick", "Loafer", "lotion", "loudspeaker", "loupe", "lumbermill", "magnetic compass", "mailbag", "mailbox", "maillot", "maillot", "manhole cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine chest", "megalith", "microphone", "microwave", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", "Model T", "modem", "monastery", "monitor", "moped", "mortar", "mortarboard", "mosque", "mosquito net", "motor scooter", "mountain bike", "mountain tent", "mouse", "mousetrap", "moving van", "muzzle", "nail", "neck brace", "necklace", "nipple", "notebook", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "organ", "oscilloscope", "overskirt", "oxcart", "oxygen mask", "packet", "paddle", "paddlewheel", "padlock", "paintbrush", "pajama", "palace", "panpipe", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "passenger car", "patio", "pay-phone", "pedestal", "pencil box", "pencil sharpener", "perfume", "Petri dish", "photocopier", "pick", "pickelhaube", "picket fence", "pickup", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate", "pitcher", "plane", "planetarium", "plastic bag", "plate rack", "plow", "plunger", "Polaroid camera", "pole", "police van", "poncho", "pool table", "pop bottle", "pot", "potter's wheel", "power drill", "prayer rug", "printer", "prison", "projectile", "projector", "puck", "punching bag", "purse", "quill", "quilt", "racer", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle", "reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "rubber eraser", "rugby ball", "rule", "running shoe", "safe", "safety pin", "saltshaker", "sandal", "sarong", "sax", "scabbard", "scale", "school bus", "schooner", "scoreboard", "screen", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe shop", "shoji", "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "ski mask", "sleeping bag", "slide rule", "sliding door", "slot", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar dish", "sombrero", "soup bowl", "space bar", "space heater", "space shuttle", "spatula", "speedboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", "steel arch bridge", "steel drum", "stethoscope", "stole", "stone wall", "stopwatch", "stove", "strainer", "streetcar", "stretcher", "studio couch", "stupa", "submarine", "suit", "sundial", "sunglass", "sunglasses", "sunscreen", "suspension bridge", "swab", "sweatshirt", "swimming trunks", "swing", "switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy", "television", "tennis ball", "thatch", "theater curtain", "thimble", "thresher", "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toyshop", "tractor", "trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright", "vacuum", "vase", "vault", "velvet", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "warplane", "washbasin", "washer", "water bottle", "water jug", "water tower", "whiskey jug", "whistle", "wig", "window screen", "window shade", "Windsor tie", "wine bottle", "wing", "wok", "wooden spoon", "wool", "worm fence", "wreck", "yawl", "yurt", "web site", "comic book", "crossword puzzle", "street sign", "traffic light", "book jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "ice lolly", "French loaf", "bagel", "pretzel", "cheeseburger", "hotdog", "mashed potato", "head cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "custard apple", "pomegranate", "hay", "carbonara", "chocolate sauce", "dough", "meat loaf", "pizza", "potpie", "burrito", "red wine", "espresso", "cup", "eggnog", "alp", "bubble", "cliff", "coral reef", "geyser", "lakeside", "promontory", "sandbar", "seashore", "valley", "volcano", "ballplayer", "groom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "hip", "buckeye", "coral fungus", "agaric", "gyromitra", "stinkhorn", "earthstar", "hen-of-the-woods", "bolete", "ear", "toilet tissue"]


# ==Default Settings==================================================================


O4_DEFAULTS = {
    'classify': {
        'prompt_no': 5,
    },
    'detect': {
        'prompt_no': 6,
        'n_iters': 6
    },
    'detect_naive': {
        'prompt_no': 5,
    },
    'segment': {
        'prompt_no': 2,
        'shape': 'rectangle'
    },
    'segment_sans_context': {
        'prompt_no': 1,
        'shape': 'rectangle',
        'n_segments': 400,
    },
    'segment_naive': {
        'prompt_no': 5,
    },
    'group': {
        'prompt_no': 2,
        'shape': 'curve'
    },
    'depth': {
        'prompt_no': 4,
        'shape': 'curve'
    },
    'normals': {
        'prompt_no': 4,
        'shape': 'rectangle'
    }
}


GEMINI_DEFAULTS = {
    'classify': {
        'prompt_no': 1
    },
    'detect': {
        'prompt_no': 6,
        'n_iters': 8,
        'independent': True
    },
    'detect_naive': {
        'prompt_no': 5,
        'normalization': 1000
    },
    'segment': {
        'prompt_no': 1,
        'shape': 'point'
    },
    'segment_naive': {
        'prompt_no': 1,
    },
    'segment_sans_context': {
        'prompt_no': 1,
        'shape': 'rectangle',
        'n_segments': 400,
    },
    'group': {
        'prompt_no': 2,
        'shape': 'curve'
    },
    'depth': {
        'prompt_no': 5,
        'shape': 'point'
    },
    'normals': {
        'prompt_no': 1,
        'shape': 'curve'
    }
}


CLAUDE_DEFAULTS = {
    'classify': {
        'prompt_no': 5
    },
    'detect': {
        'prompt_no': 3,
        'n_iters': 7,
        'classification_type': 'classify_mult'
    },
    'detect_naive': {
        'prompt_no': 1,
        'classification_type': 'classify_mult',
    },
    'segment': {
        'prompt_no': 2,
        'shape': 'point'
    },
    'segment_naive': {
        'prompt_no': 1,
    },
    'segment_sans_context': {
        'prompt_no': 1,
        'shape': 'rectangle',
        'n_segments': 400,
    },
    'group': {
        'prompt_no': 2,
        'shape': 'curve'
    },
    'depth': {
        'prompt_no': 2,
        'shape': 'rectangle',
        'n_threads': 10
    },
    'normals': {
        'prompt_no': 3,
        'shape': 'rectangle'
    }
}


LLAMA_DEFAULTS = {
    'classify': {
        'prompt_no': 3,
    },
    'detect': {
        'prompt_no': 7,
        'n_iters': 8,
        'classification_type': 'classify_mult',
        'independent': True,
        'no_context': True,
        'mark_rectangle': True
    },
    'segment_sans_context': {
        'prompt_no': 1,
        'shape': 'rectangle',
        'n_segments': 400,
    },
    'group_sans_context': {
        'prompt_no': 1,
        'shape': 'curve'
    },
    'depth': {
        'prompt_no': 3,
        'shape': 'point',
    },
    'normals': {
        'prompt_no': 4,
        'shape': 'rectangle',
        'n_threads': 10,
    }
}

QWEN2_DEFAULTS = {
    'classify': {
        'prompt_no': 3,
    },
    'detect': {
        'prompt_no': 6,
        'n_iters': 6
    },
    'detect_naive': {
        'prompt_no': 5,
    },
    'segment': {
        'prompt_no': 1,
        'shape': 'rectangle'
    },
    'group': {
        'prompt_no': 2,
        'shape': 'curve'
    },
    'depth': {
        'prompt_no': 4,
        'shape': 'curve'
    },
    'normals': {
        'prompt_no': 4,
        'shape': 'rectangle'
    }
}


# ==Miscellaneous==================================================================


COCO_COLOR_MAP = {"person": [220, 20, 60], "bicycle": [119, 11, 32], "car": [0, 0, 142], "motorcycle": [0, 0, 230], "airplane": [106, 0, 228], "bus": [0, 60, 100], "train": [0, 80, 100], "truck": [0, 0, 70], "boat": [0, 0, 192], "traffic light": [250, 170, 30], "fire hydrant": [100, 170, 30], "stop sign": [220, 220, 0], "parking meter": [175, 116, 175], "bench": [250, 0, 30], "bird": [165, 42, 42], "cat": [255, 77, 255], "dog": [0, 226, 252], "horse": [182, 182, 255], "sheep": [0, 82, 0], "cow": [120, 166, 157], "elephant": [110, 76, 0], "bear": [174, 57, 255], "zebra": [199, 100, 0], "giraffe": [72, 0, 118], "backpack": [255, 179, 240], "umbrella": [0, 125, 92], "handbag": [209, 0, 151], "tie": [188, 208, 182], "suitcase": [0, 220, 176], "frisbee": [255, 99, 164], "skis": [92, 0, 73], "snowboard": [133, 129, 255], "sports ball": [78, 180, 255], "kite": [0, 228, 0], "baseball bat": [174, 255, 243], "baseball glove": [45, 89, 255], "skateboard": [134, 134, 103], "surfboard": [145, 148, 174], "tennis racket": [255, 208, 186], "bottle": [197, 226, 255], "wine glass": [171, 134, 1], "cup": [109, 63, 54], "fork": [207, 138, 255], "knife": [151, 0, 95], "spoon": [9, 80, 61], "bowl": [84, 105, 51], "banana": [74, 65, 105], "apple": [166, 196, 102], "sandwich": [208, 195, 210], "orange": [255, 109, 65], "broccoli": [0, 143, 149], "carrot": [179, 0, 194], "hot dog": [209, 99, 106], "pizza": [5, 121, 0], "donut": [227, 255, 205], "cake": [147, 186, 208], "chair": [153, 69, 1], "couch": [3, 95, 161], "potted plant": [163, 255, 0], "bed": [119, 0, 170], "dining table": [0, 182, 199], "toilet": [0, 165, 120], "tv": [183, 130, 88], "laptop": [95, 32, 0], "mouse": [130, 114, 135], "remote": [110, 129, 133], "keyboard": [166, 74, 118], "cell phone": [219, 142, 185], "microwave": [79, 210, 114], "oven": [178, 90, 62], "toaster": [65, 70, 15], "sink": [127, 167, 115], "refrigerator": [59, 105, 106], "book": [142, 108, 45], "clock": [196, 172, 0], "vase": [95, 54, 80], "scissors": [128, 76, 255], "teddy bear": [201, 57, 1], "hair drier": [246, 0, 122], "toothbrush": [191, 162, 208], "banner": [255, 255, 128], "blanket": [147, 211, 203], "bridge": [150, 100, 100], "cardboard": [168, 171, 172], "counter": [146, 112, 198], "curtain": [210, 170, 100], "door-stuff": [92, 136, 89], "floor-wood": [218, 88, 184], "flower": [241, 129, 0], "fruit": [217, 17, 255], "gravel": [124, 74, 181], "house": [70, 70, 70], "light": [255, 228, 255], "mirror-stuff": [154, 208, 0], "net": [193, 0, 92], "pillow": [76, 91, 113], "platform": [255, 180, 195], "playingfield": [106, 154, 176], "railroad": [230, 150, 140], "river": [60, 143, 255], "road": [128, 64, 128], "roof": [92, 82, 55], "sand": [254, 212, 124], "sea": [73, 77, 174], "shelf": [255, 160, 98], "snow": [255, 255, 255], "stairs": [104, 84, 109], "tent": [169, 164, 131], "towel": [225, 199, 255], "wall-brick": [137, 54, 74], "wall-stone": [135, 158, 223], "wall-tile": [7, 246, 231], "wall-wood": [107, 255, 200], "water-other": [58, 41, 149], "window-blind": [183, 121, 142], "window-other": [255, 73, 97], "tree-merged": [107, 142, 35], "fence-merged": [190, 153, 153], "ceiling-merged": [146, 139, 141], "sky-other-merged": [70, 130, 180], "cabinet-merged": [134, 199, 156], "table-merged": [209, 226, 140], "floor-other-merged": [96, 36, 108], "pavement-merged": [96, 96, 96], "mountain-merged": [64, 170, 64], "grass-merged": [152, 251, 152], "dirt-merged": [208, 229, 228], "paper-merged": [206, 186, 171], "food-other-merged": [152, 161, 64], "building-other-merged": [116, 112, 0], "rock-merged": [0, 114, 143], "wall-other-merged": [102, 102, 156], "rug-merged": [250, 141, 255], "unknown": [0, 0, 0]}


COCO_LABEL_2_ID = {'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4, 'bus': 5, 'train': 6, 'truck': 7, 'boat': 8, 'traffic light': 9, 'fire hydrant': 10, 'stop sign': 11, 'parking meter': 12, 'bench': 13, 'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19, 'elephant': 20, 'bear': 21, 'zebra': 22, 'giraffe': 23, 'backpack': 24, 'umbrella': 25, 'handbag': 26, 'tie': 27, 'suitcase': 28, 'frisbee': 29, 'skis': 30, 'snowboard': 31, 'sports ball': 32, 'kite': 33, 'baseball bat': 34, 'baseball glove': 35, 'skateboard': 36, 'surfboard': 37, 'tennis racket': 38, 'bottle': 39, 'wine glass': 40, 'cup': 41, 'fork': 42, 'knife': 43, 'spoon': 44, 'bowl': 45, 'banana': 46, 'apple': 47, 'sandwich': 48, 'orange': 49, 'broccoli': 50, 'carrot': 51, 'hot dog': 52, 'pizza': 53, 'donut': 54, 'cake': 55, 'chair': 56, 'couch': 57, 'potted plant': 58, 'bed': 59, 'dining table': 60, 'toilet': 61, 'tv': 62, 'laptop': 63, 'mouse': 64, 'remote': 65, 'keyboard': 66, 'cell phone': 67, 'microwave': 68, 'oven': 69, 'toaster': 70, 'sink': 71, 'refrigerator': 72, 'book': 73, 'clock': 74, 'vase': 75, 'scissors': 76, 'teddy bear': 77, 'hair drier': 78, 'toothbrush': 79, 'banner': 80, 'blanket': 81, 'bridge': 82, 'cardboard': 83, 'counter': 84, 'curtain': 85, 'door-stuff': 86, 'floor-wood': 87, 'flower': 88, 'fruit': 89, 'gravel': 90, 'house': 91, 'light': 92, 'mirror-stuff': 93, 'net': 94, 'pillow': 95, 'platform': 96, 'playingfield': 97, 'railroad': 98, 'river': 99, 'road': 100, 'roof': 101, 'sand': 102, 'sea': 103, 'shelf': 104, 'snow': 105, 'stairs': 106, 'tent': 107, 'towel': 108, 'wall-brick': 109, 'wall-stone': 110, 'wall-tile': 111, 'wall-wood': 112, 'water-other': 113, 'window-blind': 114, 'window-other': 115, 'tree-merged': 116, 'fence-merged': 117, 'ceiling-merged': 118, 'sky-other-merged': 119, 'cabinet-merged': 120, 'table-merged': 121, 'floor-other-merged': 122, 'pavement-merged': 123, 'mountain-merged': 124, 'grass-merged': 125, 'dirt-merged': 126, 'paper-merged': 127, 'food-other-merged': 128, 'building-other-merged': 129, 'rock-merged': 130, 'wall-other-merged': 131, 'rug-merged': 132}
