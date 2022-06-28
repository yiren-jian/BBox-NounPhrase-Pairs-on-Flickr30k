import torch
import clip
from PIL import Image
import numpy as np

import json
import csv
import base64
import numpy as np
import os
import sys

####
import cv2

### from cocoapi
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

import random
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer
import spacy
import copy

import requests
stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
stopwords = set(stopwords_list.decode().splitlines())


#stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our',
			'ours', 'ourselves', 'you', 'your', 'yours',
			'yourself', 'yourselves', 'he', 'him', 'his',
			'himself', 'she', 'her', 'hers', 'herself',
			'it', 'its', 'itself', 'they', 'them', 'their',
			'theirs', 'themselves', 'what', 'which', 'who',
			'whom', 'this', 'that', 'these', 'those', 'am',
			'is', 'are', 'was', 'were', 'be', 'been', 'being',
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at',
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after',
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again',
			'further', 'then', 'once', 'here', 'there', 'when',
			'where', 'why', 'how', 'all', 'any', 'both', 'each',
			'few', 'more', 'most', 'other', 'some', 'such', 'no',
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
			'very', 's', 't', 'can', 'will', 'just', 'don',
			'should', 'now', '']


#cleaning up text
import re
def get_only_chars(line):

    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line


def get_nps(text):
    nlp = spacy.load('en_core_web_sm')   ### load model
    text = get_only_chars(text)   ### pre-process text
    doc = nlp(text)
    noun_phrases = set(chunk.text.strip().lower() for chunk in doc.noun_chunks)
    NPs = [n for n in noun_phrases if n not in stopwords]
    return NPs


##################################################
##############   flickr30k   #####################
##################################################
from flickr30k_entities_utils import get_sentence_data
from flickr30k_entities_utils import get_annotations

import colorsys
import numpy as np
import matplotlib.cm as cm

def scalars_to_colors(float_vals, colormap='jet'):
    cmap = cm.get_cmap(colormap)
    mappable = cm.ScalarMappable(cmap=cmap)
    colors = mappable.to_rgba(float_vals)
    return colors

import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors as mcolors
from PIL import Image
from matplotlib import transforms

def colored_text(in_text, scores=None, colors=None, colormap='jet', for_html=False, space_char=' '):
    if colors is None:
        colors = scalars_to_colors(scores, colormap)

    codes = []
    for c in colors:
        codes.append(mcolors.to_hex(c))   # color as hex.

    res = ''
    if for_html:
        for token, code in zip(in_text, codes):
            if len(res) > 0:
                res += space_char # add space
            res += '<span style="color:{};"> {} </span>'.format(code, token)
    else:
        res = codes
    return res

def colored_text_to_figure(in_text, scores=None, colors=None, figsize=(10, 0.5), colormap='jet', **kw):
    """
    Input: in_text: (list) of strings
            scores: same size list/array of floats, if None: colors arguement must be not None.
            colors: if not None, it will be used instead of scores.
    """
    fig = plt.figure(frameon=False, figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    t = plt.gca().transData

    if colors is None:
        colors = scalars_to_colors(scores, colormap)

    for token, col in zip(in_text, colors):
        text = plt.text(0, 0, ' ' + token + ' ', color=col, transform=t, **kw)
        text.draw(fig.canvas.get_renderer())
        ex = text.get_window_extent()
        t = transforms.offset_copy(text._transform, x=ex.width, units='dots')
    return fig

def stack_images_horizontally(file_names, save_file=None):
    ''' Opens the images corresponding to file_names and
    creates a new image stacking them horizontally.
    '''
    images = list(map(Image.open, file_names))
    widths, heights = list(zip(*(i.size for i in images)))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGBA', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    if save_file is not None:
        new_im.save(save_file)
    return new_im

def stack_images_vertically(file_names, save_file=None):
    ''' Opens the images corresponding to file_names and
    creates a new image stacking them horizontally.
    '''
    images = list(map(Image.open, file_names))
    widths, heights = list(zip(*(i.size for i in images)))
    total_width = max(widths)
    max_height = sum(heights)
    new_im = Image.new('RGBA', (total_width, max_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
    if save_file is not None:
        new_im.save(save_file)
    return new_im


if __name__ == '__main__':

    ### from https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
    imagenet_classes = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper", "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra", "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake", "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider", "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl", "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog", "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf", "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly", "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse", "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish", "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)", "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra", "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton", "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw", "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker", "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard", "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed", "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed", "freight car", "French horn", "frying pan", "fur coat", "garbage truck", "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine", "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag", "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor", "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa", "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail", "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag", "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho", "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug", "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard", "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store", "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater", "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge", "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling", "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink", "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle", "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing", "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website", "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate", "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef", "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player", "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom", "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]

    imagenet_templates = [
        'a bad photo of a {}.',
        'a photo of many {}.',
        'a sculpture of a {}.',
        'a photo of the hard to see {}.',
        'a low resolution photo of the {}.',
        'a rendering of a {}.',
        'graffiti of a {}.',
        'a bad photo of the {}.',
        'a cropped photo of the {}.',
        'a tattoo of a {}.',
        'the embroidered {}.',
        'a photo of a hard to see {}.',
        'a bright photo of a {}.',
        'a dark photo of the {}.',
        'a drawing of a {}.',
        'a photo of my {}.',
        'the plastic {}.',
        'a close-up photo of a {}.',
        'a black and white photo of the {}.',
        'a painting of the {}.',
        'a painting of a {}.',
        'a pixelated photo of the {}.',
        'a sculpture of the {}.',
        'a bright photo of the {}.',
        'a cropped photo of a {}.',
        'a plastic {}.',
        'a jpeg corrupted photo of a {}.',
        'a blurry photo of the {}.',
        'a photo of the {}.',
        'a good photo of the {}.',
        'a rendering of the {}.',
        'a {} in a video game.',
        'a photo of one {}.',
        'a doodle of a {}.',
        'a close-up photo of the {}.',
        'a photo of a {}.',
        'the origami {}.',
        'the {} in a video game.',
        'a sketch of a {}.',
        'a doodle of the {}.',
        'a origami {}.',
        'a low resolution photo of a {}.',
        'the toy {}.',
        'a rendition of the {}.',
        'a rendition of a {}.',
        'a blurry photo of a {}.',
        'a cartoon {}.',
        'art of a {}.',
        'a sketch of the {}.',
        'a embroidered {}.',
        'a pixelated photo of a {}.',
        'itap of the {}.',
        'a jpeg corrupted photo of the {}.',
        'a good photo of a {}.',
        'a plushie {}.',
        'the cartoon {}.',
        'art of the {}.',
        'a drawing of the {}.',
        'a photo of the large {}.',
        'a black and white photo of a {}.',
        'the plushie {}.',
        'a dark photo of a {}.',
        'itap of a {}.',
        'graffiti of the {}.',
        'a toy {}.',
        'itap of my {}.',
        'a tattoo of the {}.',
    ]

    print(f"{len(imagenet_classes)} classes, {len(imagenet_templates)} templates")

    ########################################################
    ##################     Load Flickr    ##################
    ########################################################

    import os
    from os import listdir
    from os.path import isfile, join
    import random
    import cv2
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_idx', default=0, type=int)
    args = parser.parse_args()

    flickr30k_path = '/home/yiren/flickr30k_entities/flickr30k-images'
    filenames = [f for f in listdir(flickr30k_path) if isfile(join(flickr30k_path, f)) if f.endswith('jpg')]
    ids = [filename.split('.')[0] for filename in filenames]

    #### randomly sample an image id
    # id = random.sample(ids, k=1)[0]
    # id = "463786229"   ##### a bee and flowers
    # id = 3466353172
    # id = 3031093209
    # id = 2115849046
    id = ids[args.image_idx]
    print("image id: ", id)

    #### get the 5 captions for the image
    caption_filename = "/home/yiren/flickr30k_entities/Sentences/%s.txt"%(id)
    captions = get_sentence_data(caption_filename)

    #### get the annotations (bboxes, etc) for the image
    anns = get_annotations('/home/yiren/flickr30k_entities/Annotations/%s.xml'%(id))

    # #### load images with opencv
    # im = cv2.imread('flickr30k-images/%s.jpg'%(id))

    phrase_ids = []
    notvisual_phrase_ids = []
    phrase_id2phrases = dict()
    for caption in captions:
        for phrase in caption['phrases']:
            if phrase['phrase_id'] not in phrase_id2phrases:
                phrase_id2phrases[phrase['phrase_id']] = [phrase['phrase']]
            else:
                phrase_id2phrases[phrase['phrase_id']].append(phrase['phrase'])
            phrase_ids.append(phrase['phrase_id'])
            if 'notvisual' in phrase['phrase_type']:
                notvisual_phrase_ids.append(phrase['phrase_id'])
    phrase_ids = list(set(phrase_ids))     #### ['37494', '37487', '37492', '37493', '37486', '37490', '37491']
    notvisual_phrase_ids =list(set(notvisual_phrase_ids))
    # print("notvisual_phrase_ids", notvisual_phrase_ids)
    print("phrase_id2phrases", phrase_id2phrases)

    score2phrase_id = dict()     #### This score is used for generating colored text and boxes
    phrase_id2score = dict()
    for i in range(len(phrase_ids)):
        score2phrase_id[i] = phrase_ids[i]
        phrase_id2score[phrase_ids[i]] = i
    # print(score2phrase_id)   ### {0: '123237', 1: '123233', 2: '123231', 3: '123236', 4: '123234', 5: '123232', 6: '123235'}
    # print(phrase_id2score)   ### {'123237': 0, '123233': 1, '123231': 2, '123236': 3, '123234': 4, '123232': 5, '123235': 6}

    ########## OPTION: CHANGE NONVISUAL TO BACKGROUND COLOR
    for phrase_id in phrase_id2score:
        if phrase_id in notvisual_phrase_ids:
            phrase_id2score[phrase_id] = -1     ##### use -1 for background
    print("phrase_id2score", phrase_id2score)   ### for the nonvisual to same as background

    colors = scalars_to_colors([-1] + [i for i in score2phrase_id], colormap='jet')    ### adding -1 as additional color for background text
    #### colors are list of 3D arrays, each array is a color.

    phrases_for_all_captions = []
    phrase_id_for_all_captions = []
    no_caption = 0
    all_sentence_list = []
    all_sentence_colors = []
    for caption in captions:   #### 5 captions per image, for example
        no_caption += 1
        phrases = caption['phrases']    ### several phrases (noun phrases) per caption
        sentence = caption['sentence']
        sentence_list = []
        start_idxs = []
        end_idxs = []

        phrase_text2phrase_id = dict()
        phrases_for_one_caption = []
        phrase_id_for_one_caption = []
        for phrase in phrases:

            phrase_text2phrase_id[phrase['phrase']] = phrase['phrase_id']
            if 'notvisual' in phrase['phrase_type']:
                pass   ### skip not visible description
            else:
                start_idx = [m.start() for m in re.finditer(phrase['phrase'], sentence)]
                end_idx = [idx+len(phrase['phrase']) for idx in start_idx]
                start_idxs += start_idx
                end_idxs += end_idx

                phrases_for_one_caption.append(phrase['phrase'])    #### skip nonvisual
                phrase_id_for_one_caption.append(phrase['phrase_id'])

        phrases_for_all_captions.append(phrases_for_one_caption)   ### ['Two females', 'blue colored clothes', 'their backs', 'us']
        phrase_id_for_all_captions.append(phrase_id_for_one_caption)   ### ['182374', '182376', '182377', '182382']

        # print(phrase_text2phrase_id)  ### {'A yellow snow plow': '159134', 'the street of snow': '159136'}
        cut_idx = start_idxs + end_idxs
        cut_idx.sort()
        i = 0
        for idx in cut_idx:
            sentence_list.append(sentence[i:idx])
            i = idx
        sentence_list.append(sentence[i:])     ##### parsing sentence based on  (noun) phrases,
                                            ###### e.g., ['', 'Two security guards', ' watch ', 'museum goers', ' examine ', 'an exhibit', ' .']
        all_sentence_list.append(sentence_list)
        sentence_colors = []
        for l in range(len(sentence_list)):
            if sentence_list[l] in phrase_text2phrase_id:
                phrase_id = phrase_text2phrase_id[sentence_list[l]]   ### 'A yellow snow plow' --> '159134'
                sentence_colors.append(colors[phrase_id2score[phrase_id]+1])       ##### '159134' --> 2 --> 2+1=3 --> [0.         0.         0.5        1.        ]
            else:
                sentence_colors.append(colors[0])   ### color[0] is for score -1, which is background
        plt.figure()   ### create a new figure for each caption
        colored_text_to_figure(sentence_list, colors=sentence_colors)
        plt.savefig('caption%s.png'%(str(no_caption)))
        all_sentence_colors.append([sentence_colors, colors[0]])    ###### only store colors

    print(phrases_for_all_captions)
    print(phrase_id_for_all_captions)

    #### draw bbox

    #### load images with plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    plt.figure()   ### create a new figrue
    plt.imshow(Image.open('/home/yiren/flickr30k_entities/flickr30k-images/%s.jpg'%(id)))

    num_bboxes = 0
    # print(colors)  #### list of RGB colors in float
    for phrase_id in anns['boxes']:    #### phrase_id is the id for an object
        bboxes = anns['boxes'][phrase_id]   ### get bboxes ( maybe > 1) for an object class
        for bbox in bboxes:
            bbox_color = colors[phrase_id2score[phrase_id]+1]    ### because we have appended -1 above at the beginning

            start_point = (round(bbox[0]), round(bbox[1]))
            end_point = (round(bbox[2]), round(bbox[3]))
            plt.gca().add_patch(Rectangle(start_point, round(bbox[2])-round(bbox[0]), round(bbox[3])-round(bbox[1]),
                    edgecolor=bbox_color,
                    linestyle = '-',
                    facecolor='none',
                    lw=4))
            num_bboxes += 1

    if num_bboxes > 10:
        print("*****************  Too many boxes in this figure, try another one  *****************")
        exit()

    plt.savefig('image.png')

    #### finalize everyting
    stack_images_vertically(['image.png'] + ['caption%s.png'%(str(i+1)) for i in range(len(captions))], save_file='GT_%s.png'%(id))


    ########################################################
    ##################     Start CLIP     ##################
    ########################################################

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    ######## DO NOT USE COCO OBJECT CLASSES
    # dataDir='/home/yiren/cocoapi'
    # dataType='val2017'
    # annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

    # initialize COCO api for instance annotations
    # coco=COCO(annFile)

    # display COCO categories and supercategories
    # cats = coco.loadCats(coco.getCatIds())
    # nms=[cat['name'] for cat in cats]      ###### nms is further used for nagative classes in CLIP
    nms = []


    ###### load the image again
    I = Image.open('/home/yiren/flickr30k_entities/flickr30k-images/%s.jpg'%(id))   ### image file

    sentence_list = all_sentence_list[0]    #####  study the first caption only
    sentence_colors = all_sentence_colors[0]    #####  study the first caption only
    nps = phrases_for_all_captions[0]    #####  study the first caption only

    print(sentence_list)
    correct_count = 0
    wrong_count = 0
    correct_matched_nps = []
    wrong_matched_nps = dict()   ### {'phrase_text': wrong_id}
    correct_predicted_bboxes_coordinate = []

    #######
    probs_matrix = []
    boxes_coordinates = []
    boxes_nps = []
    for phrase_id in anns['boxes']:    #### phrase_id is the id for an object
        bboxes = anns['boxes'][phrase_id]   ### get bboxes ( maybe > 1) for an object class
        no_box_per_id = 0
        for bbox in bboxes:
            no_box_per_id += 1
            # bbox_color = colors[phrase_id2score[phrase_id]+1]    ### because we have appended -1 above at the beginning


            image = I.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

            image = preprocess(image).unsqueeze(0).to(device)

            print("querying np that matches bbox with phrase_id: %s and %s-th one"%(str(phrase_id), str(no_box_per_id)) )
            print("this bbox should match to one of ", phrase_id2phrases[phrase_id])

            # classnames = copy.copy(nms)
            matched_np_with_probs = dict()     ### multiple nps can match to a bbox
            classnames = nps    ###### the noun phrases
            with torch.no_grad():
                zeroshot_weights = []
                for classname in tqdm(classnames):
                    texts = [template.format(classname) for template in imagenet_templates] #format with class
                    texts = clip.tokenize(texts).cuda() #tokenize
                    class_embeddings = model.encode_text(texts) #embed with text encoder
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    class_embedding = class_embeddings.mean(dim=0)
                    class_embedding /= class_embedding.norm()
                    zeroshot_weights.append(class_embedding)
                zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()

                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = 100. * image_features @ zeroshot_weights

                _, idx = logits.max(dim=-1)
                predicted_classname = classnames[int(idx)]

                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            # print("Label probs for the noun phrase: %.4f"%(probs), "noun phrase: %s, predicted: %s"%(np, predicted_classname))
            for i in range(len(probs)):
                print(nps[i], probs[i])
            if probs.max() < 0.9:
                pass
            else:
                boxes_coordinates.append([bbox[0], bbox[1], bbox[2], bbox[3]])
                boxes_nps.append(phrase_id2phrases[phrase_id])
                probs_matrix.append(probs)

    probs_matrix = np.stack(probs_matrix, axis=0)
    print(probs_matrix)
    print(boxes_coordinates)
    print(nps)

    cost_matrix = - np.asarray(probs_matrix)
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    row_ind = list(row_ind)
    col_ind = list(col_ind)

    correct_predicted_bboxes_coordinate = []
    for i in range(len(row_ind)):
        j = row_ind[i]
        k = col_ind[i]
        if nps[k] in boxes_nps[j]:
            correct_predicted_bboxes_coordinate.append(boxes_coordinates[j])
            correct_matched_nps.append(nps[k])
    print(correct_predicted_bboxes_coordinate)

    print("DONE!")
    print("For a total of %d bbox and %d noun phrases from the first caption corresponding to this image,"%(num_bboxes, len(nps)))
    correct_count = len(correct_predicted_bboxes_coordinate)
    wrong_count = len(row_ind) - len(correct_predicted_bboxes_coordinate)
    print("we have %d correct matching and %d wrong matching."%(correct_count, wrong_count))

                                                        ### sentence_colors has 2 elements, a list and a value
    new_sentence_colors = sentence_colors[0].copy()    #### This is a list
    background_color = sentence_colors[1].copy()     #### this is a value (not list)
    for i in range(len(sentence_list)):
        if sentence_list[i] in correct_matched_nps:
            pass   #### keep the original color
        else:
            new_sentence_colors[i] = background_color

    plt.figure()   ### create a new figure for each caption
    colored_text_to_figure(sentence_list, colors=new_sentence_colors)
    plt.savefig('caption_predicted.png')

    caption = captions[0]
    phrases = caption['phrases']    ### several phrases (noun phrases) per caption
    phrase_text2phrase_id = dict()
    all_phrase_ids_in_this_caption = []
    for phrase in phrases:
        phrase_text2phrase_id[phrase['phrase']] = phrase['phrase_id']
        all_phrase_ids_in_this_caption.append(phrase['phrase_id'])

    #####  count total number of "correct/positive" pairs
    tot = 0
    for phrase_id in anns['boxes']:    #### phrase_id is the id for an object
        if phrase_id in all_phrase_ids_in_this_caption:
            bboxes = anns['boxes'][phrase_id]   ### get bboxes ( maybe > 1) for an object class
            for bbox in bboxes:
                tot += 1

    with open("stats.txt", "a") as f:
        to_write_down = "id:%s, num_bboxes:%d, num_nps:%d, all matchings:%d, correct_count:%d, wrong_count:%d \n"%(id, num_bboxes, len(nps), tot, correct_count, wrong_count)
        f.write(to_write_down)

    plt.figure()   ### create a new figure for each caption
    colored_text_to_figure([to_write_down], scores=[1])
    plt.savefig('stat.png')

    ########## Re-draw the figure
    plt.figure()   ### create a new figrue
    plt.imshow(Image.open('/home/yiren/flickr30k_entities/flickr30k-images/%s.jpg'%(id)))


    # print(colors)  #### list of RGB colors in float
    for phrase_id in anns['boxes']:    #### phrase_id is the id for an object
        bboxes = anns['boxes'][phrase_id]   ### get bboxes ( maybe > 1) for an object class
        for bbox in bboxes:
            if ([bbox[0], bbox[1], bbox[2], bbox[3]]) in correct_predicted_bboxes_coordinate:
                bbox_color = colors[phrase_id2score[phrase_id]+1]    ### because we have appended -1 above at the beginning
                start_point = (round(bbox[0]), round(bbox[1]))
                end_point = (round(bbox[2]), round(bbox[3]))
                plt.gca().add_patch(Rectangle(start_point, round(bbox[2])-round(bbox[0]), round(bbox[3])-round(bbox[1]),
                        edgecolor=bbox_color,
                        facecolor='none',
                        lw=4))

    plt.savefig('image2.png')

    ###### put everyting together
    stack_images_vertically(['image2.png'] + ['caption_predicted.png', 'stat.png'], save_file='PD_%s.png'%(id))
