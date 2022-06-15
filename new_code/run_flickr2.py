#############################################################
########## This version will save figrue, boxes, and captions
#############################################################

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

####### for colored text and box

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
    ]      ####### I have removed adjectives in the prompts

    print(f"{len(imagenet_classes)} classes, {len(imagenet_templates)} templates")

    ########################################################
    ##################     Load artemis    ##################
    ########################################################

    import os
    from os import listdir
    from os.path import isfile, join
    import random
    import cv2
    import argparse
    import  pickle

    ##################################################
    ##############   flickr30k   #####################
    ##################################################
    from flickr30k_entities_utils import get_sentence_data
    from flickr30k_entities_utils import get_annotations

    import colorsys
    import numpy as np
    import matplotlib.cm as cm

    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=0, type=int)
    args = parser.parse_args()


    from os import listdir
    from os.path import isfile, join
    import pickle

    with open(os.path.join('/home/yiren/flickr30k_entities/', 'flickr30k_split.pkl'),'rb') as file:
        paints_ids_dict = dict(pickle.load(file))

    image_ids = []
    for paint_name, paint_id in paints_ids_dict.items():
        image_root = '/home/yiren/flickr30k_entities/flickr30k-images/' + str(paint_name) + '.jpg'
        image_ids.append([image_root, paint_id])

    image_filename, image_id = image_ids[args.run][0], image_ids[args.run][1]
    image_name = image_filename.split('/')[-1].split('.')[0]

    #### get the 5 captions for the image
    caption_filename = "/home/yiren/flickr30k_entities/Sentences/%s.txt"%(image_name)
    captions = get_sentence_data(caption_filename)

    #### get the annotations (bboxes, etc) for the image
    caption_filename = "/home/yiren/flickr30k_entities/Sentences/%s.txt"%(image_name)
    sentence_data = get_sentence_data(caption_filename)

    captions = []
    for item in sentence_data:
        captions.append(item['sentence'])

    with open(os.path.join('/home/yiren/artemis-speaker-tools-c/features_path','wikiart_split.pkl'),'rb') as file:
        paints_ids_list = pickle.load(file)
        paints_ids_dict = dict(paints_ids_list)

    ####
    print(image_name, image_filename, image_id)   ## 3559429170 /home/yiren/flickr30k_entities/flickr30k-images/3559429170.jpg 0

    #### load images with plt, save in images1 fro all 36 detections
    from matplotlib.patches import Rectangle
    from PIL import Image

    plt.figure()   ### create a new figrue
    plt.imshow(Image.open(image_filename))

    #### run Faster-RCNN
    from faster_rcnn import run_rcnn
    output_rcnn = run_rcnn(image_filename, image_id)

    if output_rcnn['num_boxes'] == 0:
        print("******************* %s *******************"%("Faster-RCNN fails in this image"))
        exit()
    bboxes = output_rcnn['boxes']    #### 36 detected bboxes

    for bbox in bboxes:
        start_point = (round(bbox[0]), round(bbox[1]))
        end_point = (round(bbox[2]), round(bbox[3]))
        plt.gca().add_patch(Rectangle(start_point, round(bbox[2])-round(bbox[0]), round(bbox[3])-round(bbox[1]),
                edgecolor='red',
                linestyle = '-',
                facecolor='none',
                lw=1))

    plt.savefig('images1/%d.png'%image_id)

    ########################################################
    ##################     Start CLIP     ##################
    ########################################################

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    ###### load the image again
    I = Image.open(image_filename)   ### image file

    all_matched_nps_boxes_dict = dict()    ###### a dictionary with noun phrase as key and values of a list of associtated bboxes: {'a nice car': [[x,y,z], [x,y,z], ...]}
    final_bboxes = []      ##### not used at all? Control+F
    all_noun_phrases_in_this_image = []    ##### all noun phrases of all captions (by spacy)
    detected_noun_phrases_in_this_images = []     ### all noun phrases "associated" by our algorithm
    for sentence_id in range(len(captions)):
        sentence_str = captions[sentence_id]    #####  study the first caption only

        import spacy
        nlp = spacy.load('en_core_web_sm')
        text = get_only_chars(sentence_str)
        doc = nlp(text)
        noun_phrases = set(chunk.text.strip().lower() for chunk in doc.noun_chunks)
        nps = [n for n in noun_phrases if n not in stopwords]
        print('the original caption is: ', text)    ##### this is the raw caption of the image
        print('noun phrases extracted by spacy are: ', nps)     ##### this is the noun phrases extracted by spacy
        all_noun_phrases_in_this_image += nps    ##### for final output

        #######

        #######   get only nouns from noun phrases. In the next step, these words are used to find "negatives" from wordnet
        import spacy
        nlp = spacy.load('en_core_web_sm')
        ns = []
        for np in nps:
            text = get_only_chars(np)
            doc = nlp(text)
            for token in doc:
                if token.pos_ == "NOUN":
                    if token.text not in ns:
                        ns.append(token.text)

        print('nouns in noun phrases are: ', ns)     ##### ns is the "noun" from nps

        from nltk.corpus import wordnet as wn
        nps_synsets = []
        for n in ns:
            n_synsets = wn.synsets(n, pos=wn.NOUN)
            if len(n_synsets) > 0:
                nps_synsets.append(n_synsets[0])

        new_nps = []
        new_nps += nps    #### init new_nps with nps. Next: add negatives
        for i in range(1000):
            if len(new_nps) >= 20:   #### len(nps + negatives) to a total of 20
                break
            sampled_obj = random.sample(imagenet_classes, k=1)[0]   #### sample an imagenet name
            sampled_syn = wn.synsets(sampled_obj, pos=wn.NOUN)    #### get synset
            if len(sampled_syn) > 0:     ### but only sampled_syn[0] will be used
                sim = []
                for n_synset in nps_synsets:
                    sim.append(n_synset.path_similarity(sampled_syn[0]))     #### compute sim scores between this "sampled imagenet name" and all "nps"
                if max(sim) < 0.2:    ##### dog and cat has similarity of 0.2
                    new_nps.append(sampled_obj)    ##### if all sim scores are low, this "sampled imagenet name" can be a good negative
            else:
                continue    #### no synset for that sampled imagenet class name

        print('*noun phrases* and *negative nouns* sent to CLIP are: ', new_nps)    ###### original noun phrases and additional negative object classes (total number is 20)

        ################ partition sentence str into list of str (for further coloring) ################
        start_idxs = []
        end_idxs = []
        text = get_only_chars(sentence_str)
        print('the lower case of original sentence str: ', text)
        for np in nps:
            start_idx = [m.start() for m in re.finditer(np, text)]
            end_idx = [idx+len(np) for idx in start_idx]
            print(start_idx, end_idx)
            start_idxs += start_idx
            end_idxs += end_idx

        cut_idx = start_idxs + end_idxs
        cut_idx.sort()
        i = 0
        sentence_list = []
        for idx in cut_idx:
            sentence_list.append(text[i:idx])
            i = idx
        sentence_list.append(text[i:])
        print("partitioned sentences: ", sentence_list)

        sentence_item2color_idx = {}
        for idx in range(len(sentence_list)):
            sentence_item2color_idx[sentence_list[idx]] = idx
        colors = scalars_to_colors([-1] + [i for i in range(len(sentence_list))], colormap='jet')
        ################################################################################################


        logits_matrix = []
        ####### for each noun phrase in the caption
        for np in new_nps:
            with torch.no_grad():
                texts = [template.format(np) for template in imagenet_templates]
                texts = clip.tokenize(texts).cuda() #tokenize
                np_embeddings = model.encode_text(texts) #embed with text encoder
                np_embeddings /= np_embeddings.norm(dim=-1, keepdim=True)
                np_embedding = np_embeddings.mean(dim=0)
                np_embedding /= np_embedding.norm()

                zeroshot_weights = []
                for bbox in bboxes:    #### phrase_id is the id for an object
                    image = I.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                    image = preprocess(image).unsqueeze(0).to(device)

                    image_features = model.encode_image(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    zeroshot_weights.append(image_features.squeeze(0))    ##### append [512], instead of [1,512]
                zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
                logits = 100. * np_embedding @ zeroshot_weights     ##### logits has shape [36]
                logits_matrix.append(logits)

        logits_matrix_tensor = torch.stack(logits_matrix, dim=1)   #### dimension is [36, 20]

        probs_matrix_col1 = torch.softmax(logits_matrix_tensor, dim=0)
        probs_matrix_row1 = torch.softmax(logits_matrix_tensor, dim=1)

        max_v, max_idx = torch.max(probs_matrix_row1[:, :len(nps)], dim=1)      ####### only consider first len(nps) items (which correspond to "true" nps)
        threshold, _ = torch.topk(max_v, k=10)
        threshold = threshold[-1]

        cost_matrix = []
        seleced_boxes_by_1st_iter = []      ########### filter some box that has no link to noun phrases
        for no_box in range(len(bboxes)):
            score = probs_matrix_row1[no_box,:]
            # print(score.shape)  #### shape is 20
            ######    ******** filter out boxes which has low score to all nps   ********
            for j in range(len(nps)):      ####### only consider first len(nps) items (which correspond to "true" nps)
                 if score[j] >= threshold:   #### filter some box that has no link to noun phrases
                     cost_matrix.append(probs_matrix_row1[no_box, :])
                     seleced_boxes_by_1st_iter.append(bboxes[no_box])
                     break

        if len(cost_matrix) == 0:
            print("empty cost matrix, exit(). \n\n")
            break

        cost_matrix = torch.stack(cost_matrix, dim=0)
        print('dimension of the cost matrix: ', cost_matrix.shape)
        print("After filtering some bboxes,  we have %d boxes left!"%(cost_matrix.shape[0]))

        ######## running linear assignment on the sub-matrix
        cost_matrix = - cost_matrix.cpu().numpy()
        from scipy.optimize import linear_sum_assignment
        # cost_matrix = cost_matrix[:,0:len(nps)]   ##### this is for flickr that you know the ground truth
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        row_ind = list(row_ind)
        col_ind = list(col_ind)
        print('linear assignment --> row ind', row_ind)
        print('linear assignment --> col ind', col_ind)


        matched_bboxes = []    ##### store all matched boxes
        matched_nps = []    #### store all matched noun phrases
        color_for_draw_box = []     ####### store integer to colors
        for i in range(len(row_ind)):
            j = row_ind[i]
            k = col_ind[i]

            assigment_score = (-cost_matrix[j, k])
            if assigment_score >= 0:     ##### the assigned pair should have a large enough score
                if k <= len(nps)-1:      #### filter out appened negative nouns, i.e., a box is assigned to a negative noun.u
                    matched_bboxes.append(seleced_boxes_by_1st_iter[j])
                    matched_nps.append(nps[k])

                    ########### dealing with color text here only
                    sentence_item = nps[k]
                    color_idx = 1 + sentence_item2color_idx[sentence_item]
                    color_for_draw_box.append(colors[color_idx])

        print("The matched nps for this sentence: ", matched_nps)
        detected_noun_phrases_in_this_images += matched_nps

        for ii in range(len(matched_bboxes)):
            if matched_nps[ii] not in all_matched_nps_boxes_dict:
                all_matched_nps_boxes_dict[matched_nps[ii]] = [matched_bboxes[ii]]
            else:
                all_matched_nps_boxes_dict[matched_nps[ii]].append(matched_bboxes[ii])

        ########## make a figure for colored caption (only linked for this sentence)
        plt.figure()   ### create a new figure for each caption
        sentence_colors = []
        for sentence_item in sentence_list:
            if sentence_item in matched_nps:
                color_idx = 1 + sentence_item2color_idx[sentence_item]     ###### we added -1 (background) before for colors
                sentence_colors.append(colors[color_idx])     #### do not change color
            else:
                sentence_colors.append(colors[0])
        colored_text_to_figure(sentence_list, colors=sentence_colors)
        plt.savefig('images3/caption-%d_%d.png'%(image_id, sentence_id))


        ########## make a figure for images with bbox (only linked for this sentence)
        plt.figure()   ### create a new figrue
        plt.imshow(Image.open(image_filename))

        for n_bbox in range(len(matched_bboxes)):
            bbox = matched_bboxes[n_bbox]
            start_point = (round(bbox[0]), round(bbox[1]))
            end_point = (round(bbox[2]), round(bbox[3]))
            plt.gca().add_patch(Rectangle(start_point, round(bbox[2])-round(bbox[0]), round(bbox[3])-round(bbox[1]),
                    edgecolor=color_for_draw_box[n_bbox],
                    linestyle = '-',
                    facecolor='none',
                    lw=1))

        plt.savefig('images3/%d_%d.png'%(image_id, sentence_id))
        # print(matched_bboxes)
        # print(matched_nps)
        # print(all_matched_nps_boxes_dict)
        print('\n\n')


        ######## merge two figures into one.
        stack_images_vertically(['images3/%d_%d.png'%(image_id, sentence_id)] + ['images3/caption-%d_%d.png'%(image_id, sentence_id)], save_file='images4/%d_%d.png'%(image_id, sentence_id))



    ##### finally save the figure with selected boxes
    plt.figure()   ### create a new figrue
    plt.imshow(Image.open(image_filename))

    final_num_selected_boxes = 0
    for noun in all_matched_nps_boxes_dict:
        for bbox in all_matched_nps_boxes_dict[noun]:
            final_num_selected_boxes += 1
            start_point = (round(bbox[0]), round(bbox[1]))
            end_point = (round(bbox[2]), round(bbox[3]))
            plt.gca().add_patch(Rectangle(start_point, round(bbox[2])-round(bbox[0]), round(bbox[3])-round(bbox[1]),
                    edgecolor='red',
                    linestyle = '-',
                    facecolor='none',
                    lw=1))

    plt.savefig('images2/%d.png'%image_id)

    print("All nps in this images by spacy (including all captions): \n", all_noun_phrases_in_this_image)
    print("Total associated nps in this image (including all captions): \n", detected_noun_phrases_in_this_images)

    print('total noun phrases of all captions: ', len(all_noun_phrases_in_this_image))
    print('noun phrases with boxes of all captions: ', len(detected_noun_phrases_in_this_images))
    print('selected number of boxes: ', final_num_selected_boxes)
    print('\n\n')

    with open("stats.txt", "a") as f:
        to_write_down = "%d, %d, %d, %d"%(image_id, len(all_noun_phrases_in_this_image), len(detected_noun_phrases_in_this_images), final_num_selected_boxes)
        f.write(to_write_down)
        f.write("\n")
