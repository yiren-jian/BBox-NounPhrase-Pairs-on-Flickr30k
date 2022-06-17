### install CLIP related
```shell
TODO
```

### install Faster-RCNN related
```shell
TODO
```

### install nltk packages
```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### download CLIP model
```python
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
```
