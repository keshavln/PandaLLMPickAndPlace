import os
import supervision as sv
from PIL import Image
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import GroundingDINO.groundingdino.datasets.transforms as T

CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
WEIGHTS_PATH = os.path.join("weights", WEIGHTS_NAME)    

model = load_model(CONFIG_PATH, WEIGHTS_PATH)

def detect_objects(image, text_prompt, box_thr=0.35, text_thr=0.25):
    """
    Detects objects in the image using Grounding DINO.
    """
    pil_image = Image.fromarray(image)

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img, _ = transform(pil_image, target=None)
    boxes, logits, phrases = predict(model=model, image=img, caption=text_prompt, box_threshold=box_thr, text_threshold=text_thr)
    annotated_frame = annotate(image_source=image, boxes=boxes, logits=logits, phrases=phrases)
    return annotated_frame, boxes