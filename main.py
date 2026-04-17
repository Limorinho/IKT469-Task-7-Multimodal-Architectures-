import torch
from src.combiner.classfier import MultimodalClassifier
from src.data.loader import get_loaders, GENRES
from src.data.get import _dataset_dir

dir = _dataset_dir()
data = get_loaders(dir)

train_loader = data["train"]
val_loader = data["val"]
test_loader = data["test"]

batch = next(iter(train_loader))
print(batch["input_ids"].shape)
print(batch["image"].shape)
print(batch["labels"].shape)


from transformers import BertModel          
from torchvision.models import resnet50, ResNet50_Weights
                                              
text_encoder = BertModel.from_pretrained("bert-base-uncased")                                           
image_encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
                                                                                                          
# ResNet's final layer is a classifier — replace it with an identity
# so it outputs features instead of class scores                                                        
image_encoder.fc = torch.nn.Identity()


model = MultimodalClassifier(
    text_embedding=text_encoder,  # Replace with actual text embedding model
    image_embedding=image_encoder,  # Replace with actual image embedding model
    text_dim=768,  # BERT's hidden size
    image_dim=2048,  # ResNet-50's output feature size
    num_classes=len(GENRES),  # Replace with actual number of classes
)

output = model(batch["attention_mask"], batch["input_ids"], batch["image"])

