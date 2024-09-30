
import torch
from torch import nn
from torchvision import models, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_model():

  class_names_x = ['Adhirasam',
      'Aloo gobi',
      'Aloo matar',
      'Aloo methi',
      'Aloo shimla mirch',
      'Aloo tikki',
      'Anarsa',
      'Ariselu',
      'Bandar laddu',
      'Basundi',
      'Bhatura',
      'Bhindi masala',
      'Biryani',
      'Boondi',
      'Butter chicken',
      'Chak hao kheer',
      'Cham cham',
      'Chana masala',
      'Chapati',
      'Chhena kheeri',
      'Chicken razala',
      'Chicken tikka',
      'Chicken tikka masala',
      'Chikki',
      'Daal baati churma',
      'Daal puri',
      'Dal makhani',
      'Dal tadka',
      'Dharwad pedha',
      'Doodhpak',
      'Double ka meetha',
      'Dum aloo',
      'Gajar ka halwa',
      'Gavvalu',
      'Ghevar',
      'Gulab jamun',
      'Imarti',
      'Jalebi',
      'Kachori',
      'Kadai paneer',
      'Kadhi pakoda',
      'Kajjikaya',
      'Kakinada khaja',
      'Kalakand',
      'Karela bharta',
      'Kofta',
      'Kuzhi paniyaram',
      'Lassi',
      'Ledikeni',
      'Litti chokha',
      'Lyangcha',
      'Maach jhol',
      'Makki di roti sarson da saag',
      'Malapua',
      'Misi roti',
      'Misti doi',
      'Modak',
      'Mysore pak',
      'Naan',
      'Navrattan korma',
      'Palak paneer',
      'Paneer butter masala',
      'Phirni',
      'Pithe',
      'Poha',
      'Poornalu',
      'Pootharekulu',
      'Qubani ka meetha',
      'Rabri',
      'Ras malai',
      'Rasgulla',
      'Sandesh',
      'Shankarpali',
      'Sheer korma',
      'Sheera',
      'Shrikhand',
      'Sohan halwa',
      'Sohan papdi',
      'Sutar feni',
      'Unni appam']

  model_ft = models.resnet18()
  num_ftrs = model_ft.fc.in_features
  # Here the size of each output sample is set to 2.
  # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names_x))``.
  model_ft.fc = nn.Linear(num_ftrs, len(class_names_x))

  PATH = 'model_indian_food_classifier_2.pt'
  # model_ft.load_state_dict(torch.load(PATH), map_location=torch.device('cpu'))
  model = torch.load(PATH, map_location=device)
  model_ft.load_state_dict(model)

  data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
   }

  return model_ft, data_transforms, class_names_x
