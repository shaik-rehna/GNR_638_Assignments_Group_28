import torch
import torch.nn as nn
import timm

# load pretrained backbone
def load_model(model_name, num_classes=30, pretrained=True):

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes)

    return model


# Freeze Backbone(Linear Probe)
def freeze_backbone(model):

    for param in model.parameters():
        param.requires_grad = False

    # keep classifier trainable
    for param in model.get_classifier().parameters():
        param.requires_grad = True

    return model

# freeze everything
def freeze_everything(model):
    for param in model.parameters():
        param.requires_grad = False
    return model
    
# unfreeze classifier
def unfreeze_classifier(model):
    classifier = model.get_classifier()
    if classifier is not None:
        for param in classifier.parameters():
            param.requires_grad = True
    return model
    
# Unfreeze Last Block(Fine-tuning strategy)
def unfreeze_last_block(model, model_name):

    if "resnet" in model_name:
        for param in model.layer4.parameters():
            param.requires_grad = True

    elif "densenet" in model_name:
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True

    elif "efficientnet" in model_name:
        for param in model.blocks[-1].parameters():
            param.requires_grad = True

    return model

# Selective 20% Parameter Unfreeze
# Deeper layers capture high-level semantic features 
# and adapt better to new tasks, so we unfreeze them first
def unfreeze_percentage(model, percentage=0.2):

    total_params = sum(p.numel() for p in model.parameters())
    target_params = total_params * percentage

    running = 0

    for p in reversed(list(model.parameters())):

        if running + p.numel() > target_params:
            break

        p.requires_grad = True
        running += p.numel()

    return model

def unfreeze_percentage_eff_b0(model, percentage=0.2):

    total_params = sum(p.numel() for p in model.parameters())
    target_params = total_params * percentage

    running = 0

    for p in reversed(list(model.parameters())):

        p.requires_grad = True
        running += p.numel()

        if running >= target_params:
            break

    return model

# Full Fine-tuning
def unfreeze_all(model):

    for param in model.parameters():
        param.requires_grad = True

    return model

# Count Trainable Parameters(for efficiency analysis)
def count_trainable_params(model):

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    return trainable, total

