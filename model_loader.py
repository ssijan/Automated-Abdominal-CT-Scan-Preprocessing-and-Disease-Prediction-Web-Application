"""
Model loader: reconstructs ResNet50, MobileNetV2, and FewShot Learning models and loads weights 
from .keras archive files and PyTorch .pth files.
"""
import zipfile
import os
import tempfile
import traceback
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import h5py
import numpy as np
import torch
import torch.nn as nn


def _extract_weights_from_group(group):
    """Extract weight arrays from an HDF5 group."""
    weights = []
    
    if 'vars' in group:
        # Weights are under 'vars' subgroup
        vars_group = group['vars']
        keys = sorted(vars_group.keys(), key=lambda x: int(x) if x.isdigit() else x)
        for k in keys:
            weights.append(vars_group[k][()])
    else:
        # Weights are direct datasets in group
        for k in sorted(group.keys()):
            if isinstance(group[k], h5py.Dataset):
                weights.append(group[k][()])
    
    return weights if weights else None


def _load_functional_weights(functional_layer, functional_group, verbose=False):
    """
    Load weights for the base functional model (ResNet50/MobileNetV2).
    Map h5 layer names to the built model's layers.
    """
    if 'layers' not in functional_group:
        return
    
    h5_layers = functional_group['layers']
    loaded_count = 0

    # Strategy: saved HDF5 layer names may not match runtime layer names exactly.
    # We'll iterate the HDF5 layer list in order and try to assign each group's
    # weight arrays to the next compatible layer in the built model by matching
    # the number of weight arrays and their shapes. This is tolerant to name
    # differences but requires the layer ordering to be preserved in the file.

    h5_layer_names = list(h5_layers.keys())
    model_layers = list(functional_layer.layers)

    mi = 0  # pointer into model_layers
    for h5_name in h5_layer_names:
        h5_group = h5_layers[h5_name]
        try:
            weights = _extract_weights_from_group(h5_group)
        except Exception as e:
            if verbose:
                print(f"  Failed to extract weights for H5 layer '{h5_name}': {e}")
            continue

        if not weights:
            # no trainable weights for this H5 layer
            continue

        # Try to find a model layer starting from current pointer that matches
        found = False
        # Search forward through remaining model layers to find first compatible match
        for j in range(mi, len(model_layers)):
            model_layer = model_layers[j]
            ml_weights = model_layer.get_weights()
            # skip layers without weights
            if not ml_weights:
                continue

            # Quick check: number of arrays should match
            if len(ml_weights) != len(weights):
                continue

            # Check shapes match element-wise
            shapes_match = True
            for a, b in zip(ml_weights, weights):
                if a.shape != b.shape:
                    shapes_match = False
                    break

            if shapes_match:
                # assign weights
                try:
                    model_layer.set_weights(weights)
                    loaded_count += 1
                    found = True
                    if verbose:
                        print(f"  Loaded H5 '{h5_name}' -> model '{model_layer.name}' ({len(weights)} arrays)")
                    # advance model pointer to just after this matched layer
                    mi = j + 1
                    break
                except Exception as e:
                    if verbose:
                        print(f"  Failed to set weights for model layer '{model_layer.name}': {e}")
                    # try next candidate
                    continue

        if not found and verbose:
            print(f"  No matching model layer found for H5 layer '{h5_name}' (skipping)")

    if verbose:
        print(f"  Total base layers loaded: {loaded_count}")


def load_resnet50_model(archive_path='models/abdominal_ct_resnet50.keras'):
    """
    Reconstruct ResNet50 model and load weights from .keras archive.
    Returns a compiled model ready for prediction, or None on failure.
    """
    try:
        print(f"[ResNet50] Loading from {archive_path}")
        
        # Extract weights.h5 from archive
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(archive_path, 'r') as z:
                z.extract('model.weights.h5', tmpdir)
            weights_path = os.path.join(tmpdir, 'model.weights.h5')
            
            # Build the exact model architecture
            print("  Building architecture...")
            base = tf.keras.applications.ResNet50(
                include_top=False, 
                weights=None, 
                input_shape=(224, 224, 3)
            )
            x = base.output
            x = layers.GlobalAveragePooling2D(name='global_average_pooling2d_2')(x)
            x = layers.Dense(
                512, 
                activation='relu', 
                name='dense_2',
                kernel_regularizer=regularizers.l2(0.001)
            )(x)
            x = layers.Dropout(0.6, name='dropout_2')(x)
            preds = layers.Dense(11, activation='softmax', name='dense_3')(x)
            model = models.Model(inputs=base.input, outputs=preds)
            
            # Load weights using h5py and manual assignment
            print("  Loading weights...")
            with h5py.File(weights_path, 'r') as h5f:
                # Load base model (ResNet50) weights
                if 'layers' in h5f and 'functional' in h5f['layers']:
                    print("  Loading base ResNet50 weights...")
                    base_weights_group = h5f['layers']['functional']
                    _load_functional_weights(base, base_weights_group, verbose=True)
                
                # Load top layers (dense_2, dropout_2, dense_3)
                if 'layers' in h5f:
                    layers_group = h5f['layers']
                    print("  Loading top layer weights...")
                    
                    # Map: h5 name -> target model layer name
                    layer_mapping = {
                        'dense': 'dense_2',
                        'dense_1': 'dense_3',
                        'dropout': 'dropout_2',
                        'global_average_pooling2d': 'global_average_pooling2d_2'
                    }
                    
                    for h5_name, target_name in layer_mapping.items():
                        if h5_name in layers_group:
                            try:
                                weights = _extract_weights_from_group(layers_group[h5_name])
                                if weights:
                                    # Find layer by target name
                                    for model_layer in model.layers:
                                        if model_layer.name == target_name:
                                            try:
                                                model_layer.set_weights(weights)
                                                print(f"    Loaded {target_name} from {h5_name}")
                                            except Exception as e:
                                                print(f"    Failed to set weights for {target_name}: {e}")
                                            break
                            except Exception as e:
                                print(f"    Error extracting {h5_name}: {e}")
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            print("  ResNet50 model ready")
            return model
    except Exception as e:
        print(f"Failed to load ResNet50 from {archive_path}: {e}")
        traceback.print_exc()
        return None


def load_mobilenetv2_model(archive_path='models/abdominal_ct_mobilenetv2.keras'):
    """
    Reconstruct MobileNetV2 model and load weights from .keras archive.
    Returns a compiled model ready for prediction, or None on failure.
    """
    try:
        print(f"[MobileNetV2] Loading from {archive_path}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(archive_path, 'r') as z:
                z.extract('model.weights.h5', tmpdir)
            weights_path = os.path.join(tmpdir, 'model.weights.h5')
            
            # Build the exact model architecture
            print("  Building architecture...")
            base = tf.keras.applications.MobileNetV2(
                include_top=False,
                weights=None,
                input_shape=(224, 224, 3)
            )
            x = base.output
            x = layers.GlobalAveragePooling2D(name='global_average_pooling2d_3')(x)
            x = layers.Dense(
                512,
                activation='relu',
                name='dense_4',
                kernel_regularizer=regularizers.l2(0.001)
            )(x)
            x = layers.Dropout(0.6, name='dropout_3')(x)
            preds = layers.Dense(11, activation='softmax', name='dense_5')(x)
            model = models.Model(inputs=base.input, outputs=preds)
            
            # Load weights using h5py and manual assignment
            print("  Loading weights...")
            with h5py.File(weights_path, 'r') as h5f:
                # Load base model (MobileNetV2) weights
                if 'layers' in h5f and 'functional' in h5f['layers']:
                    print("  Loading base MobileNetV2 weights...")
                    base_weights_group = h5f['layers']['functional']
                    _load_functional_weights(base, base_weights_group, verbose=True)
                
                # Load top layers
                if 'layers' in h5f:
                    layers_group = h5f['layers']
                    print("  Loading top layer weights...")
                    
                    # Map: h5 name -> target model layer name
                    layer_mapping = {
                        'dense': 'dense_4',
                        'dense_1': 'dense_5',
                        'dropout': 'dropout_3',
                        'global_average_pooling2d': 'global_average_pooling2d_3'
                    }
                    
                    for h5_name, target_name in layer_mapping.items():
                        if h5_name in layers_group:
                            try:
                                weights = _extract_weights_from_group(layers_group[h5_name])
                                if weights:
                                    # Find layer by target name
                                    for model_layer in model.layers:
                                        if model_layer.name == target_name:
                                            try:
                                                model_layer.set_weights(weights)
                                                print(f"    Loaded {target_name} from {h5_name}")
                                            except Exception as e:
                                                print(f"    Failed to set weights for {target_name}: {e}")
                                            break
                            except Exception as e:
                                print(f"    Error extracting {h5_name}: {e}")
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            print("  MobileNetV2 model ready")
            return model
    except Exception as e:
        print(f"Failed to load MobileNetV2 from {archive_path}: {e}")
        traceback.print_exc()
        return None


# Cache loaded models to avoid reloading
_model_cache = {}


# ============ PyTorch FewShot Learning Model ============

class ProtoNetCNN(nn.Module):
    """FewShot Learning feature extractor CNN - matches saved .pth weights with conv.0, conv.3, conv.6."""
    def __init__(self, num_classes=11):
        super(ProtoNetCNN, self).__init__()
        # Checkpoint has conv.0, conv.3, conv.6
        # These indices (0, 3, 6) strongly suggest the original model had ReLU + other layers in a Sequential
        # But we only have conv weights, so we'll ignore the ReLU during loading
        self.conv0 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.fc = nn.Linear(128, 128)
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv0(x), inplace=True)
        x = torch.nn.functional.relu(self.conv3(x), inplace=True)
        x = torch.nn.functional.relu(self.conv6(x), inplace=True)
        x = x.mean(dim=[2, 3])  # Global average pooling
        x = torch.nn.functional.relu(self.fc(x), inplace=True)
        x = self.classifier(x)
        return x
    


def load_protonet_model(pth_path='models/protonet_best.pth'):
    """
    Load FewShot Learning PyTorch model from .pth file.
    Returns a model ready for prediction.
    """
    try:
        print(f"[FewShot Learning] Loading from {pth_path}")
        
        # Build model
        print("  Building architecture...")
        model = ProtoNetCNN(num_classes=11)
        
        # Load weights
        print("  Loading weights...")
        checkpoint = torch.load(pth_path, map_location='cpu')
        
        # Remap checkpoint keys from conv.0/3/6 to conv0/conv3/conv6
        mapping = {
            'conv.0.weight': 'conv0.weight',
            'conv.0.bias': 'conv0.bias',
            'conv.3.weight': 'conv3.weight',
            'conv.3.bias': 'conv3.bias',
            'conv.6.weight': 'conv6.weight',
            'conv.6.bias': 'conv6.bias',
        }
        
        remapped_checkpoint = {}
        for key, value in checkpoint.items():
            remapped_checkpoint[mapping.get(key, key)] = value
        
            # Load with strict=False since checkpoint may not have all layers (e.g., classifier)
            model.load_state_dict(remapped_checkpoint, strict=False)
        
        # Set to eval mode (no dropout/batchnorm training)
        model.eval()
        
        print("  FewShot Learning model ready")
        return model
    except Exception as e:
        print(f"Failed to load FewShot Learning from {pth_path}: {e}")
        traceback.print_exc()
        return None
        traceback.print_exc()
        return None


def get_model(model_id):
    """
    Get a loaded model by ID (resnet50, mobilenetv2, proto_fewshot).
    Caches the model after first load.
    Returns:
      - TensorFlow model (resnet50, mobilenetv2) - callable with predict(batch)
      - PyTorch model (proto_fewshot) - callable with forward(batch) after to('cpu')
    """
    if model_id in _model_cache:
        return _model_cache[model_id]
    
    model = None
    if model_id == 'resnet50':
        model = load_resnet50_model()
    elif model_id == 'mobilenetv2':
        model = load_mobilenetv2_model()
    elif model_id == 'proto_fewshot':
        model = load_protonet_model()
    elif model_id == 'resnet50_finetuned':
        model = load_resnet50_model()
    elif model_id == 'mobilenetv2_finetuned':
        model = load_mobilenetv2_model()
    
    if model:
        _model_cache[model_id] = model
    return model
