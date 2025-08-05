#!/usr/bin/env python3
"""
Script to automatically migrate all remaining datasets from Dassl to custom infrastructure.
Phase 3: Complete Dassl removal.
"""

import os
import re
import glob

def migrate_dataset_file(filepath):
    """Migrate a single dataset file from Dassl to custom infrastructure."""
    print(f"Migrating {filepath}")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Replace imports
    content = re.sub(
        r'from dassl\.data\.datasets import DATASET_REGISTRY, Datum, DatasetBase',
        'from utils.dataset_base import DATASET_REGISTRY, Datum, DatasetBase, mkdir_if_missing, listdir_nohidden',
        content
    )
    
    content = re.sub(
        r'from dassl\.utils import ([^,\n]+)',
        lambda m: f'# {m.group(0)} - functions now imported from utils.dataset_base',
        content
    )
    
    # Replace cfg parameter with config
    content = re.sub(r'def __init__\(self, cfg\):', 'def __init__(self, config):', content)
    
    # Replace cfg usages
    content = re.sub(r'cfg\.DATASET\.ROOT', 'config.dataset.root', content)
    content = re.sub(r'cfg\.DATASET\.NUM_SHOTS', 'config.dataset.num_shots', content) 
    content = re.sub(r'cfg\.SEED', 'config.seed', content)
    content = re.sub(r'cfg\.DATASET\.SUBSAMPLE_CLASSES', "getattr(config.dataset, 'subsample_classes', 'all')", content)
    
    # Write back
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"✅ Migrated {filepath}")

def main():
    """Migrate all dataset files."""
    dataset_files = glob.glob('datasets/*.py')
    
    # Skip files we already migrated or don't need to migrate
    skip_files = ['__init__.py', 'caltech101.py', 'oxford_pets.py', 'dtd.py', 
                  'imagenet_templates.py', 'imagenet_a_r_indexes_v2.py',
                  'oxford_pets_backup.py', 'oxford_pets_corrupted.py']
    
    for filepath in dataset_files:
        filename = os.path.basename(filepath)
        if filename in skip_files:
            print(f"⏭️  Skipping {filename}")
            continue
            
        try:
            migrate_dataset_file(filepath)
        except Exception as e:
            print(f"❌ Error migrating {filepath}: {e}")

if __name__ == "__main__":
    main()
