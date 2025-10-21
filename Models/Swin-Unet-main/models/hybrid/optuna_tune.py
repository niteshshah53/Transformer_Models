#!/usr/bin/env python3
"""
Optuna Hyperparameter Tuning for Hybrid Models
Automatically finds optimal hyperparameters for historical document segmentation
"""

import argparse
import os
import random
import sys
import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import json
from datetime import datetime

# Add common directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

warnings.filterwarnings("ignore")

# Import training modules
from trainer_optuna import trainer_hybrid_optuna  # Modified trainer that returns metric

# Import dataset classes
from datasets.dataset_udiadsbib import UDiadsBibDataset
try:
    from datasets.dataset_divahisdb import DivaHisDBDataset
    DIVAHISDB_AVAILABLE = True
except ImportError:
    DivaHisDBDataset = None
    DIVAHISDB_AVAILABLE = False


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def get_model(model_type, num_classes, img_size):
    """Create and initialize the Hybrid model."""
    if model_type == 'hybrid1':
        print(f"[Trial] Loading Hybrid1: EfficientNet-Swin model...")
        from hybrid1.hybrid_model import HybridEfficientNetB4SwinDecoder
        model = HybridEfficientNetB4SwinDecoder(
            num_classes=num_classes,
            img_size=img_size,
            pretrained=True
        )
    elif model_type == 'hybrid2':
        print(f"[Trial] Loading Hybrid2: Swin-EfficientNet model...")
        from hybrid2.hybrid_model import create_hybrid2_model
        model = create_hybrid2_model(
            num_classes=num_classes,
            img_size=img_size,
            efficientnet_variant='b4'
        )
    else:
        raise ValueError(f"Unknown model: {model_type}")
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    return model


def setup_datasets(args):
    """Create training and validation datasets."""
    if args.dataset == 'UDIADS_BIB':
        print(f"[Trial] Setting up U-DIADS-Bib dataset for {args.manuscript}...")
        
        if args.manuscript in ['Syr341FS', 'Syr341']:
            num_classes = 5
        else:
            num_classes = 6
        
        args.num_classes = num_classes
        
        train_dataset = UDiadsBibDataset(
            root_dir=args.udiadsbib_root,
            manuscript=args.manuscript,
            split='training',
            patch_size=args.img_size,
            use_patched_data=args.use_patched_data,
            model_type='hybrid' + args.model[-1],  # 'hybrid1' or 'hybrid2'
            num_classes=num_classes
        )
        
        val_dataset = UDiadsBibDataset(
            root_dir=args.udiadsbib_root,
            manuscript=args.manuscript,
            split='validation',
            patch_size=args.img_size,
            use_patched_data=args.use_patched_data,
            model_type='hybrid' + args.model[-1],  # 'hybrid1' or 'hybrid2'
            num_classes=num_classes
        )
        
    elif args.dataset == 'DIVAHISDB':
        if not DIVAHISDB_AVAILABLE:
            raise ImportError("DivaHisDB dataset not available")
        
        print(f"[Trial] Setting up DivaHisDB dataset for {args.manuscript}...")
        args.num_classes = 4
        
        train_dataset = DivaHisDBDataset(
            root_dir=args.divahisdb_root,
            manuscript=args.manuscript,
            split='train',
            patch_size=args.img_size,
            use_patched_data=args.use_patched_data,
            model_type='hybrid' + args.model[-1]  # 'hybrid1' or 'hybrid2'
        )
        
        val_dataset = DivaHisDBDataset(
            root_dir=args.divahisdb_root,
            manuscript=args.manuscript,
            split='val',
            patch_size=args.img_size,
            use_patched_data=args.use_patched_data,
            model_type='hybrid' + args.model[-1]  # 'hybrid1' or 'hybrid2'
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    return train_dataset, val_dataset


def objective(trial, base_args):
    """
    Optuna objective function - called for each trial.
    
    Args:
        trial: Optuna trial object
        base_args: Base arguments from command line
        
    Returns:
        float: Best validation loss (to minimize)
    """
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-3, 0.1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 24, 32])
    optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'Adam', 'SGD'])
    scheduler_type = trial.suggest_categorical('scheduler', ['CosineAnnealing', 'ReduceLROnPlateau', 'CosineWarmRestarts'])
    
    # Additional hyperparameters
    if optimizer_name == 'SGD':
        momentum = trial.suggest_float('momentum', 0.8, 0.99)
    else:
        momentum = 0.9  # Default for AdamW/Adam
    
    if scheduler_type == 'CosineWarmRestarts':
        T_0 = trial.suggest_categorical('T_0', [10, 20, 30, 50])
    else:
        T_0 = 30
    
    # Print trial parameters
    print(f"\n{'='*80}")
    print(f"OPTUNA TRIAL {trial.number}")
    print(f"{'='*80}")
    print(f"Learning Rate: {learning_rate:.6f}")
    print(f"Weight Decay: {weight_decay:.6f}")
    print(f"Batch Size: {batch_size}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Scheduler: {scheduler_type}")
    if optimizer_name == 'SGD':
        print(f"Momentum: {momentum:.3f}")
    if scheduler_type == 'CosineWarmRestarts':
        print(f"T_0: {T_0}")
    print(f"{'='*80}\n")
    
    # Create a copy of args with trial hyperparameters
    import copy
    args = copy.deepcopy(base_args)
    args.base_lr = learning_rate
    args.weight_decay = weight_decay
    args.batch_size = batch_size
    args.optimizer_name = optimizer_name
    args.momentum = momentum
    args.scheduler_type = scheduler_type
    args.T_0 = T_0
    
    # Set reduced epochs for faster tuning
    args.max_epochs = args.optuna_max_epochs
    args.patience = args.optuna_patience
    
    # Create unique output directory for this trial
    trial_output_dir = os.path.join(
        args.output_dir,
        f"trial_{trial.number:03d}"
    )
    os.makedirs(trial_output_dir, exist_ok=True)
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup datasets
    try:
        train_dataset, val_dataset = setup_datasets(args)
    except Exception as e:
        print(f"[Trial {trial.number}] Dataset setup failed: {e}")
        return float('inf')
    
    # Create model
    try:
        model = get_model(args.model, args.num_classes, args.img_size)
    except Exception as e:
        print(f"[Trial {trial.number}] Model creation failed: {e}")
        return float('inf')
    
    # Train model
    try:
        best_val_loss = trainer_hybrid_optuna(
            args, model, trial_output_dir, 
            train_dataset, val_dataset,
            trial=trial  # Pass trial for pruning
        )
    except optuna.TrialPruned:
        print(f"[Trial {trial.number}] Trial pruned by Optuna")
        raise
    except Exception as e:
        print(f"[Trial {trial.number}] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return float('inf')
    
    print(f"\n[Trial {trial.number}] Completed with validation loss: {best_val_loss:.4f}\n")
    
    return best_val_loss


def parse_arguments():
    """Parse command line arguments for Optuna tuning."""
    parser = argparse.ArgumentParser(description='Optuna Hyperparameter Tuning for Hybrid Models')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='hybrid1', 
                       choices=['hybrid1', 'hybrid2'],
                       help='Model type to tune')
    parser.add_argument('--img_size', type=int, default=224, 
                       help='Input image size')
    
    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='UDIADS_BIB',
                       choices=['UDIADS_BIB', 'DIVAHISDB'],
                       help='Dataset to use')
    parser.add_argument('--manuscript', type=str, default='Latin2',
                       help='Manuscript to train on')
    parser.add_argument('--udiadsbib_root', type=str, 
                       default='../../U-DIADS-Bib-MS_patched',
                       help='Root directory for UDIADS_BIB dataset')
    parser.add_argument('--divahisdb_root', type=str, 
                       default='../../DivaHisDB_patched',
                       help='Root directory for DIVAHISDB dataset')
    parser.add_argument('--use_patched_data', action='store_true', default=False,
                       help='Use pre-generated patches')
    
    # Optuna configuration
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of Optuna trials to run')
    parser.add_argument('--optuna_max_epochs', type=int, default=50,
                       help='Max epochs per trial (reduced for faster tuning)')
    parser.add_argument('--optuna_patience', type=int, default=15,
                       help='Early stopping patience per trial')
    parser.add_argument('--study_name', type=str, default=None,
                       help='Optuna study name (for resuming)')
    parser.add_argument('--storage', type=str, default=None,
                       help='Optuna storage URL (e.g., sqlite:///optuna.db)')
    
    # Training configuration
    parser.add_argument('--n_gpu', type=int, default=1, 
                       help='Number of GPUs')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=1234,
                       help='Random seed')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for Optuna results')
    
    return parser.parse_args()


def main():
    """Main function for Optuna hyperparameter tuning."""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up Optuna study
    study_name = args.study_name or f"hybrid_{args.model}_{args.dataset}_{args.manuscript}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if args.storage:
        storage = args.storage
    else:
        # Default: SQLite database in output directory
        storage = f"sqlite:///{os.path.join(args.output_dir, 'optuna_study.db')}"
    
    print(f"\n{'='*80}")
    print(f"OPTUNA HYPERPARAMETER TUNING")
    print(f"{'='*80}")
    print(f"Study Name: {study_name}")
    print(f"Storage: {storage}")
    print(f"Number of Trials: {args.n_trials}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Manuscript: {args.manuscript}")
    print(f"Max Epochs per Trial: {args.optuna_max_epochs}")
    print(f"Early Stopping Patience: {args.optuna_patience}")
    print(f"Output Directory: {args.output_dir}")
    print(f"{'='*80}\n")
    
    # Create Optuna study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='minimize',  # Minimize validation loss
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        load_if_exists=True  # Resume if study exists
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        catch=(Exception,)
    )
    
    # Print results
    print(f"\n{'='*80}")
    print(f"OPTUNA OPTIMIZATION COMPLETED")
    print(f"{'='*80}")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"\nBest trial:")
    trial = study.best_trial
    print(f"  Trial number: {trial.number}")
    print(f"  Best validation loss: {trial.value:.4f}")
    print(f"\n  Best hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print(f"{'='*80}\n")
    
    # Save results
    results_file = os.path.join(args.output_dir, 'optuna_best_params.json')
    with open(results_file, 'w') as f:
        json.dump({
            'best_trial_number': trial.number,
            'best_validation_loss': trial.value,
            'best_params': trial.params,
            'study_name': study_name,
            'n_trials': len(study.trials),
            'model': args.model,
            'dataset': args.dataset,
            'manuscript': args.manuscript
        }, f, indent=2)
    
    print(f"Best parameters saved to: {results_file}")
    
    # Generate visualization (optional - requires plotly)
    try:
        import optuna.visualization as vis
        
        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html(os.path.join(args.output_dir, 'optimization_history.html'))
        
        # Parameter importances
        fig = vis.plot_param_importances(study)
        fig.write_html(os.path.join(args.output_dir, 'param_importances.html'))
        
        # Parallel coordinate plot
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(os.path.join(args.output_dir, 'parallel_coordinate.html'))
        
        print(f"Visualizations saved to: {args.output_dir}")
    except ImportError:
        print("Install plotly for visualizations: pip install plotly")
    except Exception as e:
        print(f"Visualization generation failed: {e}")


if __name__ == '__main__':
    main()

