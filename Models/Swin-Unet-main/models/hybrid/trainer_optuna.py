#!/usr/bin/env python3
"""
Modified trainer for Optuna hyperparameter tuning
Returns best validation loss instead of message
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import optuna

# Import from original trainer
from trainer import (
    setup_logging,
    create_data_loaders,
    compute_class_weights,
    create_loss_functions,
    run_training_epoch,
    validate_with_sliding_window
)


def create_optimizer_and_scheduler_optuna(model, args):
    """
    Create optimizer and scheduler based on Optuna trial parameters.
    
    Args:
        model: Neural network model
        args: Arguments with Optuna-suggested hyperparameters
        
    Returns:
        tuple: (optimizer, scheduler)
    """
    # Create optimizer based on trial suggestion
    if args.optimizer_name == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.base_lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
    elif args.optimizer_name == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.base_lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
    elif args.optimizer_name == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.base_lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            nesterov=True
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer_name}")
    
    # Create scheduler based on trial suggestion
    if args.scheduler_type == 'CosineAnnealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.max_epochs,
            eta_min=1e-7
        )
    elif args.scheduler_type == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )
    elif args.scheduler_type == 'CosineWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.T_0,
            T_mult=2,
            eta_min=1e-7
        )
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler_type}")
    
    print(f"Optimizer: {args.optimizer_name} (lr={args.base_lr:.6f}, weight_decay={args.weight_decay:.6f})")
    print(f"Scheduler: {args.scheduler_type}")
    
    return optimizer, scheduler


def trainer_hybrid_optuna(args, model, snapshot_path, train_dataset=None, val_dataset=None, trial=None):
    """
    Modified training function for Optuna - returns best validation loss.
    
    Args:
        args: Command line arguments with training configuration
        model: Neural network model to train
        snapshot_path: Directory to save models and logs
        train_dataset: Training dataset
        val_dataset: Validation dataset
        trial: Optuna trial object (for pruning)
        
    Returns:
        float: Best validation loss achieved
    """
    # Set up logging
    logger = setup_logging(snapshot_path)
    
    # Set up multi-GPU training if available
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dataset, val_dataset,
        args.batch_size * args.n_gpu,
        args.num_workers,
        args.seed
    )
    
    # Compute class weights
    if hasattr(train_dataset, 'mask_paths'):
        class_weights = compute_class_weights(train_dataset, args.num_classes)
    else:
        class_weights = torch.ones(args.num_classes)
    
    # Create loss functions, optimizer, and scheduler
    ce_loss, focal_loss, dice_loss = create_loss_functions(class_weights, args.num_classes)
    optimizer, scheduler = create_optimizer_and_scheduler_optuna(model, args)
    
    # Set up TensorBoard logging
    writer = SummaryWriter(os.path.join(snapshot_path, 'tensorboard_logs'))
    
    # Training loop
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    patience = args.patience
    
    print(f"\nStarting training: {args.max_epochs} epochs, patience={patience}")
    print("-" * 80)
    
    for epoch in range(args.max_epochs):
        # Training phase
        train_loss = run_training_epoch(model, train_loader, ce_loss, focal_loss, dice_loss, optimizer, class_weights)
        
        # Validation phase
        val_loss = validate_with_sliding_window(model, val_dataset, ce_loss, focal_loss, dice_loss)
        
        # Log results
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.max_epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")
        
        # Update learning rate
        if args.scheduler_type == 'ReduceLROnPlateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            
            # Save best model
            save_mode_path = os.path.join(snapshot_path, 'best_model_latest.pth')
            if args.n_gpu > 1:
                torch.save(model.module.state_dict(), save_mode_path)
            else:
                torch.save(model.state_dict(), save_mode_path)
            
            print(f"  ✓ New best model! Val Loss: {best_val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement ({epochs_without_improvement}/{patience})")
        
        # Optuna pruning - prune unpromising trials
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                print(f"Trial pruned at epoch {epoch+1}")
                raise optuna.TrialPruned()
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping after {epochs_without_improvement} epochs without improvement")
            logger.info(f"Early stopping after {epochs_without_improvement} epochs without improvement")
            break
    
    writer.close()
    
    print(f"\nTraining completed. Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    
    return best_val_loss

