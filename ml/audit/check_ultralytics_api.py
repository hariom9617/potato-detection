"""Check if cls_weights parameter is supported in Ultralytics 8.0.196."""

try:
    from ultralytics import YOLO
    import inspect
    
    # Create a model instance
    model = YOLO('yolov8n.pt')
    
    # Get train method signature
    train_signature = inspect.signature(model.train)
    
    print("="*60)
    print("YOLOV8 TRAIN METHOD PARAMETERS")
    print("="*60)
    
    params = train_signature.parameters
    
    # Check if cls_weights is in parameters
    if 'cls_weights' in params:
        print("✓ cls_weights parameter IS supported")
        print(f"  Default: {params['cls_weights'].default}")
    else:
        print("✗ cls_weights parameter NOT in train() signature")
        print("\nNote: This parameter may not be supported in Ultralytics 8.0.196")
        print("      The commented-out line in the notebook should include a warning")
    
    print(f"\nTotal parameters available: {len(params)}")
    print("\nSample of available parameters:")
    for i, (name, param) in enumerate(list(params.items())[:20]):
        default = param.default if param.default != inspect.Parameter.empty else "required"
        print(f"  {name}: {default}")
    
except Exception as e:
    print(f"Error checking API: {e}")
    print("\nThis is expected if not running in the virtual environment.")
