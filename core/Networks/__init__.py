def build_network(cfg=None):
    if cfg is None:
        raise ValueError("Configuration must be provided to build the network.")
    name = cfg.network
    model_cfg = cfg.SAMFlow_base
    if name == 'SAMFlow_base':
        from .SAMFlow import SAMFlow_base
        return SAMFlow_base(model_cfg)
    if name == 'SAM2Flow':
        from .SAMFlow import SAM2Flow
        return SAM2Flow(model_cfg)
    raise ValueError(f"Unsupported network type: {name}")
