

from torch import optim as optim


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def get_pretrain_param_groups(model, logger, skip_list=(), skip_keywords=()):
        has_decay = []
        no_decay = []
        has_decay_name = []
        no_decay_name = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                    check_keywords_in_name(name, skip_keywords):
                no_decay.append(param)
                no_decay_name.append(name)
            else:
                has_decay.append(param)
                has_decay_name.append(name)
        logger.info(f'No decay params: {no_decay_name}')
        logger.info(f'Has decay params: {has_decay_name}')
        return [{'params': has_decay},
                {'params': no_decay, 'weight_decay': 0.}]

def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def build_pretrain_optimizer(args, model, logger):
    logger.info('>>>>>>>>>> Build Optimizer for Pre-training Stage')
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
        logger.info(f'No weight decay: {skip}')
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
        logger.info(f'No weight decay keywords: {skip_keywords}')

    parameters = get_pretrain_param_groups(model, logger, skip, skip_keywords)
    # print("Parameters as for the SimMIM implementation: ")
    # print(model.parameters())
    # paramaters_dino = get_params_groups(model)
    # print("Parameters as for the DINO implementation: ")
    # print(paramaters_dino)


    opt_lower = args.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=args.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=args.TRAIN.BASE_LR, weight_decay=args.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=args.TRAIN.OPTIMIZER.EPS, betas=args.TRAIN.OPTIMIZER.BETAS,
                                lr=args.TRAIN.BASE_LR, weight_decay=args.TRAIN.WEIGHT_DECAY)

    logger.info(optimizer)
    return optimizer