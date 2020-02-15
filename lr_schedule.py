def step_lr(ep):
    if ep<50:
        lr = 0.01
    elif ep<100:
        lr = 0.005
    elif ep<150:
        lr = 0.001
    elif ep<200:
        lr = 0.0005
    else:
        lr = 0.0001
    return lr