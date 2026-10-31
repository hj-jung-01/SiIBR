def normalize(x):
    # min-max normalization to uint8 for visualization
    return ((x-x.min())/(x.max()-x.min()+1e-8)*255).astype('uint8')
