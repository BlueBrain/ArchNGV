import seaborn as sns

def spatial_distribution_mosaic(points):
    return sns.jointplot(points[:, 0], points[:, 2], kind='hex')
