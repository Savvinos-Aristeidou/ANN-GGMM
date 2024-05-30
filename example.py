import numpy as np
from gmpe import AristeidouEtAl2024

# Insert the period with at least 1 decimal point
im_name = "FIV3(0.88s)"

gmpe = AristeidouEtAl2024()

mean, stdev = gmpe.get_mean_and_stddevs(
    im_name,
    mag=[6.5, 5.0],
    rjb=[30, 30],
    rrup=[30, 30],
    d_hyp=[10, 10],
    vs30=[300, 300],
    mechanism=[0, 0],
    z2pt5=[1000, 1000],
    rx=[30, 30],
    ztor=[1, 1],
    component_definition="RotD50"
)

print("Output", np.exp(mean), np.squeeze(stdev))
