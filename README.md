# ANN GGMM
 Generalised ground motion model developed using artificial neural network

# Reference
Aristeidou, S., Shahnazaryan, D. and O’Reilly, G.J. (2024) ‘Artificial neural network-based ground motion model for next-generation seismic intensity measures’, Under Review

### 1. Install requirements

```shell
pip3 install -r requirements.txt
```

### 2. Run a sample code (example.py)

```python
from gmpe import AristeidouEtAl2024


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
)

print("Output", mean, stdev)
```