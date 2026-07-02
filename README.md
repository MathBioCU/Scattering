# Scattering (UNDER CONSTRUCTION)

<img width="3702" height="1452" alt="kdv_noisy_tracking_snapshots" src="https://github.com/user-attachments/assets/1b48f9ac-e8ef-4942-9e9c-d63707de8f90" />

Python code accompanying the manuscript **"Learning effective soliton dynamics from scattering data,"** submitted to the *Journal of Nonlinear Waves.*
```
BibTex citation coming soon...
```

Also see:
- Zenodo: `coming soon...`
- ArXiV pre-print: `coming soon...`
- To recreate results in the paper, see `recreate_paper_results.ipynb` and `recreate_figures.ipynb`.
- [The tutorials and examples located here](https://github.com/SethMinor/PyWSINDy-for-PDEs) for instructions on how to use the `wsindy_ode`.

We use experimental data taken from [a study by Heinrich et al. (2026)]((https://dataverse.no/dataset.xhtml;jsessionid=a491137f64bab65ab95af677ea7a?persistentId=doi%3A10.18710%2FMRTNPI&version=&q=&fileAccess=&fileTag=&fileSortField=&fileSortOrder=&tagPresort=false&folderPresort=true)
):
```
@data{MRTNPI_2026,
      author = {Kjell Søren Heinrich and Svensson Seth, Douglas and Ehrnstrom, Mats and Ellingsen, Simen Ådnøy},
      publisher = {DataverseNO},
      title = {{Replication Data for: Rediscovering shallow-water equations from experimental data}},
      year = {2026},
      version = {V1},
      doi = {10.18710/MRTNPI},
      url = {https://doi.org/10.18710/MRTNPI}
}
```

###### This algorithm uses the following dependencies:
```python
torch
scipy
numpy
itertools
symengine
tqdm
```

###### Install WSINDy in a Bash environment:
```python3
wget -q https://raw.githubusercontent.com/SethMinor/PyWSINDy-for-PDEs/main/wsindy.py
wget -q https://raw.githubusercontent.com/SethMinor/PyWSINDy-for-PDEs/main/wsindy_ode.py
wget -q https://raw.githubusercontent.com/SethMinor/PyWSINDy-for-PDEs/main/helper_fcns.py
```
